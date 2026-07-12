"""
ActiveGS Hybrid planner v3 for ActiveOpenSem.

Combines:
- SAM+CLIP hybrid exploration (stage 0) from ActiveGSHybridPlanner
- v3 exploration/post-refinement mechanics (caps, revisit, dual stop)
- SAM+CLIP mask coverage for post-refinement semantic stop criterion
"""

from __future__ import annotations

import os
from typing import List, Optional

import mmengine
import numpy as np
import torch
import torch.nn.functional as F

from src.planner.active_gs_planner_v3 import ActiveGSPlannerv3
from src.planner.exploration_semantic_tactic import (
    HybridSemanticExplorationScorer,
    corrclip_kwargs_from_config,
)
from src.utils.general_utils import InfoPrinter

SEMANTIC_PICK_BANNER = """
#######################
#      Semantic selection        #
#######################
"""


class ActiveGSHybridPlannerv3(ActiveGSPlannerv3):
    """Geometry + SAM+CLIP exploration with v3 post-refinement dual stop."""

    def __init__(
        self,
        main_cfg: mmengine.Config,
        info_printer: InfoPrinter,
    ) -> None:
        super().__init__(main_cfg, info_printer)
        sem_cfg = getattr(self.planner_cfg, "semantic_exploration", None)
        if sem_cfg is None:
            sem_cfg = dict(
                enabled_stages=[0],
                novelty_aggregation="mean",
                min_bank_masks=1,
                max_masks_per_candidate=40,
                max_semantic_candidates=8,
            )

        self._sem_enabled_stages = set(sem_cfg.get("enabled_stages", [0]))
        self._sem_scorer: Optional[HybridSemanticExplorationScorer] = None
        self._max_semantic_candidates = int(
            sem_cfg.get("max_semantic_candidates", 8)
        )
        self._log_semantic_scores = sem_cfg.get("log_semantic_scores", True)
        self._post_refine_semantic_enabled = bool(
            sem_cfg.get("post_refinement_semantic", True)
        )
        self._post_refine_max_semantic_kfs = sem_cfg.get(
            "post_refinement_max_keyframes", None
        )
        self._post_refine_semantic_eval_indices: Optional[List[int]] = None

        if self.main_cfg.slam.get("save_clip_features", False):
            lang_dir = os.path.join(
                self.main_cfg.dirs.result_dir, "language_features"
            )
            sam_cfg = self.main_cfg.get("sam_clip", {})
            corrclip = corrclip_kwargs_from_config(sam_cfg)
            self._sem_scorer = HybridSemanticExplorationScorer(
                lang_feat_dir=lang_dir,
                sam_ckpt_path=sam_cfg.get(
                    "sam_ckpt_path", "ckpts/sam_vit_b_01ec64.pth"
                ),
                clip_model=sam_cfg.get("clip_model", "ViT-B-16"),
                clip_pretrained=sam_cfg.get(
                    "clip_pretrained", "laion2b_s34b_b88k"
                ),
                device=sam_cfg.get("device", "cuda:1"),
                max_masks_per_candidate=sem_cfg.get(
                    "max_masks_per_candidate", 40
                ),
                novelty_aggregation=sem_cfg.get("novelty_aggregation", "mean"),
                min_bank_masks=sem_cfg.get("min_bank_masks", 1),
                sam_points_per_side=sem_cfg.get("sam_points_per_side", 32),
                sam_crop_n_layers=sem_cfg.get("sam_crop_n_layers", 1),
                sam_crop_n_points_downscale_factor=sem_cfg.get(
                    "sam_crop_n_points_downscale_factor", 1
                ),
                **corrclip,
            )
            self._sem_scorer.encode_keyframes_if_missing = sem_cfg.get(
                "encode_keyframes_if_missing", True
            )
            corrclip_on = corrclip["corrclip_mask_merge"] or (
                corrclip["corrclip_interclass_suppress_alpha"] > 0.0
            )
            info_printer(
                "ActiveOpenSem: SAM+CLIP exploration stages "
                f"{sorted(self._sem_enabled_stages)}; post-refinement semantic="
                f"{self._post_refine_semantic_enabled}; planner CorrCLIP="
                f"{corrclip_on}.",
                0,
                self.__class__.__name__,
            )
        else:
            info_printer(
                "ActiveOpenSem: save_clip_features=False — semantic scoring disabled.",
                0,
                self.__class__.__name__,
            )

    def use_semantic_exploration(self) -> bool:
        return (
            self._sem_scorer is not None
            and self.exploration_stage in self._sem_enabled_stages
        )

    def maybe_log_semantic_exploration_pick(
        self,
        geo_best_idx: int,
        final_best_idx: int,
        semantic_active: bool,
    ) -> None:
        if semantic_active and geo_best_idx != final_best_idx:
            for line in SEMANTIC_PICK_BANNER.strip().splitlines():
                self.info_printer(line, self.step, self.__class__.__name__)
            self.info_printer(
                f"                            NBV: geometry argmax={geo_best_idx} "
                f"→ semantic argmax={final_best_idx}",
                self.step,
                self.__class__.__name__,
            )

    def _simulate_rgb_uint8(self, cand_pose: torch.Tensor) -> np.ndarray:
        sim_out = self.sim.simulate(
            self.pose_conversion_slam2sim(cand_pose).detach().cpu().numpy(),
            no_print=True,
        )
        color = sim_out["color"]
        if isinstance(color, torch.Tensor):
            arr = color.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.permute(1, 2, 0)
            if arr.dtype != np.uint8:
                return (arr.numpy() * 255).clip(0, 255).astype(np.uint8)
            return arr.numpy()
        return np.asarray(color)

    def _render_rgb_uint8(self, gs_slam, cand_pose: torch.Tensor) -> np.ndarray:
        color = gs_slam.render(cand_pose)[0]
        img = color.permute(1, 2, 0).clamp(0, 1)
        return (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    def score_exploration_semantics(
        self,
        cand_poses: torch.Tensor,
        gs_slam,
        geo_weights: torch.Tensor = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_semantic_exploration():
            return None

        n_cand = cand_poses.shape[0]
        keyframe_ids = [int(kf["id"]) for kf in gs_slam.keyframe_list]

        if geo_weights is not None:
            k = min(self._max_semantic_candidates, n_cand)
            top_idx = torch.topk(geo_weights, k).indices.tolist()
        else:
            top_idx = list(range(min(self._max_semantic_candidates, n_cand)))

        if self._log_semantic_scores:
            self.info_printer(
                f"                            Semantic START    : top-{len(top_idx)} "
                f"of {n_cand} candidates (SAM+CLIP per view ~5-10s)",
                self.step,
                self.__class__.__name__,
            )

        bank_masks = self._sem_scorer.refresh_bank(
            keyframe_ids, keyframe_list=gs_slam.keyframe_list
        )
        if bank_masks < self._sem_scorer.min_bank_masks:
            if self._log_semantic_scores:
                self.info_printer(
                    f"                            Semantic SKIP     : bank={bank_masks} masks "
                    f"(bank_too_small) — geometry only",
                    self.step,
                    self.__class__.__name__,
                )
            return None

        raw_novelties: List[float] = []
        for j, idx in enumerate(top_idx):
            if self._log_semantic_scores:
                self.info_printer(
                    f"                            Semantic [{j + 1}/{len(top_idx)}] "
                    f"pool_idx={idx} — SAM+CLIP…",
                    self.step,
                    self.__class__.__name__,
                )
            rgb = self._simulate_rgb_uint8(cand_poses[idx])
            nov, ran_sam, n_masks, sam_sec = self._sem_scorer.score_candidate_rgb(
                rgb
            )
            if self._log_semantic_scores and ran_sam:
                self.info_printer(
                    f"                            Semantic [{j + 1}/{len(top_idx)}] "
                    f"novelty={nov:.3f}, masks={n_masks}, {sam_sec:.1f}s",
                    self.step,
                    self.__class__.__name__,
                )
            raw_novelties.append(nov)

        sem_weights = torch.ones(n_cand, dtype=torch.float32)
        sub_sm = F.softmax(
            torch.log(torch.tensor(raw_novelties, dtype=torch.float32) + 1e-6),
            dim=0,
        )
        for local_i, pool_idx in enumerate(top_idx):
            sem_weights[pool_idx] = sub_sm[local_i]
        return sem_weights

    def compute_post_refinement_seman_igs(
        self,
        cand_data,
        cand_keys,
        cand_poses: torch.Tensor,
        gs_slam,
        color_igs: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if (
            not self._post_refine_semantic_enabled
            or self._sem_scorer is None
        ):
            return None

        indices = list(range(len(cand_keys)))
        max_kfs = self._post_refine_max_semantic_kfs
        if (
            max_kfs is not None
            and color_igs is not None
            and len(indices) > int(max_kfs)
        ):
            sorted_idx = torch.argsort(color_igs).tolist()
            indices = sorted_idx[: int(max_kfs)]

        self._post_refine_semantic_eval_indices = indices

        scores: List[float] = []
        score_by_idx = {}
        total_sam_sec = 0.0

        self.info_printer(
            f"Post-refinement semantic eval on {len(indices)}/{len(cand_keys)} "
            f"global keyframes (SAM+CLIP coverage)…",
            self.step,
            self.__class__.__name__,
        )

        for j, i in enumerate(indices):
            kf_id = int(cand_keys[i])
            rgb = self._render_rgb_uint8(gs_slam, cand_poses[i])
            coverage, ran_sam, sam_sec = (
                self._sem_scorer.score_keyframe_render_coverage(kf_id, rgb)
            )
            if ran_sam:
                total_sam_sec += sam_sec
            score_by_idx[i] = coverage
            if self._log_semantic_scores:
                self.info_printer(
                    f"  Semantic KF [{j + 1}/{len(indices)}] id={kf_id} "
                    f"coverage={coverage:.3f}, {sam_sec:.1f}s",
                    self.step,
                    self.__class__.__name__,
                )

        for i in range(len(cand_keys)):
            scores.append(float(score_by_idx.get(i, 0.0)))

        self.info_printer(
            f"Post-refinement semantic eval done ({total_sam_sec:.1f}s total).",
            self.step,
            self.__class__.__name__,
        )
        return torch.tensor(scores, dtype=torch.float32)

    def _post_refinement_quality_met(
        self,
        color_igs: torch.Tensor,
        seman_igs: Optional[torch.Tensor],
    ) -> bool:
        color_thre = self.main_cfg.slam.global_keyframe.color_thre
        color_req = (color_igs > color_thre).sum() / len(color_igs) > 0.9
        if seman_igs is None:
            return bool(color_req)

        eval_idx = self._post_refine_semantic_eval_indices
        if eval_idx is not None and len(eval_idx) < len(seman_igs):
            sem = seman_igs[eval_idx]
        else:
            sem = seman_igs
        seman_req = (sem > self._get_seman_thre()).sum() / len(sem) > 0.9
        return bool(color_req and seman_req)
