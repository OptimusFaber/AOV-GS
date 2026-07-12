"""
ActiveGS planner with hybrid exploration on stage 0:
geometry (unexplored pixels) + SAM+CLIP embedding novelty.

Stage 1+ uses pure geometry, same as ActiveGSPlanner.
"""

from __future__ import annotations

import os
from typing import List, Optional

import mmengine
import numpy as np
import torch
import torch.nn.functional as F

from src.planner.active_gs_planner import ActiveGSPlanner
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


class ActiveGSHybridPlanner(ActiveGSPlanner):
    """Geometry + SAM+CLIP semantic novelty on the first exploration stage."""

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
        self._max_semantic_candidates = int(sem_cfg.get("max_semantic_candidates", 8))
        self._log_semantic_scores = sem_cfg.get("log_semantic_scores", True)

        if self.main_cfg.slam.get("save_clip_features", False):
            lang_dir = os.path.join(self.main_cfg.dirs.result_dir, "language_features")
            sam_cfg = self.main_cfg.get("sam_clip", {})
            corrclip = corrclip_kwargs_from_config(sam_cfg)
            self._sem_scorer = HybridSemanticExplorationScorer(
                lang_feat_dir=lang_dir,
                sam_ckpt_path=sam_cfg.get("sam_ckpt_path", "ckpts/sam_vit_b_01ec64.pth"),
                clip_model=sam_cfg.get("clip_model", "ViT-B-16"),
                clip_pretrained=sam_cfg.get("clip_pretrained", "laion2b_s34b_b88k"),
                device=sam_cfg.get("device", "cuda:1"),
                max_masks_per_candidate=sem_cfg.get("max_masks_per_candidate", 40),
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
                "Hybrid semantic exploration enabled for stages "
                f"{sorted(self._sem_enabled_stages)} (SAM+CLIP embedding novelty, "
                f"top-{self._max_semantic_candidates} candidates per step, "
                f"planner CorrCLIP={corrclip_on}).",
                0,
                self.__class__.__name__,
            )
        else:
            info_printer(
                "Hybrid planner: save_clip_features=False, semantic scoring disabled.",
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

    def score_exploration_semantics(
        self,
        cand_poses: torch.Tensor,
        gs_slam,
        geo_weights: torch.Tensor = None,
    ) -> Optional[torch.Tensor]:
        """
        Semantic weights only for geometry top-K (others = 1.0).
        """
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

        # First the keyframe embedding bank (no SAM on candidates).
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
            self._sem_scorer.last_stats = {
                "bank_masks": bank_masks,
                "n_candidates": n_cand,
                "semantic_active": False,
                "reason": "bank_too_small",
            }
            return None

        raw_novelties: List[float] = []
        total_sam_sec = 0.0
        total_masks = 0
        for j, idx in enumerate(top_idx):
            if self._log_semantic_scores:
                self.info_printer(
                    f"                            Semantic [{j + 1}/{len(top_idx)}] "
                    f"pool_idx={idx} — SAM+CLIP…",
                    self.step,
                    self.__class__.__name__,
                )
            rgb = self._simulate_rgb_uint8(cand_poses[idx])
            nov, ran_sam, n_masks, sam_sec = self._sem_scorer.score_candidate_rgb(rgb)
            if self._log_semantic_scores and ran_sam:
                self.info_printer(
                    f"                            Semantic [{j + 1}/{len(top_idx)}] "
                    f"novelty={nov:.3f}, masks={n_masks}, {sam_sec:.1f}s",
                    self.step,
                    self.__class__.__name__,
                )
            raw_novelties.append(nov)
            if ran_sam:
                total_sam_sec += sam_sec
                total_masks += n_masks

        self._sem_scorer._fill_last_stats(
            bank_masks=bank_masks,
            n_candidates=n_cand,
            n_sam_calls=len(top_idx),
            total_cand_masks=total_masks,
            bank_sec=0.0,
            sam_sec=total_sam_sec,
            scores=raw_novelties,
        )
        stats = self._sem_scorer.last_stats

        sem_weights = torch.ones(n_cand, dtype=torch.float32)
        sub_sm = F.softmax(
            torch.log(torch.tensor(raw_novelties, dtype=torch.float32) + 1e-6),
            dim=0,
        )
        for local_i, pool_idx in enumerate(top_idx):
            sem_weights[pool_idx] = sub_sm[local_i]

        if self._log_semantic_scores:
            self.info_printer(
                f"                            Semantic bank     : {stats['bank_masks']} masks, "
                f"SAM on {stats['n_sam_calls']}/{len(top_idx)} cands "
                f"({stats['total_cand_masks']} mask embs, {stats['sam_sec']:.1f}s)",
                self.step,
                self.__class__.__name__,
            )
            self.info_printer(
                f"                            Semantic novelty  : min={stats['novelty_min']:.3f} "
                f"max={stats['novelty_max']:.3f} mean={stats['novelty_mean']:.3f}",
                self.step,
                self.__class__.__name__,
            )
        return sem_weights
