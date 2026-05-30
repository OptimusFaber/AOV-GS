"""
ActiveGS planner v3 — geometry NBV hooks (v1) + exploration/post-refinement mechanics (v2).

Intended as base for ActiveOpenSem: capped post-refinement with dual stop
(budget OR color+semantic quality on global keyframes).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import mmengine
import torch

from src.planner.active_gs_planner import ActiveGSPlanner
from src.utils.general_utils import InfoPrinter
from third_parties.splatam.utils.slam_external import calc_psnr


class ActiveGSPlannerv3(ActiveGSPlanner):
    """active_gs + v2 exploration limits, revisit cap, and post-refinement dual stop."""

    def __init__(
        self,
        main_cfg: mmengine.Config,
        info_printer: InfoPrinter,
    ) -> None:
        super().__init__(main_cfg, info_printer)
        self.post_refine_counter = 0
        self._explore_pool_ever_populated = False
        self.max_revisit_count = int(
            getattr(self.planner_cfg, "max_revisit_count", 3)
        )

    def init_data(self, sim2slam: torch.Tensor) -> None:
        super().init_data(sim2slam)
        self.max_exploration_steps = int(
            self.planner_cfg.get("max_exploration_steps", 1000)
        )

    def update_explore_pool_cand(
        self,
        explore_igs: torch.Tensor,
        cand_keys: List[Tuple],
        next_visit: int,
    ) -> None:
        for i in range(len(explore_igs)):
            self.explore_pool[cand_keys[i]]["ig"] = explore_igs[i]
            if "visit" not in self.explore_pool[cand_keys[i]]:
                self.explore_pool[cand_keys[i]]["visit"] = 0
        self.explore_pool[cand_keys[next_visit]]["visit"] += 1

    def del_explore_pool_cand(
        self,
        explore_igs: torch.Tensor,
        cand_keys: List[Tuple],
        explore_thre: float,
    ) -> None:
        explore_mask = explore_igs < (self.img_h * self.img_w) * explore_thre
        revisit = torch.tensor(
            [self.explore_pool[cand]["visit"] for cand in cand_keys]
        ).to(explore_mask.device)
        visit_mask = revisit > self.max_revisit_count
        rm_idx = torch.where(explore_mask | visit_mask)[0]
        for i in rm_idx:
            del self.explore_pool[cand_keys[i]]

    def _is_exploration_done(self) -> bool:
        if len(self.explore_pool) > 0:
            self._explore_pool_ever_populated = True
        done = (
            len(self.explore_pool) == 0 and self._explore_pool_ever_populated
        )
        if self.step > self.max_exploration_steps:
            done = True
            self.info_printer(
                f"Current state: {self.state} | {self.planning_state}: "
                f"Run out maximum exploration steps - {self.exploration_stage} , "
                f"starting evaluation...",
                self.step,
                self.__class__.__name__,
            )
        return done

    def select_exploration_pose_and_idx(
        self,
        cand_poses: torch.Tensor,
        explore_igs_sm: torch.Tensor,
        dists_sm: torch.Tensor,
        gs_slam,
    ) -> Tuple[torch.Tensor, int]:
        geo_weights = (1 - dists_sm) * explore_igs_sm
        geo_best_idx = int(torch.argmax(geo_weights).item())

        sem_weights = self.score_exploration_semantics(
            cand_poses, gs_slam, geo_weights=geo_weights
        )
        semantic_active = sem_weights is not None
        if semantic_active:
            final_weights = geo_weights * sem_weights.to(geo_weights.device)
            self.info_printer(
                f"                            Semantic Novelty : {sem_weights.to(geo_weights.device)}",
                self.step,
                self.__class__.__name__,
            )
        else:
            final_weights = geo_weights

        final_best_idx = int(torch.argmax(final_weights).item())
        self.maybe_log_semantic_exploration_pick(
            geo_best_idx, final_best_idx, semantic_active
        )
        return cand_poses[final_best_idx], final_best_idx

    def score_exploration_semantics(
        self,
        cand_poses: torch.Tensor,
        gs_slam,
        geo_weights: torch.Tensor = None,
    ):
        return None

    def maybe_log_semantic_exploration_pick(
        self,
        geo_best_idx: int,
        final_best_idx: int,
        semantic_active: bool,
    ) -> None:
        pass

    def _get_seman_thre(self) -> float:
        if hasattr(self.planner_cfg, "seman_thre"):
            return float(self.planner_cfg.seman_thre)
        gkf = getattr(self.main_cfg.slam, "global_keyframe", None)
        if gkf is not None and hasattr(gkf, "seman_thre"):
            return float(gkf.seman_thre)
        return 0.9

    def compute_post_refinement_seman_igs(
        self,
        cand_data,
        cand_keys,
        cand_poses: torch.Tensor,
        gs_slam,
        color_igs: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Override in hybrid v3 for SAM+CLIP coverage. None → color-only stop."""
        return None

    def _post_refinement_quality_met(
        self,
        color_igs: torch.Tensor,
        seman_igs: Optional[torch.Tensor],
    ) -> bool:
        color_thre = self.main_cfg.slam.global_keyframe.color_thre
        color_req = (color_igs > color_thre).sum() / len(color_igs) > 0.9
        if seman_igs is None:
            return bool(color_req)
        seman_req = (
            (seman_igs > self._get_seman_thre()).sum() / len(seman_igs) > 0.9
        )
        return bool(color_req and seman_req)

    def rendering_based_planning(self, cur_pose, gs_slam):
        new_pose = cur_pose

        if (
            self.planning_state == "exploration"
            and self.exploration_stage < self.num_exploration_stage
        ):
            self.info_printer(
                f"Current state: {self.state} | {self.planning_state}: "
                f"Getting New Exploration Map",
                self.step,
                self.__class__.__name__,
            )

            gs_z_levels = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            new_free_voxels = gs_slam.explr_map.get_new_free_voxels(
                use_xyz_filter=True,
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels,
            )
            new_free_locs_sim = (
                gs_slam.explr_map.origin
                + new_free_voxels * gs_slam.explr_map.voxel_size
            )

            gs_slam.explr_map.update_prev_free_voxels(
                use_xyz_filter=True,
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels,
            )

            if new_free_locs_sim.shape[0] != 0:
                new_cand_poses = self.generate_candidate_poses(
                    new_free_locs_sim,
                    gs_slam.explr_map.sim2slam,
                ).reshape(-1, 4, 4)

                free_vxl_idx_exp = new_free_voxels.unsqueeze(1).repeat(
                    1, self.num_dir_samples[self.exploration_stage], 1
                )
                view_rot_idx_exp = self.view_rot_idx[self.exploration_stage].repeat(
                    new_free_voxels.shape[0], 1, 1
                )
                new_cand_pose_key = torch.cat(
                    [free_vxl_idx_exp, view_rot_idx_exp], dim=-1
                ).reshape(-1, 4)

                self.add_explore_pool_cand(new_cand_poses, new_cand_pose_key)

            is_explore_done = self._is_exploration_done()

            if is_explore_done:
                if self.exploration_stage < self.num_exploration_stage:
                    self.info_printer(
                        f"Current state: {self.state} | {self.planning_state}: "
                        f"Done Exploration Stage - {self.exploration_stage} , "
                        f"starting evaluation...",
                        self.step,
                        self.__class__.__name__,
                    )
                    eval_dir_suffix = f"exploration_stage_{self.exploration_stage}"
                    dataset_len = len(self.gs_slam.dataset_eval)
                    user_max = getattr(
                        self.main_cfg.slam, "eval_during_training_max_frames", None
                    )

                    if (
                        user_max is None
                        or int(user_max) >= dataset_len
                        or int(user_max) == -1
                    ):
                        max_frames = None
                        self.info_printer(
                            f"Evaluating on all {dataset_len} frames",
                            self.step,
                            self.__class__.__name__,
                        )
                    else:
                        max_frames = min(self.step + 1, int(user_max))
                        self.info_printer(
                            f"Evaluating on {max_frames} frames (limited)",
                            self.step,
                            self.__class__.__name__,
                        )

                    self.gs_slam.eval_result(
                        eval_dir_suffix=eval_dir_suffix,
                        ignore_first_frame=True,
                        save_frames=False,
                        max_frames=max_frames,
                    )
                    self.gs_slam.print_and_save_result(
                        eval_dir_suffix=eval_dir_suffix,
                        is_prune_gaussians=False,
                        ignore_first_frame=True,
                    )
                    eval_dir = self.gs_slam.eval_dir + "_" + eval_dir_suffix
                    import os

                    os.makedirs(eval_dir, exist_ok=True)
                    with open(
                        os.path.join(eval_dir, "exploration_info.txt"), "w"
                    ) as f:
                        f.writelines(
                            f"exploration_stage_{self.exploration_stage}_step: {self.step}\n"
                        )
                        f.writelines("global_keyframe: ")
                        if self.main_cfg.slam.use_global_keyframe:
                            f.writelines(
                                f"{sorted(self.gs_slam.global_keyframe_indices)}\n"
                            )

                    self.exploration_stage += 1
                    self.planning_state = "exploration"

                if self.exploration_stage == self.num_exploration_stage:
                    self.info_printer(
                        f"Current state: {self.state} | {self.planning_state}: "
                        f"Done All Exploration.",
                        self.step,
                        self.__class__.__name__,
                    )
                    if self.main_cfg.slam.use_global_keyframe:
                        self.planning_state = "post_refinement"
                        self.post_refine_counter = 0
                    else:
                        self.planning_state = "done"
                else:
                    gs_z_levels = self.gs_z_levels[self.exploration_stage]
                    xy_sampling_step = self.planner_cfg.xy_sampling_step[
                        self.exploration_stage
                    ]
                    gs_slam.explr_map.prev_free_voxels = torch.empty(0, 3).to(
                        self.device
                    )

            else:
                self.info_printer(
                    f"Current state: {self.state} | {self.planning_state}: "
                    f"Evaluate Exploration Candidate I.G.",
                    self.step,
                    self.__class__.__name__,
                )
                self.planning_state = "exploration"
                cand_poses, cand_keys = self.get_explore_pool_poses()
                explore_igs = []

                dists = (
                    torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6
                )
                dists_sm = torch.nn.functional.softmax(dists, dim=0)
                for i, cand_pose in enumerate(cand_poses):
                    _r = gs_slam.render(cand_pose)
                    img, depth, valid_mask = _r[0], _r[1], _r[2]

                    depth_gt = self.sim.simulate(
                        self.pose_conversion_slam2sim(cand_pose)
                        .detach()
                        .cpu()
                        .numpy(),
                        no_print=True,
                    )["depth"]
                    valid_sim_mask = depth_gt > 0.2
                    valid_mask[0][~valid_sim_mask] = True

                    _, self.img_h, self.img_w = img.shape
                    explore_ig = (valid_mask == 0).sum()
                    explore_igs.append(explore_ig)

                explore_igs = torch.stack(explore_igs).float()
                explore_igs_sm = torch.nn.functional.softmax(
                    torch.log(explore_igs), dim=0
                )
                new_pose, next_visit = self.select_exploration_pose_and_idx(
                    cand_poses, explore_igs_sm, dists_sm, gs_slam
                )

                self.update_explore_pool_cand(explore_igs, cand_keys, next_visit)
                self.del_explore_pool_cand(
                    explore_igs, cand_keys, self.planner_cfg.explore_thre
                )
                self.info_printer(
                    f"Current state: {self.state} [Exploration Pool: {len(self.explore_pool)}]",
                    self.step,
                    self.__class__.__name__,
                )
                self.info_printer(
                    f"                            Exploration I.G.   : {explore_igs}",
                    self.step,
                    self.__class__.__name__,
                )

        if self.planning_state == "refinement":
            self.info_printer(
                f"Current state: {self.state} | {self.planning_state}: "
                f"Evaluate Refinement Candidate I.G.",
                self.step,
                self.__class__.__name__,
            )
            self.planning_state = "refinement"
            new_kfs = self.gs_slam.get_new_keyframe_idxs()
            self.gs_slam.update_prev_keyframes()
            selected_kf_list = [
                elem
                for elem, mask in zip(self.gs_slam.keyframe_list, new_kfs)
                if mask
            ]
            self.add_refine_pool_cand(selected_kf_list)

            cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
            color_igs = []
            depth_igs = []

            for i, cand_pose in enumerate(cand_poses):
                color, depth, valid_mask = gs_slam.render(cand_pose)[:3]
                valid_depth_mask = cand_data[i]["depth"] > 0
                color_ig = calc_psnr(
                    color * valid_depth_mask,
                    cand_data[i]["color"] * valid_depth_mask,
                ).mean()
                color_igs.append(color_ig)
                depth_ig = (
                    torch.abs(
                        depth * valid_depth_mask
                        - cand_data[i]["depth"] * valid_depth_mask
                    )
                    / (cand_data[i]["depth"] + 1e-8)
                ).sum() / valid_depth_mask.sum()
                depth_igs.append(depth_ig)

            color_igs = torch.stack(color_igs).float()
            depth_igs = torch.stack(depth_igs).float()
            color_igs_sm = 1 - torch.nn.functional.softmax(color_igs, dim=0)
            refine_igs = color_igs_sm
            best_key = torch.argmax(refine_igs)
            new_pose = cand_poses[best_key]

            self.del_refine_pool_cand(
                color_igs,
                depth_igs,
                cand_keys,
                self.planner_cfg.color_ig_thre,
                self.planner_cfg.depth_ig_thre,
            )
            if len(self.refine_pool) == 0:
                self.planning_state = "post_refinement"
                self.post_refine_counter = 0

        if self.planning_state == "post_refinement":
            self.post_refine_counter += 1
            if self.post_refine_counter >= self.planner_cfg.post_refine_steps:
                self.planning_state = "done"
                self.info_printer(
                    f"Post-refinement budget reached "
                    f"({self.planner_cfg.post_refine_steps} steps).",
                    self.step,
                    self.__class__.__name__,
                )

            if self.step % self.planner_cfg.post_refinement_eval_freq == 0:
                self.info_printer(
                    f"Current state: {self.state} | {self.planning_state}: "
                    f"Evaluate Post-Refinement Candidate I.G.",
                    self.step,
                    self.__class__.__name__,
                )
                self.planning_state = "post_refinement"

                if len(self.refine_pool) == 1:
                    selected_kf_list = [
                        self.gs_slam.keyframe_list[i]
                        for i in self.gs_slam.global_keyframe_indices
                    ]
                    self.add_refine_pool_cand(selected_kf_list)

                cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
                color_igs = []

                for i, cand_pose in enumerate(cand_poses):
                    color, depth, valid_mask = gs_slam.render(cand_pose)[:3]
                    valid_depth_mask = cand_data[i]["depth"] > 0
                    color_ig = calc_psnr(
                        color * valid_depth_mask,
                        cand_data[i]["color"] * valid_depth_mask,
                    ).mean()
                    color_igs.append(color_ig)

                color_igs = torch.stack(color_igs).float()
                seman_igs = self.compute_post_refinement_seman_igs(
                    cand_data, cand_keys, cand_poses, gs_slam, color_igs=color_igs
                )

                self.info_printer(
                    f"Current state: {self.state} [Refinement Pool: {len(self.refine_pool)}]",
                    self.step,
                    self.__class__.__name__,
                )
                self.info_printer(
                    f"Refinement ColorIG: {color_igs}",
                    self.step,
                    self.__class__.__name__,
                )
                self.info_printer(
                    f"Refinement ColorIG [Min, Avg]: "
                    f"[{torch.min(color_igs).item():.2f}, {torch.mean(color_igs).item():.2f}]",
                    self.step,
                    self.__class__.__name__,
                )
                if seman_igs is not None:
                    self.info_printer(
                        f"Refinement SemanticIG: {seman_igs}",
                        self.step,
                        self.__class__.__name__,
                    )
                    self.info_printer(
                        f"Refinement SemanticIG [Min, Avg]: "
                        f"[{torch.min(seman_igs).item():.2f}, {torch.mean(seman_igs).item():.2f}]",
                        self.step,
                        self.__class__.__name__,
                    )

                if self._post_refinement_quality_met(color_igs, seman_igs):
                    self.planning_state = "done"
                    self.info_printer(
                        "Post-refinement quality target met "
                        "(>90% global keyframes pass color + semantic thresholds).",
                        self.step,
                        self.__class__.__name__,
                    )

        if self.planning_state == "done":
            self.planning_state = "done"
            self.info_printer(
                f"Current state: Exploration + Refinement All Done!",
                self.step,
                self.__class__.__name__,
            )

        return dict(new_pose=new_pose)
