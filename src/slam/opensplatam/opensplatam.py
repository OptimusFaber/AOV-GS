"""
OpenSplatam — SplaTAM-style RGB-D mapping with **simulator-first** frame-0 init.

Mirrors :class:`SemSplatam`'s simulator-aligned initialisation, but without
OneFormer / semantic heads. Pure geometry SplaTAM, seeded from the first live
``HabitatSim.simulate`` frame at ``traj.txt`` pose #0. No extra vertical flips,
no disk-warping — the Gaussian world lives in whatever convention the simulator
hands us, exactly like ``SemSplatam``.

Only :meth:`init_camera_parameters_from_simulator` and :meth:`online_recon_step`
are overridden w.r.t. :class:`SplatamOurs`; mapping, densification, keyframe
selection, eval and checkpointing are inherited unchanged.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import mmengine
import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.slam.splatam.splatam import SplatamOurs
from src.utils.general_utils import InfoPrinter

sys.path.append("third_parties/splatam")
from utils.common_utils import save_params_ckpt
from utils.recon_helpers import setup_camera

from src.slam.splatam.modified_ver.scripts.splatam import (
    get_pointcloud,
    initialize_params,
)
from src.slam.semsplatam.modified_ver.splatam.export_helper import save_rgb_ply


class OpenSplatam(SplatamOurs):
    """SplaTAM with simulator-aligned frame-0 init (no semantics).

    Configure with ``slam.method = "opensplatam"``. All other SplaTAM options
    (tracking, densification, eval, etc.) work unchanged.
    """

    def __init__(
        self,
        main_cfg: mmengine.Config,
        info_printer: InfoPrinter,
        logger: SummaryWriter,
    ) -> None:
        super().__init__(main_cfg, info_printer, logger)

    # ------------------------------------------------------------------ init
    def init_camera_parameters_from_simulator(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        c2w: torch.Tensor,
    ) -> None:
        """Seed Gaussians + rasterizer cameras from the first live simulator frame.

        Mirrors :meth:`SemSplatam.init_camera_parameters_from_simulator` modulo
        the semantic branch.
        """
        # Intrinsics from dataset metadata only; do not touch dataset frame 0.
        intrinsics = self._get_scaled_camera_intrinsics()

        color_processed = color.permute(2, 0, 1).to(self.device) / 255.0
        depth_processed = depth.unsqueeze(0).to(self.device)

        H, W = color_processed.shape[1], color_processed.shape[2]

        w2c = torch.linalg.inv(c2w.to(self.device))

        cam = setup_camera(W, H, intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

        mask = (depth_processed > 0).reshape(-1)
        init_pt_cld, mean3_sq_dist = get_pointcloud(
            color_processed, depth_processed, intrinsics, w2c,
            mask=mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=self.config['mean_sq_dist_method'],
        )
        params, variables = initialize_params(
            init_pt_cld, self.num_frames, mean3_sq_dist,
            self.config['gaussian_distribution'],
        )
        variables['scene_radius'] = torch.max(depth_processed) / self.config['scene_radius_depth_ratio']

        print(f"✅ Initialized {params['means3D'].shape[0]:,} Gaussian points from simulator")

        # Densification camera setup (matches semsplatam logic).
        dataset_config = self.config["data"]
        if "densification_image_height" not in dataset_config:
            self.seperate_densification_res = False
            self.densify_intrinsics = intrinsics
            self.densify_cam = cam
        else:
            if dataset_config["densification_image_height"] != H or dataset_config["densification_image_width"] != W:
                self.seperate_densification_res = True
                densify_h = dataset_config["densification_image_height"]
                densify_w = dataset_config["densification_image_width"]
                densify_fx = intrinsics[0, 0] * densify_w / W
                densify_fy = intrinsics[1, 1] * densify_h / H
                densify_cx = intrinsics[0, 2] * densify_w / W
                densify_cy = intrinsics[1, 2] * densify_h / H
                densify_intrinsics = torch.tensor([
                    [densify_fx, 0, densify_cx],
                    [0, densify_fy, densify_cy],
                    [0, 0, 1],
                ], device=self.device, dtype=torch.float32)
                self.densify_intrinsics = densify_intrinsics
                self.densify_cam = setup_camera(
                    densify_w, densify_h,
                    densify_intrinsics.cpu().numpy(),
                    w2c.detach().cpu().numpy(),
                )
            else:
                self.seperate_densification_res = False
                self.densify_intrinsics = intrinsics
                self.densify_cam = cam

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = w2c
        self.cam = cam

        if self.seperate_tracking_res:
            self.tracking_cam = setup_camera(
                self.tracking_color.shape[2], self.tracking_color.shape[1],
                self.tracking_intrinsics.cpu().numpy(),
                w2c.detach().cpu().numpy(),
            )

    # --------------------------------------------------------------- online loop
    def online_recon_step(
        self,
        time_idx: int,
        color: torch.Tensor,
        depth: torch.Tensor,
        c2w: torch.Tensor,
        force_map_update: bool = False,
        dont_add_kf: bool = False,
        only_use_global_keyframe: bool = False,
        keyframes_extra_subdir: Optional[str] = None,
    ) -> List:
        """Identical to ``SplatamOurs.online_recon_step`` but seeds from simulator at t=0."""
        if time_idx == 0:
            self.init_camera_parameters_from_simulator(color, depth, c2w)
        self.update_gs_map(
            time_idx, color, depth, c2w,
            force_map_update, dont_add_kf, only_use_global_keyframe,
        )
        self.update_explr_map(time_idx, depth, c2w, force_map_update)

    # ------------------------------------------------------------ checkpointing
    def save_checkpoint(self, eval_dir_suffix: str) -> None:
        """Save SplaTAM params + coloured PLY (no semantic channel)."""
        params = self.params
        variables = self.variables
        keyframe_time_indices = self.keyframe_time_indices
        ckpt_dir = os.path.join(self.results_dir, eval_dir_suffix)
        os.makedirs(ckpt_dir, exist_ok=True)
        _idx = 0
        save_params_ckpt(params, variables, ckpt_dir, _idx)
        save_rgb_ply(params, ckpt_dir, _idx)
        np.save(
            os.path.join(ckpt_dir, "keyframe_time_indices.npy"),
            np.array(keyframe_time_indices),
        )
        print(f"[OpenSplatam checkpoint] Saved to: {ckpt_dir}")
