"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import cv2
import mmengine
import numpy as np
import os
import torch
from typing import List

from src.planner.active_lang_planner import ActiveLangPlanner
from src.slam.splatam.splatam import SplatamOurs as Splatam

from src.visualization.visualizer import Visualizer
from src.utils.display_utils import has_gui_display
from src.utils.general_utils import InfoPrinter, create_class_colormap, apply_colormap

from typing import Union
from third_parties.coslam.utils import colormap_image


class ActiveLangVisualizer(Visualizer):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            vis_cfg (mmengine.Config) : visualizer model configurations
            info_printer (InfoPrinter): information printer
            
        """
        super(ActiveLangVisualizer, self).__init__(main_cfg, info_printer)

        keys = ['rgbd', 'pose', 'planning_path', 'lookat_tgts', 'state', 'information_gain', 'rendered_rgbd']
        ### create directory ###
        for key in keys:
            if self.vis_cfg.get(f"save_{key}", False):
                vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", key)
                os.makedirs(vis_dir, exist_ok=True)

        ### write remark ###
        os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization"), exist_ok=True)
        with open(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "README.md"), 'w') as f:
            f.writelines("rgbd: GT RGB-D visualization\n")
            f.writelines("render_rgbd: render RGB-D visualization\n")
            f.writelines("poses: (np.ndarray, [4,4]). Camera-to-world. RUB system. \n")
            f.writelines("planning_path: (np.ndarray, [N,4,4]), each element is a planning pose \n")
            f.writelines("lookat_tgts: (np.ndarray, [N,3]), uncertaint target observation locations to lookat.  \n")
            f.writelines("state: (str), planner state \n")
            

    def main(self,
             slam           : Splatam,
             planner        : ActiveLangPlanner,
             color          : torch.Tensor,
             depth          : torch.Tensor,
             im              :torch.Tensor,
             rastered_depth  : torch.Tensor,
             pose            : torch.Tensor,
             ) -> None:
        """ save data for visualization purpose

        Args:
            slam           : SLAM module
            planner        : Planner module
            color          : [H,W,3], color image. Range  : 0-1
            depth          : [H,W,3], depth image.
            im           : [H,W,3], rendered color image. Range  : 0-1
            rastered_depth         : [H,W,3], rastered depth image.
            pose           : [4,4],   current pose. Format: camera-to-world, RUB system

        Returns:


        Attributes:

        """
        ### GT RGB-D ###
        if self.vis_cfg.get('save_rgbd', False):
            self.info_printer("Saving RGBD for visualization", self.step, self.__class__.__name__)
            self.save_rgbd(color, depth)

        ### render RGB-D ###
        if self.vis_cfg.get('save_rendered_rgbd', False):
            self.info_printer("Saving rendered RGBD for visualization", self.step, self.__class__.__name__)
            self.save_render_rgbd(im,rastered_depth)

        ### pose ###
        if self.vis_cfg.get('save_pose', False):
            self.info_printer("Saving pose for visualization", self.step, self.__class__.__name__)
            pose_np = pose.detach().cpu().numpy()
            self.save_pose(pose_np)

        ### state ###
        if self.vis_cfg.get('save_state', False):
            self.info_printer("Saving state for visualization", self.step, self.__class__.__name__)
            self.save_state(planner)

        # if self.step > 0:
        #     ### planning_path ###
        #     if self.vis_cfg.save_planning_path:
        #         self.info_printer("Saving planning_path for visualization", self.step, self.__class__.__name__)
        #         self.save_planning_path(planner)
        #
        #     ### lookat_tgt ###
        #     if self.vis_cfg.save_lookat_tgts:
        #         self.info_printer("Saving lookat_tgt for visualization", self.step, self.__class__.__name__)
        #         self.save_lookat_tgt(planner)

           #### information gain ##########
            # if self.vis_cfg.save_information_gain:
            #     self.info_printer("Saving state for visualization", self.step, self.__class__.__name__)
            #     self.save_igs(planner)

    def save_rgbd(self, 
                  color: torch.Tensor, 
                  depth: torch.Tensor
                  ) -> None:
        """save RGB-D visualization
    
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
    
        """
        rgbd_vis = self.visualize_rgbd(color, depth, return_vis=True)
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "rgbd", f"{self.step:04}.png")
        rgbd_vis = (rgbd_vis * 255).astype(np.uint8)
        cv2.imwrite(filepath, rgbd_vis)

    def save_render_rgbd(self, im: torch.Tensor,
                  rastered_depth: torch.Tensor):
        ### save params and render result ###
        rgbd_vis = self.visualize_rgbd(im, rastered_depth, return_vis=True)
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "rendered_rgbd", f"{self.step:04}.png")
        rgbd_vis = (rgbd_vis * 255).astype(np.uint8)
        cv2.imwrite(filepath, rgbd_vis)


    def save_pose(self, pose: np.ndarray) -> None:
        """ save pose
    
        Args:
            pose: [4,4], current pose. Format: camera-to-world, RUB system
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "pose", f"{self.step:04}.npy")
        np.save(filepath, pose)
    
    def save_planning_path(self, planner: ActiveLangPlanner) -> None:
        """ save planning path as np.ndarray (Nx3)
    
        Args:
            planner: Planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "planning_path", f"{self.step:04}.npy")
        if planner.path is not None:
            ### path (List) : each element is a pose [GoalNode, ..., CurrentNode] ###
            # path_locs = [planner.vox2loc(node._xyz_arr) for node in planner.path]
            # path_locs = np.asarray(path_locs)
            if isinstance(planner.path, torch.Tensor):
                path_locs = [pose.cpu() for pose in planner.path]
            else:
                path_locs = planner.path
            path_locs = np.asarray(path_locs)
            np.save(filepath, path_locs)
        else:
            np.save(filepath, None)
    
    def save_lookat_tgt(self, planner: ActiveLangPlanner) -> None:
        """ save lookat targets (uncertain target observations) as np.ndarray (Nx3)
    
        Args:
            planner: planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "lookat_tgts", f"{self.step:04}.npy")
        if planner.lookat_tgts is not None:
            ### lookat_tgts (List)      : uncertaint target observation locations to lookat. each element is (np.ndarray, [3]) ###
            lookat_tgt_locs = np.asarray(planner.lookat_tgts)
            np.save(filepath, lookat_tgt_locs)
        else:
            np.save(filepath, None)

    def save_state(self, planner: ActiveLangPlanner) -> None:
        """ save planner state
    
        Args:
            planner: planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "state", f"{self.step:04}.txt")
        with open(filepath, 'w') as f:
            f.writelines(f"{planner.state}")

    def visualize_rgbd_w_render(self,
                       rgb       : torch.Tensor,
                       depth     : torch.Tensor,
                       im        : torch.Tensor,
                       rastered_depth: torch.Tensor,
                       max_depth : float = 100.,
                       vis_size  : int = 320,
                       return_vis: bool = False
                       ) -> Union[None, np.ndarray]:
        """ visualiz RGB-D
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
            max_depth (float)          : maximum depth value
            vis_size (int)             : image size used for visualization
            return_vis (bool)          : return visualization (OpenCV format) if True

        Returns:
            Union:
                - image (np.ndarray, [H,W,3]): RGB-D visualization if return_vis
        """
        ## process RGB ##
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (vis_size, vis_size))

        im = cv2.cvtColor(im.cpu().numpy(), cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (vis_size, vis_size))


        ### process Depth map ###
        depth = depth.unsqueeze(0)
        mask = (depth < max_depth) * 1.0
        depth_colormap = colormap_image(depth, mask)
        depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
        depth_colormap = cv2.resize(depth_colormap, (vis_size, vis_size))

        rastered_depth = rastered_depth.unsqueeze(0)
        mask = (rastered_depth < max_depth) * 1.0
        rastered_depth_colormap = colormap_image(rastered_depth, mask)
        rastered_depth_colormap = rastered_depth_colormap.permute(1, 2, 0).cpu().numpy()
        rastered_depth_colormap = cv2.resize(rastered_depth_colormap, (vis_size, vis_size))


        ### display RGB-D ###
        image_gt = np.hstack((rgb, depth_colormap))
        image_render = np.hstack((im, rastered_depth_colormap))
        image = np.vstack((image_gt,image_render))


        ### return visualization ###
        if return_vis:
            return image
        if has_gui_display():
            cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB-D', image)
            cv2.waitKey(1)
        return None

    def visualize_semantic(self,
                           gt_semantic: torch.Tensor,
                           pred_semantic: torch.Tensor,
                           n_cls: int,
                           vis_size: int = 320,
                           ) -> None:
        """Show two panels: GT semantic (left) and predicted semantic (right).

        Args:
            gt_semantic: [H,W] int tensor, ground-truth class ids
            pred_semantic: [H,W] int tensor, predicted class ids
            n_cls: number of classes (for colormap)
            vis_size: display size
        """
        colormap = create_class_colormap(max(n_cls, 2))
        gt = gt_semantic.detach().cpu().numpy().astype(np.int64)
        pred = pred_semantic.detach().cpu().numpy().astype(np.int64)
        gt = np.clip(gt, 0, len(colormap) - 1)
        pred = np.clip(pred, 0, len(colormap) - 1)
        gt_rgb = apply_colormap(gt, colormap)
        pred_rgb = apply_colormap(pred, colormap)
        gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
        pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        gt_rgb = cv2.resize(gt_rgb, (vis_size, vis_size))
        pred_rgb = cv2.resize(pred_rgb, (vis_size, vis_size))
        image = np.hstack((gt_rgb, pred_rgb))
        if has_gui_display():
            cv2.namedWindow('Semantic (GT | Pred)', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Semantic (GT | Pred)', image)
            cv2.waitKey(1)

    # def save_igs(self, planner:ActiveLangPlanner):
    #     if planner.state == "planning" and "exploration" in planner.planning_state:
    #         igs = []
    #         breakpoint()
    #         for key, val in planner.explore_pool.items():
    #             igs.append(val['ig'].detach().cpu().numpy())
    #         igs = np.asarray(igs)
    #         filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "information_gain", f"{self.step:04}.npy")
    #         np.save(filepath, igs)
        
    # def save_color_mesh(self, slam: CoSLAM) -> None:
    #     """ save colored mesh
    
    #     Args:
    #         slam: SLAM module
    #     """
    #     if self.step % self.vis_cfg.save_mesh_freq == 0:
    #         mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "color_mesh")
    #         slam.save_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
    #     else:
    #         return
    
    # def save_uncert_mesh(self, slam: CoSLAM) -> None:
    #     """ save uncertainty mesh
    
    #     Args:
    #         slam: SLAM module
    #     """
    #     if self.step % self.vis_cfg.save_mesh_freq == 0:
    #         mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "uncert_mesh")
    #         slam.save_uncert_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
    #     else:
    #         return