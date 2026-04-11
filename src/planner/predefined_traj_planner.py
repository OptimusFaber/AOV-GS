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


import mmengine
import numpy as np
import torch
from typing import List

from src.data.pose_loader import PoseLoader
from src.planner.planner import Planner
from src.utils.general_utils import InfoPrinter

class PreTrajPlanner(Planner):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter,
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer

        Attributes:
            state (str): planner state
        """
        super(PreTrajPlanner, self).__init__(main_cfg, info_printer)
        
        ### initialize planner state ###
        self.state = "staying"

        ### initialize pose loader ###
        self.pose_loader = PoseLoader(main_cfg)

    def load_init_pose(self):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        return self.pose_loader.load_init_pose()
    
    def update_pose(self, c2w, i):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        return self.pose_loader.update_pose(c2w, i)

    def init_data(self,
                  sim2slam: torch.Tensor
                  ) -> None:
        """initialize data for naruto planner

        Args:
            # bbox (List, [3,2]): bounding box corners coordinates

        Attributes:
            sim2slam (torch.Tensor): simator-to-slam conversion
        """
        # self.path = None
        # self.lookat_tgts = None
        self.sim2slam = sim2slam

        # Accept both list and np.ndarray in config
        up_dir_cfg = self.planner_cfg.up_dir
        if isinstance(up_dir_cfg, list):
            up_dir_sim = torch.from_numpy(np.array(up_dir_cfg)).float().to(self.sim2slam.device).unsqueeze(1)
        else:
            up_dir_sim = torch.from_numpy(up_dir_cfg).float().to(self.sim2slam.device).unsqueeze(1)
        self.up_dir_slam = (self.sim2slam[:3, :3] @ up_dir_sim)[:, 0].cpu().numpy()


    def main(self, 
             uncert_sdf_vols: List,
             cur_pose       : np.ndarray,
             is_new_vols    : bool
             ) -> torch.Tensor:
        """ Naruto Planner main function
    
        Args:
            uncert_sdf_vols (List)      : Uncertainty Volume and SDF Volume
                - uncert_vol (np.ndarray, [X,Y,Z]): uncertainty volume
                - sdf_vol (np.ndarray, [X,Y,Z])   : SDF volume
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            is_new_vols (bool)          : is uncert_sdf_vols new optimized volumes
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        self.update_state(uncert_sdf_vols[1], cur_pose, is_new_vols)
        self.info_printer(f"Current state: {self.state}", self.step, self.__class__.__name__)
        new_pose = self.compute_next_state_pose(cur_pose, uncert_sdf_vols)
        new_pose = torch.from_numpy(new_pose).float()
        return new_pose
