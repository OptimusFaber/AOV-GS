import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from third_parties.splatam.datasets.gradslam_datasets.basedataset import GradSLAMDataset
import imageio

class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        load_semantics: Optional[bool] = False,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.load_semantics = load_semantics
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        if self.load_semantics:
            self.semantic_paths = self.get_semantic_filepaths()


    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results_habitat/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results_habitat/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def get_semantic_filepaths(self):
        semantic_paths = None
        if self.load_semantics:
            semantic_paths = natsorted(glob.glob(f"{self.input_folder}/results_habitat/semantic/semantic*.npy"))
        return semantic_paths


    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        num_poses_in_file = len([line for line in lines if line.strip()])  # Count non-empty lines
        
        if num_poses_in_file == 0:
            raise ValueError(f"No poses found in {self.pose_path}")
        
        # Load all poses from file (skip empty lines)
        file_poses = []
        for line in lines:
            if line.strip():  # Skip empty lines
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                # c2w[:3, 1] *= -1
                # c2w[:3, 2] *= -1
                c2w = torch.from_numpy(c2w).float()
                file_poses.append(c2w)
        
        # If we have fewer poses than images, use the last available pose for remaining frames
        # If we have more poses than images, use only the first num_imgs poses
        for i in range(self.num_imgs):
            if i < len(file_poses):
                poses.append(file_poses[i])
            else:
                # Use the last available pose for remaining frames
                if len(file_poses) > 0:
                    poses.append(file_poses[-1])
                else:
                    raise ValueError(f"Not enough poses in {self.pose_path} for {self.num_imgs} images")
        
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

    def read_semantic_from_file(self, semantic_file_path):
        semantic_map = np.load(semantic_file_path)
        semantic_map[semantic_map < 0] = 0
        return semantic_map

    def get_semantic_map(self,index):
        # Handle case when semantic_paths is shorter than dataset length
        # Use the last available semantic map if index is out of range
        if self.semantic_paths is None or len(self.semantic_paths) == 0:
            raise ValueError("Semantic paths are not loaded. Set load_semantics=True in dataset config.")
        
        if index >= len(self.semantic_paths):
            # Use the last available semantic map
            semantic_path = self.semantic_paths[-1]
        else:
            semantic_path = self.semantic_paths[index]
        
        semantics = self.read_semantic_from_file(semantic_path)
        semantics = torch.from_numpy(semantics)
        return semantics.to(self.device).type(self.dtype),

    