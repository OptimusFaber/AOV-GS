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
from typing import List, Optional

from src.utils.general_utils import InfoPrinter
from src.utils.timer import Timer


class Planner():
    def __init__(self, 
                 main_cfg: mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config)   : configurations
            planner_cfg (mmengine.Config): planner configurations
            info_printer (InfoPrinter)   : information printer
            
        """
        self.main_cfg = main_cfg
        self.planner_cfg = main_cfg.planner
        self.info_printer = info_printer
        self.step = 0

        self.init_timer()

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def init_timer(self):
        """ initialize timer if requested
        Attributes:
            timer (Timer): timer object
            
        """
        self.timer = Timer()
        if self.planner_cfg.get("enable_timing", False):
            self.enable_timing = True
        else:
            self.enable_timing = False
    
    def update_sim(self, sim):
        """ initialize/update a Simulator if requested
        Attributes:
            sim (Simulator): Simulator object
            
        """
        self.sim = sim

    def vox2loc(self, vox, bbox=None, voxel_size=None):
        """ convert voxel coordinates to metric coordinates
    
        Args:
            vox (np.ndarray, [3])   : voxel coordinates
            bbox (np.ndarray, [3,2]): bounding box corner coordinates. Use self.bbox if not provided
            voxel_size (float)      : voxel size. Unit: meter. Use self.bbox if not provided
    
        Returns:
            loc (np.ndarray, [3]): metric coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        loc = vox * voxel_size + bbox[:, 0]
        return loc
    
    def loc2vox(self, loc, bbox=None, voxel_size=None):
        """ convert metric coordinates to voxel coordinates.
    
        Args:
            loc (np.ndarray, [3])   : metric coordinates
            bbox (np.ndarray, [3,2]): bounding box corner coordinates. Use self.bbox if not provided
            voxel_size (float)      : voxel size. Unit: meter. Use self.bbox if not provided
    
        Returns:
            vox (np.ndarray, [3]): voxel coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        vox = (loc - bbox[:, 0]) / voxel_size
        return vox


def compute_camera_pose_RUB(
        A     : np.ndarray,
        B     : np.ndarray,
        up_dir: np.ndarray = np.array([0, 0, 1])
        ) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenGL (RUB) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    # viewing direction (backward)
    V = A - B

    ### FIXME: for edge case that target points in the same x,y position ###
    # if V[0] == 0 and V[1] == 0:
    if (np.cross(V, up_dir)==0).all():
        V[0] = 1e-6

    # right viewing direction
    R = np.cross(up_dir, V)

    # up viewing direction
    U = np.cross(V, R)

    # normalize
    V = V / np.linalg.norm(V)
    R = R / np.linalg.norm(R)
    U = U / np.linalg.norm(U)

    # construct pose matrix
    M = np.column_stack((R, U, V))  

    return M


def compute_camera_pose_RDF(
        A     : np.ndarray,
        B     : np.ndarray,
        up_dir: np.ndarray = np.array([0, -1, 0])
        ) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenCV (RDF) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    # +Z viewing direction (forward)
    V = B - A

    ### FIXME: for edge case that target points in the same x,y position ###
    # if V[0] == 0 and V[1] == 0:
    if (np.cross(V, up_dir)==0).all():
        V[0] = 1e-6

    # +X viewing direction
    R = np.cross(V, up_dir)

    # +Y  direction
    U = np.cross(V, R)

    # normalize
    V = V / np.linalg.norm(V)
    R = R / np.linalg.norm(R)
    U = U / np.linalg.norm(U)

    # construct pose matrix
    M = np.column_stack((R, U, V))  

    return M


def compute_camera_pose(
        A     : np.ndarray,
        B     : np.ndarray,
        up_dir: np.ndarray = np.array([0, 0, 1]),
        system: str = 'RUB'
        ) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenGL (RUB) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
        system                  : coordinate system
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    if system == "RUB":
        return compute_camera_pose_RUB(A, B, up_dir)
    elif system == "RDF":
        return compute_camera_pose_RDF(A, B, up_dir)
    else:
        raise NotImplementedError


def _normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if np.isfinite(n) and n >= 1e-9:
        return v / n
    if fallback is not None:
        return _normalize(fallback, fallback=None)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def sanitize_pose_c2w(pose, fallback=None) -> np.ndarray:
    """Finite 4x4 c2w with a proper rotation (SVD orthonormalization)."""
    pose = np.asarray(pose, dtype=np.float64).copy()
    if fallback is None:
        fallback = np.eye(4, dtype=np.float64)
    else:
        fallback = np.asarray(fallback, dtype=np.float64).copy()

    if not np.isfinite(pose).all():
        pose = fallback.copy() if np.isfinite(fallback).all() else np.eye(4)

    if not np.isfinite(pose[:3, 3]).all():
        pose[:3, 3] = fallback[:3, 3]

    rot = pose[:3, :3]
    if not np.isfinite(rot).all():
        rot = fallback[:3, :3] if np.isfinite(fallback[:3, :3]).all() else np.eye(3)

    u, _, vt = np.linalg.svd(rot)
    rot_ortho = u @ vt
    if float(np.linalg.det(rot_ortho)) < 0.0:
        u[:, -1] *= -1.0
        rot_ortho = u @ vt
    pose[:3, :3] = rot_ortho
    pose[3, :] = [0.0, 0.0, 0.0, 1.0]
    return pose


def horizon_rotation_rdf(
    pos: np.ndarray,
    look_at: np.ndarray,
    R_init_rdf: np.ndarray,
) -> np.ndarray:
    """RDF rotation: fixed Y (down) column from start pose, yaw toward ``look_at``."""
    R0 = np.asarray(R_init_rdf, dtype=np.float64)[:3, :3]
    down = _normalize(R0[:, 1])
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)
    fwd = look_at - pos
    fwd = fwd - np.dot(fwd, down) * down
    fn = float(np.linalg.norm(fwd))
    if fn < 1e-6:
        return R0.copy()
    fwd = fwd / fn
    right = np.cross(down, fwd)
    right = _normalize(right)
    fwd = np.cross(right, down)
    fwd = _normalize(fwd)
    return np.column_stack((right, down, fwd))


def horizon_level_pose_rdf(
    pose: np.ndarray,
    R_init_rdf: np.ndarray,
    look_at: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Level pose to start-pose roll/pitch; optional yaw toward ``look_at``."""
    pose = np.asarray(pose, dtype=np.float64).copy()
    pos = pose[:3, 3]
    if look_at is None:
        look_at = pos + pose[:3, 2]
    pose[:3, :3] = horizon_rotation_rdf(pos, look_at, R_init_rdf)
    return pose


def build_path_poses_look_at_goal(
    end_pose: np.ndarray,
    positions,
    R_init_rdf: np.ndarray,
) -> List[np.ndarray]:
    """Waypoint poses: start roll/pitch, look toward goal while walking."""
    goal = np.asarray(end_pose[:3, 3], dtype=np.float64).reshape(3)

    end_pose = sanitize_pose_c2w(end_pose)
    goal = np.asarray(end_pose[:3, 3], dtype=np.float64).reshape(3)

    if positions is None or len(positions) == 0:
        ep = np.asarray(end_pose, dtype=np.float64).copy()
        return [sanitize_pose_c2w(horizon_level_pose_rdf(ep, R_init_rdf, look_at=None), fallback=end_pose)]

    out: List[np.ndarray] = []
    prev_pose = end_pose
    for pos in positions:
        pos3 = np.asarray(pos, dtype=np.float64).reshape(3)
        if not np.isfinite(pos3).all():
            continue
        wp = np.eye(4, dtype=np.float64)
        wp[:3, 3] = pos3
        leveled = horizon_level_pose_rdf(wp, R_init_rdf, look_at=goal)
        leveled = sanitize_pose_c2w(leveled, fallback=prev_pose)
        out.append(leveled)
        prev_pose = leveled

    ep = np.asarray(end_pose, dtype=np.float64).copy()
    leveled_end = sanitize_pose_c2w(
        horizon_level_pose_rdf(ep, R_init_rdf, look_at=None),
        fallback=prev_pose,
    )
    if len(out) > 0 and np.allclose(out[-1][:3, 3], goal, atol=1e-3):
        out[-1] = leveled_end
    else:
        out.append(leveled_end)
    return out


def yaw_rotation_samples_rdf(R_init_rdf: np.ndarray, K: int) -> np.ndarray:
    """``K`` RDF rotations: yaw around start camera-up, fixed roll/pitch from ``R_init``."""
    from scipy.spatial.transform import Rotation as SciRot

    R0 = np.asarray(R_init_rdf, dtype=np.float64)[:3, :3]
    down = _normalize(R0[:, 1])
    axis = -down
    transforms = np.zeros((1, K, 4, 4), dtype=np.float64)
    for i in range(K):
        angle = 2.0 * np.pi * i / max(K, 1)
        R_yaw = SciRot.from_rotvec(axis * angle).as_matrix() @ R0
        transforms[0, i, :3, :3] = R_yaw
        transforms[0, i, 3, 3] = 1.0
    return transforms

