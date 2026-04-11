import os
from plyfile import PlyData, PlyElement
import numpy as np
def write_ply_file(rgbs, opacities, params, ply_savepath):
    means = params['means3D'].detach().cpu().contiguous().numpy()
    rotations = params['unnorm_rotations'].detach().cpu().contiguous().numpy()
    scales = params['log_scales'].detach().cpu().contiguous().numpy()
    normals = np.zeros_like(means)
    C0 = 0.28209479177387814
    colors = (rgbs - 0.5) / C0
    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))
    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', ]
    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_savepath)

def save_semantic_ply(params,ckpt_output_dir, time_idx):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"semantic_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    rgbs = params['semantic_colors'].detach().cpu().contiguous().numpy()
    opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(rgbs, opacities, params, ply_savepath)