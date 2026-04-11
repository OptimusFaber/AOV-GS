import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
from typing import Dict, List
import open3d as o3d

from imgviz import label_colormap
import trimesh

from src.data.pose_loader import habitat_pose_conversion, PoseLoader
from src.naruto.cfg_loader import load_cfg
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer
import argparse
from plyfile import PlyData, PlyElement
from src.utils.general_utils import create_class_colormap,apply_colormap
import numpy as np
import torch
from src.layers.c2e import C2E
import py360convert
import json
from src.utils.config_utils import load_config
import matplotlib.pyplot as plt
import cv2

sys.path.append("third_parties/splatam")
from third_parties.splatam.utils.slam_helpers import (
transformed_params2rendervar,
 transformed_params2depthplussilhouette,
)
from src.slam.splatam.eval_helper import transform_to_frame
from src.slam.semsplatam.modified_ver.splatam.splatam import calc_shannon_entropy
from third_parties.splatam.utils.slam_helpers import matrix_to_quaternion

from src.slam.semsplatam.modified_ver.splatam.splatam import transformed_params2semrendervar_sparse,set_camera_sparse,setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from sparse_channel_rasterization import GaussianRasterizer as SEMRenderer_sparse

def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments

    """
    parser = argparse.ArgumentParser(
        description="Arguments to run NARUTO."
    )
    parser.add_argument("--cfg", type=str, default="configs/default.py",
                        help="NARUTO config")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="result directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; also used as the initial pose idx for Replica")
    parser.add_argument("--enable_vis", type=int, default=None,
                        help="enable visualization. 1: True, 0: False")
    parser.add_argument("--stage", type=str, default='final',
                        help="ONLY for SplaTAM result evaluation ")
    parser.add_argument("--step", type=int, default=1100,
                        help="ONLY for SplaTAM result evaluation ")
    args = parser.parse_args()
    return args

def map_object_id_to_semlabel(object_ids,id_to_label):
    id_to_label = torch.tensor(id_to_label)
    sem_labels = id_to_label[object_ids.long()]
    sem_labels += 2
    return sem_labels

def transform_points_torch(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 transformation matrix to a set of 3D points in PyTorch.

    Args:
        points: (N, 3) tensor of 3D points (can be on CUDA)
        transform: (4, 4) transformation matrix (must be on same device)

    Returns:
        Transformed points: (N, 3) tensor
    """
    assert points.shape[-1] == 3, "Input points should have shape (N, 3)"
    assert transform.shape == (4, 4), "Transform must be of shape (4, 4)"
    device = points.device

    # Convert to homogeneous coordinates
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=device)
    points_hom = torch.cat([points, ones], dim=-1)  # (N, 4)
    transform = transform.to(device)
    # Apply transform
    transformed = (transform @ points_hom.T).T  # (N, 4)

    return transformed[:, :3]

def save_semantic_ply(params,variables, ckpt_output_dir, time_idx, n_cls=41, colormap=None, mask=None):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"transform_semantic_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    class_ids_indices = params['semantic_logits'].argmax(-1).cpu().numpy()
    topk_class = variables['seman_cls_ids'].cpu().numpy()
    class_ids = topk_class[np.arange(topk_class.shape[0]),class_ids_indices]
    if colormap is not None:
        sem_colormap = colormap
    else:
        sem_colormap = create_class_colormap(n_cls)
    sem_rgbs = apply_colormap(class_ids, sem_colormap)/255.
    # sem_opacities = self.params['logit_opacities_seman'].detach().cpu().contiguous().numpy()
    sem_opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(sem_rgbs, sem_opacities, params, ply_savepath)

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
def rot_y(theta):
    phi = (theta * (np.pi / 180.))
    rot = torch.eye(4)
    rot[:3, :3] = torch.Tensor([
        [np.cos(phi), 0, -np.sin(phi)],
        [0, 1, 0],
        [np.sin(phi), 0, np.cos(phi)]
    ])
    return rot


def rot_x(theta):
    phi = (theta * (np.pi / 180.))
    rot = torch.eye(4)
    rot[:3,:3] = torch.Tensor([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    return rot

def create_c2ws_sim(pose_loader,init_c2w):

    # init the first pose
    start_c2w = torch.from_numpy(init_c2w).float()

    start_c2w_sim = pose_loader.convert_rel2sim(start_c2w)

    pose_dict = {}
    pose_dict['F'] = start_c2w_sim
    y_thetas = [90,180,270]
    dir_name = ['L','B','R']
    for theta,d in zip(y_thetas,dir_name):
        R = rot_y(theta)
        w2c = torch.inverse(start_c2w_sim.clone())
        curr_w2c = R @ w2c
        curr_c2w = torch.inverse(curr_w2c)
        pose_dict[d] = curr_c2w

    x_thetas = [-90,90]
    dir_name = ['U', 'D']
    for theta,d in zip(x_thetas,dir_name):
        R = rot_x(theta)
        w2c = torch.inverse(start_c2w_sim.clone())
        curr_w2c = R @ w2c
        curr_c2w = torch.inverse(curr_w2c)
        pose_dict[d] = curr_c2w

    # convert the world comera to openGL
    return pose_dict


## refer to https://github.com/sunset1995/py360convert/blob/master/py360convert/c2e.py
def equirect_facetype(h, w):
    '''
    0F 1R 2B 3L 4U 5D
    '''
    tp = np.roll(np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1)

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-np.pi, np.pi, w // 4) / 4
    idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(np.int32)

def c2e(cubemap, h, w, mode='bilinear', cube_format='dice'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = py360convert.utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = py360convert.utils.cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = py360convert.utils.cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = py360convert.utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        py360convert.utils.sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec

def preprocess_color_and_depth(color,depth):
    '''
    # refer to https://github.com/spla-tam/SplaTAM/blob/main/datasets/gradslam_datasets/basedataset.py
    :param color: torch.tensor (H,W,C) # output from habitat-sim, range 0-1
    :param depth: torch.tensor (H,W)  # output from habitat-sim, range 0-depth_png_scale
    :return:
    :param color: torch.tensor (C,H,W) # output from habitat-sim, range 0-1
    :param depth: torch.tensor (1,H,W)  # output from habitat-sim, range 0-1
    '''
    depth = depth
    depth = depth.unsqueeze(-1).permute(2,0,1) # (H,W) -> (1,H,W)
    color = color.permute(2,0,1)

    return color,depth

def init_intrinsics(cfg: Dict) -> None:
    """ initialize camera parameters and camera rays

    Args:
        cfg (Dict): CoSLAM config

    Attributes:
        fx (float)                    : focal length (x)
        fy (float)                    : focal length (y)
        cx (float)                    : principal point (x)
        cy (float)                    : principal point (y)

    """
    fx, fy = cfg['cam']['fx'] // cfg['data']['downsample'], \
                       cfg['cam']['fy'] // cfg['data']['downsample']
    cx, cy = cfg['cam']['cx'] // cfg['data']['downsample'], \
                       cfg['cam']['cy'] // cfg['data']['downsample']
    intrinsics = torch.eye(3)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics

@torch.no_grad()
def render_view(curr_data, cam_params, final_params, final_variables, step, direction, plot_dir,sem_rgb_map, save_frames=False):
    '''
    curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c': curr_gt_w2c}
    '''

    np.random.seed(179868)

    plot_dict = {}

    transformed_gaussians = transform_to_frame(final_params, time_idx=0,
                                                   gaussians_grad=False,
                                                   camera_grad=False,
                                               rel_w2c= curr_data['rel_w2c'])

    # Initialize Render Variables
    rgb_rendervar = transformed_params2rendervar(final_params, transformed_gaussians)
    depth_rendervar = transformed_params2depthplussilhouette(final_params, first_frame_w2c.cuda(),
                                                                   transformed_gaussians)

    # Render Depth & Silhouette
    curr_data['depth'] = curr_data['depth'].cpu()
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_rendervar)
    rastered_depth = depth_sil[0, :, :].unsqueeze(0)

    # Render RGB and Calculate PSNR
    curr_data['im'] = curr_data['im'].cpu()
    im, radius, _  = Renderer(raster_settings=cam)(**rgb_rendervar)
    seen = radius > 0
    im = im.detach()  # (3,H,W)

    # Compute Semantic
    seman_rendervar = transformed_params2semrendervar_sparse(final_params, transformed_gaussians, seen)
    sparse_cam = set_camera_sparse(cam=curr_data['cam'], cls_ids=final_variables['seman_cls_ids'])
    # seman_rendervar['means2D'].retain_grad()
    seman_logits, _, = SEMRenderer_sparse(raster_settings=sparse_cam)(**seman_rendervar)
    class_id = seman_logits.argmax(dim=0)

    save_path = ''
    if save_frames:
        # Determine Plot Aspect Ratio
        aspect_ratio = im.shape[2] / im.shape[1]
        fig_height = 8
        fig_width = 14 / 1.55
        fig_width = fig_width * aspect_ratio
        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))

        # GT color, depth, semantic
        axs[0, 0].imshow(curr_data['im'].cpu().permute(1, 2, 0))
        axs[0, 0].set_title("Ground Truth RGB")
        plot_dict['gt_rgb'] = (255 * curr_data['im']).cpu().permute(1, 2, 0).numpy().astype(np.uint8)

        axs[0, 1].imshow(curr_data['depth'][0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
        axs[0, 1].set_title("Ground Truth Depth")

        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=0, vmax=6)
        plot_dict['gt_depth'] = cmap(norm(curr_data['depth'][0, :, :].cpu().numpy()))

        gt_seman = curr_data['seman'].cpu().numpy().astype(np.uint8)
        seman_rgb = apply_colormap(gt_seman,sem_rgb_map)
        axs[0, 2].imshow(seman_rgb)
        axs[0, 2].set_title("Ground Truth Semantic")

        plot_dict['gt_seman'] = seman_rgb

        # Save Rendered RGB, Depth and Semantic
        viz_render_im = torch.clamp(im, 0, 1)
        viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0)
        vmin = 0
        vmax = 6
        rastered_depth_viz = rastered_depth.detach()
        viz_render_depth = rastered_depth_viz[0].detach().cpu()
        axs[1, 0].imshow(viz_render_im)
        axs[1, 0].set_title("Rasterized RGB")
        plot_dict['rgb'] = (255*viz_render_im).numpy().astype(np.uint8)

        axs[1, 1].imshow(viz_render_depth, cmap='jet', vmin=0, vmax=6)
        axs[1, 1].set_title("Rasterized Depth")
        plot_dict['depth'] = cmap(norm(viz_render_depth.numpy()))

        render_seman = class_id.detach().cpu().numpy()
        render_seman_rgb = apply_colormap(render_seman.astype(np.uint8),sem_rgb_map)
        axs[1, 2].imshow(render_seman_rgb)
        axs[1, 2].set_title("Rasterized Semantic")
        plot_dict['seman'] = render_seman_rgb

        # Plot the Ground Truth and Rasterized RGB & Depth, along with Silhouette
        fig_title = f"Time Step: {step:04} View: {direction}"
        plot_name = direction
        for ax in axs.flatten():
            ax.axis('off')
        fig.suptitle(fig_title, y=0.95, fontsize=16)
        fig.tight_layout()
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        plot_dict['fig_name'] = save_path

    return plot_dict

def initialize_cam_params(num_frames):
    params = {}
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
            # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(False))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(False))
    return params



if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveSem")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')

    ##################################################
    ### initialize simulator
    ##################################################
    sim = init_simulator(main_cfg, info_printer)


    ##################################################
    ### initialize SLAM module
    ##################################################
    pose_loader = PoseLoader(main_cfg)
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    slam.load_params_by_step(step=args.step,stage=args.stage)

    ##################################################
    ### load semantic information
    ##################################################
    # # update the camera info:
    # slam_cfg = main_cfg.slam
    # config = slam.config
    # device = torch.device(config["primary_device"])

    ## modify camera
    # img_H, img_W = 1056, 1056
    # main_cfg['sim']['habitat_cfg'] = f'configs/Replica/{main_cfg.general.scene}/habitat_cube.py'
    # _, _, intrinsics, _ = slam.dataset_eval[0]
    # intrinsics[0][2] = img_H / 2
    # intrinsics[1][2] = img_W / 2
    # intrinsics = intrinsics.to(device)

    # ori_dir = f"./data/replica_v1/{main_cfg.general.scene[:-1]}_{main_cfg.general.scene[-1]}/habitat/"
    # ori_semantic_info_file = os.path.join(ori_dir, 'info_semantic.json')
    # with open(ori_semantic_info_file, 'r') as file:
    #     info_semantic = json.load(file)
    # id_to_label = np.array(info_semantic['id_to_label'])

    # sem_colormap = label_colormap(slam.n_cls)
    #
    # eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir

    # gt_mesh_ply = f"./data/replica_v1/{main_cfg.general.scene[:-1]}_{main_cfg.general.scene[-1]}/habitat/mesh_semantic.ply"
    #
    # ply = PlyData.read(gt_mesh_ply,mmap=False)
    # vertex_data = ply['vertex'].data
    # face_data = ply['face'].data
    # vertices = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    #
    # triangles = []
    # object_ids = []
    #
    # for face in face_data:
    #     verts = face[0]  # vertex_indices
    #     obj_id = face['object_id']
    #     if len(verts) == 3:
    #         triangles.append(verts)
    #         object_ids.append(obj_id)
    #     elif len(verts) == 4:
    #         i0, i1, i2, i3 = verts
    #         triangles.append([i0, i1, i2])
    #         triangles.append([i0, i2, i3])
    #         object_ids.extend([obj_id, obj_id])
    #     else:
    #         raise ValueError(f"Unsupported face with {len(verts)} vertices.")
    #
    # triangles = np.array(triangles, dtype=np.int32)
    # object_ids = np.array(object_ids, dtype=np.int32)
    # # faces = np.array([face[0] for face in face_data], dtype=np.int32)
    # # object_ids = ply['face']['object_id']
    # class_ids = id_to_label[object_ids]
    # face_colors = apply_colormap(class_ids, sem_colormap)/255.0
    #
    # # Step 3: Convert face colors to vertex colors
    # # Each vertex gets the average color of its adjacent faces
    # vertex_colors = np.zeros((len(vertices), 3))
    # count = np.zeros(len(vertices))
    #
    # for tri, color in zip(triangles, face_colors):
    #     for idx in tri:
    #         vertex_colors[idx] += color
    #         count[idx] += 1
    #
    # vertex_colors = np.divide(vertex_colors, count[:, None], where=(count[:, None] != 0))
    #
    # # Step 4: Build Open3D mesh for export
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    #
    # mask = vertices[:, 2] <= 1.0
    #
    # vertices = np.asarray(mesh.vertices)
    # triangles = np.asarray(mesh.triangles)
    #
    # # Map old vertex indices to new ones
    # old_to_new = -np.ones(len(vertices), dtype=int)
    # old_to_new[mask] = np.arange(np.sum(mask))
    #
    # # Filter triangles: only keep ones where all 3 vertices are in mask
    # keep_tris = []
    # for tri in triangles:
    #     if all(mask[tri]):
    #         keep_tris.append([old_to_new[i] for i in tri])
    #
    # # Create new mesh
    # new_mesh = o3d.geometry.TriangleMesh()
    # new_mesh.vertices = o3d.utility.Vector3dVector(vertices[mask])
    # new_mesh.triangles = o3d.utility.Vector3iVector(np.array(keep_tris))
    #
    # # Also filter vertex colors
    # if mesh.has_vertex_colors():
    #     colors = np.asarray(mesh.vertex_colors)[mask]
    #     new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #
    # eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir
    # os.makedirs(eval_dir, exist_ok=True)
    # ply_name = f"semantic_GT.ply"
    # ply_savepath = os.path.join(eval_dir, ply_name)
    # o3d.io.write_triangle_mesh(ply_savepath, new_mesh,write_ascii=False)
    # print(f"Recolored mesh saved to {ply_savepath}")

    ## RGB-mesh

    # gt_mesh_ply = f"./data/replica_v1/{main_cfg.general.scene[:-1]}_{main_cfg.general.scene[-1]}/mesh.ply"
    #
    # mesh = trimesh.load_mesh(gt_mesh_ply, process=False)
    # if not mesh.is_watertight or mesh.faces.shape[1] != 3:
    #     mesh = mesh.triangulate()
    #
    # # Convert to numpy arrays
    # vertices = mesh.vertices
    # faces = mesh.faces
    # if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
    #     colors = mesh.visual.vertex_colors[:, :3] / 255.0  # RGB only
    # else:
    #     colors = np.ones_like(vertices) * 0.5  # default gray
    #
    # # Step 1: Filter vertices by Z <= 1.0
    # mask = vertices[:, 2] <= 1.0
    # old_to_new = -np.ones(len(vertices), dtype=int)
    # old_to_new[mask] = np.arange(np.sum(mask))
    #
    # # Step 2: Filter triangles — keep only if all 3 vertices remain
    # filtered_faces = []
    # for f in faces:
    #     if np.all(mask[f]):
    #         filtered_faces.append([old_to_new[i] for i in f])
    #
    # # Step 3: Construct new mesh
    # new_mesh = o3d.geometry.TriangleMesh()
    # new_mesh.vertices = o3d.utility.Vector3dVector(vertices[mask])
    # new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[mask])
    # new_mesh.triangles = o3d.utility.Vector3iVector(np.array(filtered_faces))
    # new_mesh.compute_vertex_normals()
    #
    # # Optional: recompute normals
    # new_mesh.compute_vertex_normals()
    #
    # # Save to file
    # ply_name = f"filtered_rgb_mesh.ply"
    # ply_savepath = os.path.join(eval_dir, ply_name)
    # o3d.io.write_triangle_mesh(ply_savepath, new_mesh, write_ascii=False)
    # print("✅ Filtered mesh saved as 'filtered_rgb_mesh.ply'")



    #
    #
    ####### The reconstruction semantic & RGB mesh

    # sem_colormap = label_colormap(slam.n_cls)
    sem_colormap = create_class_colormap(slam.n_cls)

    eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir
    rel_c2w0 = pose_loader.load_init_pose()
    rel_c2w0[:3, 1] *= -1
    rel_c2w0[:3, 2] *= -1  # RDF
    T_slam2sim = rel_c2w0.clone()  # RDF

    T_sim2slam = torch.linalg.inv(rel_c2w0.clone())

    eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir

    points = slam.params['means3D']
    transform_points = transform_points_torch(points,T_slam2sim)
    # slam.params['means3D'] = transform_points

    ## filter out
    time_idx = 2000
    # z_thres = -1
    z_thres = 2.0 # MP3D
    mask = transform_points[:, 2] <= z_thres

    # time_idx = 0
    # y_thres = 1.5
    # # y_thres = 2.0 # MP3D
    # mask = transform_points[:, 1] <= y_thres


    if main_cfg.slam['method'] == 'semsplatam':
        filter_params = ['semantic_logits', 'rgb_colors',  'logit_opacities', 'log_scales','unnorm_rotations']
        filter_variables = ['seman_cls_ids']
        save_params = {}
        save_variables = {}
        save_params['means3D'] = transform_points[mask]
        for key in filter_params:
            save_params[key] = slam.params[key][mask]
        for key in filter_variables:
            save_variables[key] = slam.variables[key][mask]
        save_semantic_ply(save_params, save_variables, eval_dir, time_idx, slam.n_cls, colormap=sem_colormap, mask=None)
    elif main_cfg.slam['method'] == 'sgsslam':
        filter_params = ['semantic_colors', 'rgb_colors', 'logit_opacities', 'log_scales', 'unnorm_rotations']
        save_params = {}
        save_params['means3D'] = transform_points[mask]
        for key in filter_params:
            save_params[key] = slam.params[key][mask]

        os.makedirs(eval_dir, exist_ok=True)
        ply_name = f"transform_semantic_GS_{time_idx:04}.ply"
        ply_savepath = os.path.join(eval_dir, ply_name)
        sem_opacities = save_params['logit_opacities'].detach().cpu().contiguous().numpy()
        sem_rgbs = save_params['semantic_colors'].detach().cpu().contiguous().numpy()
        write_ply_file(sem_rgbs, sem_opacities, save_params, ply_savepath)

    os.makedirs(eval_dir, exist_ok=True)
    ply_name = f"transform_rgb_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(eval_dir, ply_name)
    opacities = save_params['logit_opacities'].detach().cpu().contiguous().numpy()
    rgbs = save_params['rgb_colors'].detach().cpu().contiguous().numpy()
    write_ply_file(rgbs, opacities, save_params, ply_savepath)

    ########## The reconstruction entropy mesh
    # rel_c2w0 = pose_loader.load_init_pose()
    # rel_c2w0[:3, 1] *= -1
    # rel_c2w0[:3, 2] *= -1  # RDF
    # T_slam2sim = rel_c2w0.clone()  # RDF
    #
    # T_sim2slam = torch.linalg.inv(rel_c2w0.clone())
    #
    # eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir
    #
    # points = slam.params['means3D']
    # transform_points = transform_points_torch(points, T_slam2sim)
    # # slam.params['means3D'] = transform_points
    #
    # ## filter out
    # z_thres = 1.5
    # mask = transform_points[:, 2] <= z_thres
    #
    # filter_params = ['semantic_logits', 'logit_opacities', 'log_scales', 'unnorm_rotations']
    # filter_variables = ['seman_cls_ids']
    # save_params = {}
    # save_variables = {}
    # save_params['means3D'] = transform_points[mask]
    # for key in filter_params:
    #     save_params[key] = slam.params[key][mask]
    # for key in filter_variables:
    #     save_variables[key] = slam.variables[key][mask]
    # save_entropy = calc_shannon_entropy(save_params['semantic_logits'],dim=-1)
    # array = save_entropy.detach().cpu().numpy()  # shape: (N,)
    # array = (255 * (array - array.min()) / (array.max() - array.min())).astype(np.uint8)
    # gray_img = array.reshape(-1, 1)
    # heatmap = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    # heatmap_rgbs = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap_rgbs = torch.from_numpy(heatmap_rgbs).float() / 255.0
    # heatmap_rgbs = heatmap_rgbs.squeeze(1)
    #
    # time_idx = 2000
    # os.makedirs(eval_dir, exist_ok=True)
    # ply_name = f"transform_entropy_GS_{time_idx:04}.ply"
    # ply_savepath = os.path.join(eval_dir, ply_name)
    # opacities = save_params['logit_opacities'].detach().cpu().contiguous().numpy()
    # entropy_rgbs = heatmap_rgbs.detach().cpu().contiguous().numpy()
    # write_ply_file(entropy_rgbs, opacities, save_params, ply_savepath)



        #     save_semantic_ply(save_params, save_variables, eval_dir, 2000, slam.n_cls, colormap=sem_colormap, mask=None)
        # elif main_cfg.slam['method'] == 'sgsslam':
        #     filter_params = ['semantic_colors', 'rgb_colors', 'logit_opacities', 'log_scales', 'unnorm_rotations']
        #     save_params = {}
        #     save_params['means3D'] = transform_points[mask]
        #     for key in filter_params:
        #         save_params[key] = slam.params[key][mask]
        #
        #     time_idx = 2000
        #     os.makedirs(eval_dir, exist_ok=True)
        #     ply_name = f"transform_semantic_GS_{time_idx:04}.ply"
        #     ply_savepath = os.path.join(eval_dir, ply_name)
        #     sem_opacities = save_params['logit_opacities'].detach().cpu().contiguous().numpy()
        #     sem_rgbs = save_params['semantic_colors'].detach().cpu().contiguous().numpy()
        #     write_ply_file(sem_rgbs, sem_opacities, save_params, ply_savepath)

    ########## The reconstruction rgb mesh
    # rel_c2w0 = pose_loader.load_init_pose()
    # rel_c2w0[:3, 1] *= -1
    # rel_c2w0[:3, 2] *= -1  # RDF
    # T_slam2sim = rel_c2w0.clone()  # RDF
    #
    # T_sim2slam = torch.linalg.inv(rel_c2w0.clone())
    #
    # eval_dir = slam.eval_dir + "_" + args.stage if args.stage else slam.eval_dir
    #
    # points = slam.params['means3D']
    # transform_points = transform_points_torch(points, T_slam2sim)
    # # slam.params['means3D'] = transform_points
    #
    # ## filter out
    # z_thres = 1.5
    # mask = transform_points[:, 2] <= z_thres
    #
    # filter_params = ['rgb_colors', 'logit_opacities', 'log_scales', 'unnorm_rotations']
    # save_params = {}
    # save_variables = {}
    # save_params['means3D'] = transform_points[mask]
    # for key in filter_params:
    #     save_params[key] = slam.params[key][mask]
    #
    # time_idx = 2000
    # os.makedirs(eval_dir, exist_ok=True)
    # ply_name = f"transform_rgb_GS_{time_idx:04}.ply"
    # ply_savepath = os.path.join(eval_dir, ply_name)
    # opacities = save_params['logit_opacities'].detach().cpu().contiguous().numpy()
    # rgbs = save_params['rgb_colors'].detach().cpu().contiguous().numpy()
    # write_ply_file(rgbs, opacities, save_params, ply_savepath)

