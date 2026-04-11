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


import argparse
from glob import glob
import numpy as np
import open3d as o3d
import os, sys
import time


sys.path.append(os.getcwd())
from src.visualization.o3d_utils import (
    create_camera_frustum, 
    save_camera_parameters, 
    # load_camera_parameters_from_json,
    create_dashed_line
    )

import open3d.visualization.webrtc_server as webrtc_server


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to visualize trajectory."
        )
    parser.add_argument("--mesh_file", type=str, default="", 
                        help="mesh file")
    parser.add_argument("--traj_file", type=str, default="",
                        help="trajectory pose dir")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="output directory to save rendered image")
    parser.add_argument("--with_interact", type=int, default=1,
                        help="with interaction for visualization")
    args = parser.parse_args()
    return args


def convert_rel2world(start_c2w_rel, rel_c2w_slam):
    c2w_slam_w = start_c2w_rel @ rel_c2w_slam
    return c2w_slam_w

def load_cam_traj(scene_path):
    params = dict(np.load(scene_path, allow_pickle=True))
    cam_traj = {}
    cam_traj['w2cs'] = params.pop('gt_w2c_all_frames')
    cam_traj['intrinsic'] = params.pop('intrinsics')
    cam_traj['height'] = params.pop('org_height')
    cam_traj['width'] = params.pop('org_width')
    return cam_traj # 80,4,4

def load_Replica_pose(line: str):
    """ load Replica pose from trajectory file

    Args:
        line (str): pose data as txt line. Format: camera-to-world, RUB

    Returns:
        c2w (np.ndarry, [4,4]): pose. Format: camera-to-world, RDF
    """
    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    return c2w

def load_camera_parameters_from_data(cam_traj: dict) -> o3d.camera.PinholeCameraParameters:
    """ load camera parameters from json

    Args:
        json_file (str): camera parameter json file

    Returns:
        cam_param (o3d.camera.PinholeCameraParameters): camera parameters
    """
    # Load intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam_traj["width"],
        cam_traj["height"],
        cam_traj["intrinsic"][0, 0],  # fx
        cam_traj["intrinsic"][1, 1],  # fy
        cam_traj["intrinsic"][0, 2],  # cx
        cam_traj["intrinsic"][1, 2]  # cy
    )

    # Load extrinsic parameters
    w2c_params = cam_traj["w2cs"][0]
    w2c = np.array(w2c_params).reshape((4, 4))
    extrinsic_matrix = np.linalg.inv(w2c)
    extrinsic_matrix = extrinsic_matrix.transpose()

    # Create PinholeCameraParameters
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = extrinsic_matrix
    cam_param.intrinsic = intrinsic

    return cam_param

if __name__ == '__main__':

    ### arguments ###
    args = argument_parsing()

    # set params ##
    HOME = "/mnt/Data2/"
    PROJ_DIR = f"{HOME}/liyan/ActiveGAMER"
    DATASET = "Replica"
    RESULT_DIR =f'{PROJ_DIR}/results'
    GT_DATA_DIR = f"{HOME}/slam_datasets/{DATASET}"

    scene = "office4"
    seed = 0
    method = "ActiveLang"
    slam = "splatam"
    args.mesh_file =f'{GT_DATA_DIR}/{scene}_mesh.ply'
    args.traj_file =f'{RESULT_DIR}/{DATASET}/{scene}/{method}/run_0/{slam}/final/params.npz'
    args.out_dir =f'{RESULT_DIR}/{DATASET}/{scene}/{method}/run_0/visualization/planning_path/'

    traj_txt=f'{GT_DATA_DIR}/{scene}/traj.txt'

    with open(traj_txt, 'r') as f:
        lines = f.readlines()
        poses = [load_Replica_pose(line) for line in lines]

    transform = poses[0]

    mesh_file = args.mesh_file
    window_hw = (1024, 1024)

    mesh = o3d.io.read_point_cloud(mesh_file)
    # modify mesh
    vertices = np.asarray(mesh.points)
    thres = 1.0
    mask = vertices[:, 2] <= thres
    filtered_vertices = vertices[mask]
    filtered_colors = np.asarray(mesh.colors)[mask]
    filtered_mesh = o3d.geometry.PointCloud()
    filtered_mesh.points = points=o3d.utility.Vector3dVector(filtered_vertices)
    filtered_mesh.colors=o3d.utility.Vector3dVector(filtered_colors)

    # mesh.transform(transform)
    # camera_trajectory = np.load(traj_file)
    camera_trajectory = load_cam_traj(args.traj_file)

    ip = '127.0.0.1'
    port = '5001'
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['OPEN3D_CPU_RENDERING'] = 'true'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = 'true'
    os.environ['WEBRTC_IP'] = ip
    os.environ['WEBRTC_PORT'] = port
    webrtc_server.enable_webrtc()

    ### initialize window ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=window_hw[1], height=window_hw[0])

    # webrtc_server.register_object("mesh", filtered_mesh)
    o3d.visualization.draw(filtered_mesh)

    ### Add mesh ###
    # vis.draw_geometries([mesh])
    # vis.add_geometry(mesh)
    # vis.add_geometry(filtered_mesh)

    ### set a view direction ###
    if args.traj_file is not None:
        vis_cam_param = load_camera_parameters_from_data(camera_trajectory)
    # view_control = vis.get_view_control()

    ### Add trajecotry ###
    skip_step = 5
    w2c_subset = camera_trajectory['w2cs'][::skip_step]
    # cam_traj_subset = camera_trajectory[::skip_step]
    # cam_traj_subset = camera_trajectory[:10]
    c2ws = [convert_rel2world(transform,np.linalg.inv(w2c)) for w2c in w2c_subset]
    w2c_subset = [np.linalg.inv(c2w) for c2w in c2ws]

    for step, w2c in enumerate(w2c_subset):
        ##################################################
        ### Add Camera ###
        ##################################################
        pose = np.linalg.inv(w2c)
        intrinsic = np.array([[300, 0, 300],
                            [0, 300, 300],
                            [0, 0, 1]])
        ### Create camera frustum ###
        if step == 0:
            color = [1, 0, 0]
        elif step == len(w2c_subset) - 1:
            color = [0, 0, 1]
        else:
            color = [0, 1, 0]
        camera_frustum = create_camera_frustum(color=color, extrinsic=pose, intrinsic=intrinsic, scale=1)
        # webrtc_server.register_object(f"camera_{step}", camera_frustum)
        o3d.visualization.draw(camera_frustum)

        # vis.add_geometry(camera_frustum)

        ##################################################
        ### Add line
        ##################################################
        if step > 0:
            points = [np.linalg.inv(w2c)[:3, 3] for w2c in w2c_subset[step-1:step+1]]
            line_set = create_dashed_line(points, color=[0, 0, 0])
            # vis.add_geometry(line_set)
            # webrtc_server.register_object(f"line_{step}", line_set)
            o3d.visualization.draw(line_set)

        ##################################################
        ### set camera view
        ##################################################
        # if args.traj_file is not None:
        #     view_control.convert_from_pinhole_camera_parameters(vis_cam_param, allow_arbitrary=True)

        ##################################################
        ### update visualizer
        ##################################################
        # vis.poll_events()
        # vis.update_renderer()

        ##################################################
        ### save visualization
        ##################################################
        # if args.out_dir is not None:
        #     os.makedirs(args.out_dir, exist_ok=True)
        #     render_filepath = os.path.join(args.out_dir, f"{step*skip_step:04}.png")
        #     vis.capture_screen_image(render_filepath)

        time.sleep(0.01)

    ### RUN ###
    # if args.with_interact:
    #     save_camera_parameters(vis)
    #     vis.run()
    #     vis.destroy_window()
    # else:
    #     vis.destroy_window()

    print("WebRTC Visualization running at: http://localhost:5001")
    while True:
        time.sleep(1)  # Keep the server running
