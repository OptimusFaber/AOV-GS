import cv2
import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
import torch
import numpy as np
from PIL import Image
import requests
import json
from torch.optim import AdamW
import torchvision.transforms as transforms
from PIL import Image

from src.naruto.cfg_loader import argument_parsing, load_cfg
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step




import json
import os

def map_object_id_to_semlabel(object_ids, id2label):
    '''

    :param object_ids: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    :return: semantic labels: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    '''
    id2label = torch.tensor(id2label)
    sem_labels = id2label[object_ids.long()]
    return sem_labels


import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import random


def generate_random_colormap(num_classes):
    """
    Generate a random colormap for a given number of classes.

    Args:
        num_classes (int): Number of unique classes in the mask.

    Returns:
        dict: A dictionary mapping class indices to random RGB colors.
    """
    random.seed(42)  # Set seed for reproducibility
    colormap = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}
    return colormap


def semantic_mask_to_rgb(mask: torch.Tensor, save_path: str, num_classes=102):
    """
    Convert a semantic mask tensor (H, W) to an RGB image and save it.

    Args:
        mask (torch.Tensor): A tensor of shape (H, W) with semantic class indices.
        save_path (str): Path to save the RGB image.
    """
    # Get the number of unique classes
    unique_classes = torch.unique(mask).tolist()
    # num_classes = max(unique_classes) + 1  # Assuming classes start from 0

    # Generate a random colormap
    colormap = generate_random_colormap(num_classes)

    # Convert mask to numpy
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Create an RGB image
    H, W = mask_np.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Map each class to its corresponding random color
    for class_id in unique_classes:
        rgb_image[mask_np == class_id] = colormap[class_id]

    # Convert to PIL image and save
    img = Image.fromarray(rgb_image)
    img.save(save_path)


if __name__ == "__main__":

    info_printer = InfoPrinter("Generate Finetune data")
    timer = Timer()
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)

    #scenes = ["office0", "office1", "office2","room0", "room1","room2", "office3", "office4"]
    scenes = [ "room2"]

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    for selected_scene in scenes:
        info_printer("Modifying configuration...", 0, "Initialization")
        main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
        info_printer.update_total_step(main_cfg.general.num_iter)
        main_cfg.general.scene = selected_scene
        main_cfg.dirs.cfg_dir = f'configs/{main_cfg.general.dataset}/{selected_scene}/'
        main_cfg.sim.habitat_cfg = f'configs/{main_cfg.general.dataset}/{selected_scene}/habitat.py'

        ### for NVS, here need to traj file
        #main_cfg.planner.SLAMData_dir = os.path.join(main_cfg.dirs.data_dir,main_cfg.general.dataset, main_cfg.general.scene)
        main_cfg.planner.SLAMData_dir = os.path.join(main_cfg.dirs.data_dir, 'replica_sim_nvs',
                                                     main_cfg.general.scene)

        info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

        ##########  load in id2label ##############
        ori_dir = f"./data/replica_v1/{main_cfg.general.scene[:-1]}_{main_cfg.general.scene[-1]}/habitat/"
        ori_semantic_info_file = os.path.join(ori_dir, 'info_semantic.json')
        with open(ori_semantic_info_file, 'r') as file:
            scene_id2label = json.load(file)['id_to_label']
        main_cfg.general.semantic_dir = ori_dir

        img_save_dir = f'./data/replica_sim_nvs/{selected_scene}/results_habitat/'
        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(f'{img_save_dir}/rgb', exist_ok=True)
        os.makedirs(f'{img_save_dir}/semantic', exist_ok=True)

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
        ### initialize planning module
        ##################################################
        planner = init_planner(main_cfg, info_printer)
        planner.update_sim(sim)

        ##################################################
        ### convert pose
        ##################################################
        ## load initial pose and convert from RUB to RDF (splatam)) ##
        c2w_slam = planner.load_init_pose()  # RUB
        c2w_slam[:3, 1] *= -1
        c2w_slam[:3, 2] *= -1  # RDF
        c2w_slam_init = c2w_slam.clone()  # RDF

        ## initialize first pose ##
        T_sim2slam = torch.inverse(
            c2w_slam_init)  # RDF # transformation that takes sim-world points to slam-world-origin (i.e. first camera)
        planner.init_data(T_sim2slam) # 'PreTrajPlanner' object has no attribute 'init_data'
        planner.timer = timer

        step_size = 1
        total_poses = len(planner.pose_loader.predefined_traj)
        if total_poses > main_cfg.general.num_iter:
            step_size = total_poses//main_cfg.general.num_iter

        for i in range(total_poses):
            update_module_step(i, [sim, planner])

            c2w_slam = planner.update_pose(c2w_slam, i).to(c2w_slam.device)  # RUB
            c2w_sim = c2w_slam.cpu().numpy().copy()  # RUB
            ## convert back to RDF (splatam) ##
            c2w_slam[:3, 1] *= -1
            c2w_slam[:3, 2] *= -1

            if i % step_size == 0:
                ##################################################
                ### Simulation
                ##################################################
                timer.start("Simulation", "General")
                # c2w_sim = np.eye(4)
                # T_rgb2sem = np.eye(4)
                # T_rgb2sem[1,2] = -1
                # T_rgb2sem[1,1] = 0
                # T_rgb2sem[2,1] = 1
                # T_rgb2sem[2,2] = 0
                c2w_sim_rgb = c2w_sim.copy()
                # c2w_sim_seman = np.linalg.inv(T_rgb2sem) @ c2w_sim.copy()
                c2w_sim_seman = c2w_sim.copy()
                sim_out_rgb = sim.simulate(c2w_sim_rgb, return_semantic=True)
                sim_out_seman = sim.simulate(c2w_sim_seman, return_semantic=True)
                sim_out = {
                    'color': sim_out_rgb['color'],
                    'seman': sim_out_seman['seman'],
                }

                #sim_out = sim.simulate(c2w_sim, return_semantic=True, return_erp=True)
                timer.end("Simulation")
                color = sim_out['color']
                to_pil = transforms.ToPILImage()
                image = to_pil(color.permute(2, 0, 1))
                image.save(f"{img_save_dir}/rgb/color_{i:04d}.jpg")

                seman = sim_out['seman'].long()
                seman = map_object_id_to_semlabel(object_ids=sim_out['seman'],id2label=scene_id2label)
                np.save(f"{img_save_dir}/semantic/semantic_map_{i:04d}.npy", seman.numpy())
                seman[seman < 0] = 0
                semantic_mask_to_rgb(seman, f"{img_save_dir}/semantic/semantic_rgb_{i:04d}.png")