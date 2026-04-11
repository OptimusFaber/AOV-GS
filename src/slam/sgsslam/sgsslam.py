"""
We have reused part of SplaTAM's code in this file.
For SplaTAM License, refer to https://github.com/spla-tam/SplaTAM/blob/main/LICENSE.
"""
import os.path
import sys
import mmengine
import torch
from tensorboardX import SummaryWriter
from typing import Dict, List, Optional, Tuple
from PIL import Image
import json
import time
from imgviz import label_colormap

from src.slam.splatam.splatam import SplatamOurs
from src.slam.splatam.eval_helper import eval, report_progress
from src.utils.general_utils import InfoPrinter
from src.slam.splatam.exploration_map import ExplorationMap

from third_parties.splatam.utils.slam_external import calc_psnr

### original Splatam modules ###
sys.path.append("third_parties/splatam")
from scripts.splatam import get_dataset, initialize_camera_pose
from utils.slam_helpers import (
    matrix_to_quaternion, transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette
)
from datasets.gradslam_datasets import (load_dataset_config,)
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians
from utils.common_utils import save_params, save_params_ckpt
from utils.recon_helpers import setup_camera

### original Semantic network modules ###
from transformers import AutoProcessor, AutoModelForUniversalSegmentation

# modified version
from src.slam.semsplatam.modified_ver.semantic.oneformer import oneformer_segmentation
from src.data.finetune_oneformer_ReplicaV2 import modify_metadata
from src.slam.semsplatam.modified_ver.scripts.splatam import get_dataset
from src.slam.semsplatam.modified_ver.splatam.export_helper import save_rgb_ply
from src.slam.sgsslam.modified_ver.scripts.eval_helper import *
from src.slam.sgsslam.modified_ver.scripts.export_helper import *
from src.slam.sgsslam.modified_ver.scripts.sgsslam import *

PRINT_INFO = True

class SGSSLAMOurs(SplatamOurs):
    def __init__(self,
                 main_cfg: mmengine.Config, info_printer: InfoPrinter, logger: SummaryWriter
                 ) -> None:
        SplatamOurs.__init__(self, main_cfg, info_printer, logger)
        ### load eval dataset with semantic
        self.load_eval_dataset_with_semantic()
        ### loading in segmantation network and Langeuage encoder ###
        # Check if semantic_device is available, fallback to primary device if not
        semantic_device_str = self.slam_cfg['semantic_device']
        try:
            semantic_device = torch.device(semantic_device_str)
            # Try to create a tensor on the device to verify it's available
            test_tensor = torch.zeros(1).to(semantic_device)
            self.semantic_device = semantic_device_str
        except (RuntimeError, AssertionError) as e:
            print(f"Warning: Semantic device {semantic_device_str} is not available. Falling back to {self.device}.")
            self.semantic_device = str(self.device)
        self.oneformer_processor = AutoProcessor.from_pretrained(self.slam_cfg['ade20k_checkpoint'])
        self.oneformer_model = AutoModelForUniversalSegmentation.from_pretrained(
            self.slam_cfg['oneformer_checkpoint'], is_training=False).to(self.semantic_device)
        self.load_semantics = self.config["data"]["load_semantics"]
        self.n_cls = self.config["data"]["num_semantic_classes"]
        self.oneformer_processor.image_processor.num_text = self.oneformer_model.config.num_queries - self.oneformer_model.config.text_encoder_n_ctx
        class_info_file = f"./configs/Replica/{main_cfg.general.scene}/class_info_file.json"
        info_semantic_file = os.path.join(self.slam_cfg['semantic_dir'], 'info_semantic.json')

        if os.path.exists(class_info_file):
            self.class_info_file = class_info_file
        else:
            self.init_replica_config_cls2label(info_semantic_file, class_info_file)
            self.class_info_file = class_info_file

        with open(info_semantic_file, 'r') as file:
            self.id2label = json.load(file)['id_to_label']

        modify_metadata(class_info_file=self.class_info_file, processor=self.oneformer_processor)

        # self.colormap = create_class_colormap(self.n_cls)
        self.colormap = label_colormap(self.n_cls)

    def load_eval_dataset_with_semantic(self):
        dataset_config = self.config["data"]

        ### gradslam data_fg ###
        if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        else:
            gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

        if "ignore_bad" not in dataset_config:
            dataset_config["ignore_bad"] = False

        if "use_train_split" not in dataset_config:
            dataset_config["use_train_split"] = True

        if "densification_image_height" not in dataset_config:
            dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
            dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
            self.seperate_densification_res = False
        else:
            if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                    dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_densification_res = True
            else:
                self.seperate_densification_res = False

        if "tracking_image_height" not in dataset_config:
            dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
            dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
            self.seperate_tracking_res = False
        else:
            if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                    dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_tracking_res = True
            else:
                self.seperate_tracking_res = False

        self.dataset_eval = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=self.slam_cfg.dataset_eval_basedir,
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            load_semantics = dataset_config['load_semantics'],
        )
    def init_replica_config_cls2label(self,info_semantic_file, save_json_file):
        with open(info_semantic_file, 'r') as file:
            info_semantic = json.load(file)
        self.num_cls = len(info_semantic["classes"])+1  # for replica, 0 is unknown
        id2label = {}
        for cls in info_semantic["classes"]:
            id2label[str(cls['id'])] =  {"isthing": 1, "name": cls["name"]}

        os.makedirs(os.path.dirname(save_json_file), exist_ok=True)
        with open(save_json_file, "w") as json_file:
            json.dump(id2label, json_file, indent=4)  # indent=4 makes it more readable
        print(f"Semantic Info Dictionary saved to {save_json_file}")

    @torch.no_grad()
    def semantic_annotation(self, input_img: torch.Tensor):
        input_img = (input_img * 255).byte()
        input_img_numpy = input_img.cpu().numpy()
        input_img = Image.fromarray(input_img_numpy)
        class_ids_from_oneformer, _ = oneformer_segmentation(input_img,
                                                              self.oneformer_processor,
                                                              self.oneformer_model,
                                                                self.semantic_device)
        class_ids_from_oneformer = class_ids_from_oneformer[0].cpu()
        class_colors_from_oneformer = apply_colormap(class_ids_from_oneformer.numpy(),self.colormap) / 255.
        return class_ids_from_oneformer, torch.from_numpy(class_colors_from_oneformer)

    # TODO: add render semantic
    @torch.no_grad()
    def render(self, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' render rgb, mask, and depth based on a given pose
        Args:
            c2w: [4,4]. camera-to-world pose, in SplaTAM system
        
        Returns:
            im: (H,W,3) # render image
            depth: (H,W) # render depth
            mask: (H,W) valid rendering mask
        '''
        cam = self.cam
        first_frame_w2c = self.first_frame_w2c
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)
        sil_thres = self.config['mapping']['sil_thres']
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c, 
                                                                     transformed_gaussians)
    
        im, _, _, = Renderer(raster_settings=cam)(**rendervar)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (depth_sil[0:1] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        return im, rastered_depth, valid_depth_mask

    def load_checkpoint(self):
        """ load checkpoint

        Attributes:
            variables: Splatam variables
            gt_w2c_all_frames: GT world to camera pose
            keyframe_list: keyframe list
            params: Splatam parameters
        """
        ### load self variables ###
        config = self.config
        variables = self.variables
        dataset = self.dataset_sample
        device = self.device
        params_opt_exclude = self.params_opt_exclude
        gt_w2c_all_frames = self.gt_w2c_all_frames
        keyframe_list = self.keyframe_list

        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        for k in params:
            if k not in params_opt_exclude:
                params[k] = torch.tensor(params[k]).to(device).float().requires_grad_(True)
            else:
                params[k] = torch.tensor(params[k]).to(device).float()
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(
            os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)

        self.variables = variables
        self.gt_w2c_all_frames = gt_w2c_all_frames
        self.keyframe_list = keyframe_list
        self.params = params

    def init_camera_parameters(self):
        """

        Attributes:
            params: Splatam parameters
            variables: Splatam variables
            intrinsics: camera intrinsics
            first_frame_w2c: first world to camera pose
            cam
            densify_intrinsics
            densify_cam
            tracking_cam

        """
        color, depth, intrinsics, pose = self.dataset_sample[0]
        semantic_id, semantic_color = self.semantic_annotation(color)
        semantic_id = semantic_id.unsqueeze(-1).to(self.device)
        semantic_color = semantic_color.to(self.device)

        if self.seperate_densification_res:
            # Initialize Parameters, Canonical & Densification Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam, params_opt_exclude, \
                densify_intrinsics, densify_cam = initialize_first_timestep(self.dataset_sample,
                                                                            semantic_id,semantic_color,
                                                                            self.num_frames,
                                                                            self.config['scene_radius_depth_ratio'],
                                                                            self.config['mean_sq_dist_method'],
                                                                            device=self.device,
                                                                            densify_dataset=self.densify_dataset_sample,
                                                                            load_semantics=self.load_semantics)
            # return params, variables, intrinsics, first_frame_w2c, cam, \
            #     densify_intrinsics, densify_cam
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        else:
            # Initialize Parameters & Canoncial Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam, \
                params_opt_exclude = initialize_first_timestep(self.dataset_sample,
                                                               semantic_id,semantic_color,
                                                               self.num_frames,
                                                               self.config['scene_radius_depth_ratio'],
                                                               self.config['mean_sq_dist_method'],
                                                               device=self.device,
                                                               load_semantics=self.load_semantics)
            # return params, variables, intrinsics, first_frame_w2c, cam
            self.densify_intrinsics = intrinsics
            self.densify_cam = cam

        if self.seperate_tracking_res:
            self.tracking_cam = setup_camera(self.tracking_color.shape[2], self.tracking_color.shape[1],
                                             self.tracking_intrinsics.cpu().numpy(),
                                             first_frame_w2c.detach().cpu().numpy())

        self.params = params
        self.variables = variables
        self.params_opt_exclude = params_opt_exclude
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam

    def online_recon_step(self,
                          time_idx        : int,
                          color           : torch.Tensor,
                          depth           : torch.Tensor,
                          c2w             : torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          keyframes_extra_subdir: Optional[str] = None,
                          ) -> List:
        ''' Run one step of the co-slam process.

        Args:
            time_idx        : Current frame step
            color           : color,        [H,W,3]
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
            only_use_global_keyframe: post-refinement stage
        
        Returns:
        '''
        if time_idx==0:
            self.init_camera_parameters()
        seg_img = color.clone().to(self.semantic_device)
        self.semantic_annotation(seg_img)
        seman_id, seman_color = self.semantic_annotation(seg_img)

        self.update_gs_map(time_idx, color, depth, seman_id, seman_color, c2w, force_map_update, dont_add_kf, only_use_global_keyframe)
        if self.slam_cfg.enable_active_planning:
            self.update_explr_map(time_idx, depth, c2w, force_map_update)

    def update_gs_map(self, 
                      time_idx: int,
                      color   : torch.Tensor,
                      depth   : torch.Tensor,
                      seman_id: torch.Tensor,
                      seman_color: torch.Tensor,
                      c2w     : torch.Tensor,
                      force_map_update: bool = False,
                      dont_add_kf: bool = False,
                      only_use_global_keyframe: bool = False,
                          ) -> List:
        ''' Run one step of the sgs-slam process. Update GS

        Args:
            time_idx: Current frame step
            color   : color,        [H,W,3]
            depth   : depth map,    [H,W]
            seman_id: semantic class ids,  [H,W]
            seman_color: semantic class rgb colors, [H,W,3]
            c2w     : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
        '''
        ### get self variables ###
        params = self.params
        variables = self.variables
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        cam = self.cam
        seperate_densification_res = self.seperate_densification_res
        if seperate_densification_res:
            densify_intrinsics = self.densify_intrinsics
            densify_cam = self.densify_cam
        config = self.config
        gt_w2c_all_frames = self.gt_w2c_all_frames
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
            wandb_mapping_step = self.wandb_mapping_step
            wandb_time_step = self.wandb_time_step
        eval_dir = self.eval_dir
        seperate_tracking_res = self.seperate_tracking_res
        if seperate_tracking_res:
            tracking_cam = self.tracking_cam
            tracking_intrinsics = self.tracking_intrinsics
        keyframe_list = self.keyframe_list
        num_frames = self.num_frames
        keyframe_time_indices = self.keyframe_time_indices


        ### Process poses ###
        gt_w2c = torch.linalg.inv(c2w)


        # Process RGB-D Data
        color = color.permute(2, 0, 1)
        color = color.to(self.device)
        depth = depth.unsqueeze(0)
        depth = depth.to(self.device)
        seman_id = seman_id.unsqueeze(0).to(self.device)
        seman_color = seman_color.permute(2, 0, 1)
        seman_color = seman_color.to(self.device)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx


        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth,
                     'semantic_id': seman_id, 'semantic_color': seman_color.to(color.dtype),
                     'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # # Initialize Data for Tracking
        if seperate_tracking_res:
            ### Load tracking data ###
            tracking_h, tracking_w = self.config['data']["tracking_image_height"], self.config['data']["tracking_image_width"]
            tracking_color = F.interpolate(color.unsqueeze(0), (tracking_h, tracking_w), mode='bilinear')[0]
            tracking_depth = F.interpolate(depth.unsqueeze(0), (tracking_h, tracking_w), mode='nearest')[0]
            tracking_seman_id = F.interpolate(seman_id.unsqueeze(0).float(), (tracking_h, tracking_w), mode='nearest')[0].long()
            tracking_seman_color = F.interpolate(seman_color.unsqueeze(0), (tracking_h, tracking_w), mode='bilinear')[0]
            
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                                  'semantic_id': tracking_seman_id, 'semantic_color': tracking_seman_color.to(tracking_color.dtype),
                                  'id': iter_time_idx,'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        ##################################################
        ### Tracking
        ##################################################
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, self.params_opt_exclude, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, device=self.device,
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter,load_semantics=self.load_semantics)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True, load_semantics=self.load_semantics)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        color_map = torch.from_numpy(self.colormap.copy()/255.).to(self.device)
                        if config['use_wandb']:
                            report_progress(params, color_map, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            device=self.device, load_semantics=self.load_semantics,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, color_map, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            device=self.device, load_semantics=self.load_semantics)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.tracking_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_count += 1

        if (time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0) and not config['tracking']['use_gt_poses']:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    color_map = torch.from_numpy(self.colormap.copy() / 255.).to(self.device)
                    if config['use_wandb']:
                        report_progress(params,color_map, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        device=self.device, load_semantics=self.load_semantics,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params,color_map, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        device=self.device, load_semantics=self.load_semantics)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            self.update_global_keyframe_set_completeness(
                depth, c2w, 
                self.slam_cfg.global_keyframe.completeness_thre, 
                time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
            )

        ##################################################
        ### Densification & KeyFrame-based Mapping
        ##################################################
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0 or force_map_update:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # resize RGBD frames for densification
                    densify_h, densify_w = self.config['data']["densification_image_height"], self.config['data']["densification_image_width"]
                    densify_color = F.interpolate(color.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]
                    densify_depth = F.interpolate(depth.unsqueeze(0), (densify_h, densify_w), mode='nearest')[0]
                    densify_seman_id = F.interpolate(seman_id.unsqueeze(0).float(), (densify_h, densify_w), mode='nearest')[0].long()
                    densify_seman_color = F.interpolate(seman_color.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]

                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx,
                                         'semantic_id': densify_seman_id, 'semantic_color': densify_seman_color.to(densify_color.dtype),
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, self.params_opt_exclude,variables, densify_curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], self.device, load_semantics=self.load_semantics)
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                ##################################################
                ### Select Keyframes for Mapping
                ##################################################
                ### overlap keyframes ###
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                if PRINT_INFO:
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    if self.slam_cfg.use_global_keyframe:
                        global_keyframe_time_indices = [frame_idx for frame_idx in self.global_keyframe_time_indices if frame_idx != time_idx]
                        print(f"\nGlobal Keyframes at Frame {time_idx}: {global_keyframe_time_indices}")
                
            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, self.params_opt_exclude, config['mapping']['lrs'], tracking=False)

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                ##################################################
                ### frame selection for map update
                ##################################################
                ### Overlap Keyframe ###
                # Randomly select a frame until current time step amongst keyframes

                if only_use_global_keyframe or (self.slam_cfg.use_global_keyframe and iter > num_iters_mapping // 2):
                    # selected_keyframes = [i for i in range(len(self.global_keyframe_indices))]
                    ### Global Keyframe ###
                    # rand_idx = np.random.randint(0, len(selected_keyframes))
                    if len(self.global_keyframe_indices) == 1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_seman_color = seman_color
                        iter_seman_id = seman_id
                    else:
                        selected_rand_keyframe_idx = np.random.choice(self.global_keyframe_indices[:-1])
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_seman_color = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                        iter_seman_id = keyframe_list[selected_rand_keyframe_idx]['semantic_id']
                else:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_seman_color = seman_color
                        iter_seman_id = seman_id
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_seman_color = keyframe_list[selected_rand_keyframe_idx]['semantic_color']
                        iter_seman_id = keyframe_list[selected_rand_keyframe_idx]['semantic_id']

                
                iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx,
                             'semantic_color': iter_seman_color.to(iter_color.dtype), 'semantic_id': iter_seman_id,
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True,
                                                   device=self.device, load_semantics=self.load_semantics)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True,load_semantics=self.load_semantics)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, self.params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'],
                                                    self.params_opt_exclude, device=self.device)
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        color_map = torch.from_numpy(self.colormap.copy() / 255.).to(self.device)
                        if config['use_wandb']:
                            report_progress(params,color_map, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True,device=self.device, load_semantics=self.load_semantics, online_time_idx=time_idx)
                        else:
                            report_progress(params, color_map, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            mapping=True, device=self.device, load_semantics=self.load_semantics,online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.mapping_iter_time_sum += iter_end_time - iter_start_time
                self.mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
            self.mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0 and not config['tracking']['use_gt_poses']:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        color_map = torch.from_numpy(self.colormap.copy() / 255.).to(self.device)
                        if config['use_wandb']:
                            report_progress(params, color_map,curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=self.device, load_semantics=self.load_semantics,online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, color_map, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'],
                                            eval_dir=self.eval_dir,
                                            mapping=True, device=self.device, load_semantics=self.load_semantics,online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')


        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            quality_method = self.slam_cfg.global_keyframe.get("quality_method", "absolute")
            if quality_method == "absolute":
                self.update_global_keyframe_set_quality(
                    color, depth, c2w, 
                    self.slam_cfg.global_keyframe.color_thre, 
                    self.slam_cfg.global_keyframe.depth_thre, 
                    time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
                )
            elif quality_method == "relative":
                if time_idx > 0 and time_idx % self.slam_cfg.global_keyframe.quality_freq == 0:
                    self.update_global_keyframe_set_quality_rel()
            else:
                raise NotImplementedError
        
        # Add frame to keyframe list
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth,
                                     'semantic_id': seman_id, 'semantic_color':seman_color}
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)

        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            save_semantic_ply(params, ckpt_output_dir, time_idx)
            save_rgb_ply(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()
        
        ##################################################
        ### update self variables
        ##################################################
        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam
        self.seperate_densification_res = seperate_densification_res
        if self.seperate_densification_res:
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        self.config = config
        self.gt_w2c_all_frames = gt_w2c_all_frames
        if self.config['use_wandb']:
            self.wandb_run = wandb_run
            self.wandb_mapping_step = wandb_mapping_step
            self.wandb_time_step = wandb_time_step
        self.eval_dir = eval_dir
        self.seperate_tracking_res = seperate_tracking_res
        if self.seperate_tracking_res:
            self.tracking_cam = tracking_cam
            self.tracking_intrinsics = tracking_intrinsics
        
        self.keyframe_list = keyframe_list
        self.num_frames = num_frames
        self.keyframe_time_indices = keyframe_time_indices

    def print_and_save_result(self, eval_dir_suffix="", is_prune_gaussians=False, ignore_first_frame=False):
        """ evaluate rendering results and save result
        """
        ### get self variables ###

        params = self.params.copy()
        params_opt_exclude = set('semantic_ids')
        variables = self.variables.copy()
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        tracking_iter_time_sum = self.tracking_iter_time_sum
        tracking_frame_time_sum = self.tracking_frame_time_sum
        tracking_iter_time_count = self.tracking_iter_time_count
        tracking_frame_time_count = self.tracking_frame_time_count
        mapping_iter_time_sum = self.mapping_iter_time_sum
        mapping_frame_time_sum = self.mapping_frame_time_sum
        mapping_iter_time_count = self.mapping_iter_time_count
        mapping_frame_time_count = self.mapping_frame_time_count
        config = self.config
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        dataset_config = self.config['data']
        gt_w2c_all_frames = self.gt_w2c_all_frames
        keyframe_time_indices = self.keyframe_time_indices
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        ### prune gaussians ###
        if is_prune_gaussians:
            optimizer = initialize_optimizer(params, params_opt_exclude, config['mapping']['lrs'], tracking=False)
            params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, 0, config['mapping']['pruning_dict'])


        # Compute Average Runtimes
        if tracking_iter_time_count == 0:
            tracking_iter_time_count = 1
            tracking_frame_time_count = 1
        if mapping_iter_time_count == 0:
            mapping_iter_time_count = 1
            mapping_frame_time_count = 1
        tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        if config['use_wandb']:
            wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})
        
        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)
            else:
                eval(dataset, params, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)

        # Add Camera Parameters to Save them
        params['timestep'] = variables['timestep']
        params['intrinsics'] = intrinsics.detach().cpu().numpy()
        params['w2c'] = first_frame_w2c.detach().cpu().numpy()
        params['org_width'] = dataset_config["desired_image_width"]
        params['org_height'] = dataset_config["desired_image_height"]
        params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
        params['keyframe_time_indices'] = np.array(keyframe_time_indices)
        
        # Save Parameters
        results_dir = os.path.join(self.results_dir, eval_dir_suffix) if eval_dir_suffix else self.results_dir
        save_params(params, results_dir)

    def load_params_by_step(self, step=1100, stage='final'):
        """ load checkpoint parameters
        Attributes:
            params: SplaTAM parameters
        """
        ### load self variables ###
        config = self.config
        checkpoint_time_idx = step
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"{stage}/params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        self.params = params
        self.params_opt_exclude = set('semantic_ids')

    def eval_semantic_result(self, eval_dir_suffix="", ignore_first_frame=False, save_frames=False):
        """ evaluate rendering results

        """
        ### get self variables ###
        params = self.params
        config = self.config

        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        save_semantic_ply(params, eval_dir, 2000)

        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval_semantic(self, dataset, params,  num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame)
            else:
                eval_semantic(self, dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame, save_frames=save_frames)
        return


