import cv2
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.utils.general_utils import *
from src.slam.semsplatam.modified_ver.splatam.eval_helper import calc_miou
from imgviz import label_colormap


sys.path.append("third_parties/splatam")
from datasets.gradslam_datasets.geometryutils import relative_transformation
from utils.recon_helpers import setup_camera
from utils.slam_external import build_rotation, calc_psnr
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    quat_mult, matrix_to_quaternion
)
from utils.eval_helpers import evaluate_ate
from src.slam.splatam.eval_helper import transform_to_frame

loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
from src.slam.sgsslam.modified_ver.scripts.sgsslam import transformed_semantics2rendervar
from src.slam.semsplatam.modified_ver.splatam.eval_helper import calc_miou,calc_topk_acc, calc_mAP,calc_f1, resize_tensor

def recolor_semantic_img(rendered_seg, gt_seg, gt_id, color_map=None):
    """Adjust the semantic color by assigning to the closest color refer to
       the ground truth semantic image or color dict.
    """
    rendered_seg = rendered_seg.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    img_shape = gt_seg.shape
    rendered_seg = rendered_seg.reshape(-1, 1, 3).type(torch.float64)  # (H*W, 1, 3)

    if color_map is None:
        gt_seg = gt_seg.reshape(-1, 3)
        # Find unique colors
        color_map, indices = torch.unique(gt_seg, dim=0, return_inverse=True)
    refer_color = color_map.reshape(1, -1, 3).type(torch.float64).to(gt_seg.device)  # (1, H*W, 3)
    # l1_distances = torch.sum(torch.abs(rendered_seg - refer_color), axis=2)
    l1_distances = torch.sqrt(torch.sum((rendered_seg - refer_color) ** 2, axis=2))
    # Find the index of the minimum distance for each pixel
    closest_indices = torch.argmin(l1_distances, axis=1)
    del l1_distances

    # Assign the closest color to the rendered semantic image
    rendered_seg[:, 0, :] = refer_color.squeeze(0)[closest_indices]
    rendered_seg = rendered_seg.squeeze().reshape(img_shape)  # (H*W, 1, 3) -> (H, W, 3)
    rendered_seg = rendered_seg.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

    return rendered_seg

def recolor_semantic_img_v2(rendered_seg, gt_seg, gt_id, color_map=None, n_class = 102):
    """Adjust the semantic color by assigning to the closest color refer to
       the ground truth semantic image or color dict.
    """
    rendered_seg = rendered_seg.permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    img_shape = gt_seg.shape
    rendered_seg = rendered_seg.reshape(-1, 1, 3).type(torch.float32)  # (H*W, 1, 3)

    if color_map is None:
        gt_seg = gt_seg.reshape(-1, 3)
        gt_id = gt_id.reshape(-1,1)
        # Find unique colors
        gt = torch.cat((gt_seg,gt_id), dim=1)
        color_id_map, indices = torch.unique(gt, dim=0, return_inverse=True)
        color_map = color_id_map[:, :3]
        id_map = color_id_map[:, 3].long().reshape(-1,1)
    else:
        id_map = torch.arange(len(color_map)).reshape(-1,1)
    refer_color = color_map.reshape(1, -1, 3).type(torch.float32).to(gt_seg.device)  # (1, H*W, 3)
    refer_id = id_map.reshape(1,-1, 1).to(gt_seg.device)
    # l1_distances = torch.sum(torch.abs(rendered_seg - refer_color), axis=2)
    l1_distances = torch.sqrt(torch.sum((rendered_seg - refer_color) ** 2, axis=2))
    # Find the index of the minimum distance for each pixel
    closest_indices = torch.argmin(l1_distances, axis=1)
    if color_map is None:
        rendered_logits = torch.zeros((img_shape[0]*img_shape[1], n_class)).to(l1_distances)
        rendered_logits[..., id_map.squeeze()] = F.softmax(-l1_distances.clone(),dim=-1)
    else:
        rendered_logits = F.softmax(-l1_distances.clone(),dim=-1)
    del l1_distances

    # Assign the closest color to the rendered semantic image
    rendered_seg[:, 0, :] = refer_color.squeeze(0)[closest_indices]
    rendered_seg = rendered_seg.squeeze().reshape(img_shape)  # (H*W, 1, 3) -> (H, W, 3)
    rendered_seg = rendered_seg.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

    rendered_id = refer_id.squeeze(0)[closest_indices]
    rendered_id = rendered_id.squeeze().reshape(img_shape[0], img_shape[1]) # h,w
    rendered_logits = rendered_logits.reshape(img_shape[0], img_shape[1], -1).permute(2,0,1) # C, H,W
    return rendered_seg, rendered_id, rendered_logits

def report_loss(losses, wandb_run, wandb_step, tracking=False, mapping=False, load_semantics=False):
    # Update loss dict
    loss_dict = {'Loss': losses['loss'].item(),
                 'Image Loss': losses['im'].item(),
                 'Depth Loss': losses['depth'].item(), }
    if load_semantics:
        loss_dict['Semantic Loss'] = losses['seg'].item()

    if tracking:
        tracking_loss_dict = {}
        for k, v in loss_dict.items():
            tracking_loss_dict[f"Per Iteration Tracking/{k}"] = v
        tracking_loss_dict['Per Iteration Tracking/step'] = wandb_step
        wandb_run.log(tracking_loss_dict)
    elif mapping:
        mapping_loss_dict = {}
        for k, v in loss_dict.items():
            mapping_loss_dict[f"Per Iteration Mapping/{k}"] = v
        mapping_loss_dict['Per Iteration Mapping/step'] = wandb_step
        wandb_run.log(mapping_loss_dict)
    else:
        frame_opt_loss_dict = {}
        for k, v in loss_dict.items():
            frame_opt_loss_dict[f"Per Iteration Current Frame Optimization/{k}"] = v
        frame_opt_loss_dict['Per Iteration Current Frame Optimization/step'] = wandb_step
        wandb_run.log(frame_opt_loss_dict)

    # Increment wandb step
    wandb_step += 1
    return wandb_step


def plot_rgbd_silhouette(color, depth, rastered_color, rastered_depth, presence_sil_mask, diff_depth_l1,
                         psnr, depth_l1, fig_title, plot_dir=None, plot_name=None, save_plot=False, seg=None,seg_id=None,color_map=None,
                         rastered_seg=None, wandb_run=None, wandb_step=None, wandb_title=None, diff_rgb=None):
    # Determine Plot Aspect Ratio
    aspect_ratio = color.shape[2] / color.shape[1]
    fig_height = 8
    fig_width = 14 / 1.55
    # Adjust number of subplots and figure size based on 'seg' variable
    num_cols = 4 if seg is not None else 3
    # Scale width for additional column if seg is not None
    fig_width = fig_width * aspect_ratio * num_cols / 3
    # Plot the Ground Truth and Rasterized RGB & Depth,
    # along with Diff Depth & Silhouette, and semantic image
    fig, axs = plt.subplots(2, num_cols, figsize=(fig_width, fig_height))
    axs[0, 0].imshow(color.cpu().permute(1, 2, 0))
    axs[0, 0].set_title("Ground Truth RGB")
    axs[0, 1].imshow(depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[0, 1].set_title("Ground Truth Depth")
    rastered_color = torch.clamp(rastered_color, 0, 1)
    axs[1, 0].imshow(rastered_color.cpu().permute(1, 2, 0))
    axs[1, 0].set_title("Rasterized RGB, PSNR: {:.2f}".format(psnr))
    axs[1, 1].imshow(rastered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=6)
    axs[1, 1].set_title("Rasterized Depth, L1: {:.2f}".format(depth_l1))
    if diff_rgb is not None:
        axs[0, 2].imshow(diff_rgb.cpu(), cmap='jet', vmin=0, vmax=6)
        axs[0, 2].set_title("Diff RGB L1")
    else:
        axs[0, 2].imshow(presence_sil_mask, cmap='gray')
        axs[0, 2].set_title("Rasterized Silhouette")
    diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
    axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=6)
    axs[1, 2].set_title("Diff Depth L1")

    if seg is not None:
        rastered_seg, render_id = recolor_semantic_img(rastered_seg, seg, seg_id, color_map)
        miou = calc_miou(pred=render_id,target=seg_id.long())
        # miou = calc_miou(rastered_seg, seg)
        axs[0, 3].imshow(seg.cpu().permute(1, 2, 0))
        axs[0, 3].set_title("Ground Truth Semantic Map")
        axs[1, 3].imshow(rastered_seg.cpu().permute(1, 2, 0))
        axs[1, 3].set_title("Rasterized Semantic Map, IOU: {:.4f}".format(miou))

    for ax in axs.flatten():
        ax.axis('off')
    fig.suptitle(fig_title, y=0.95, fontsize=16)
    fig.tight_layout()
    if save_plot:
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
    if wandb_run is not None:
        if wandb_step is None:
            wandb_run.log({wandb_title: fig})
        else:
            wandb_run.log({wandb_title: fig}, step=wandb_step)
    plt.close()


def report_progress(params, color_map, data, i, progress_bar, iter_time_idx, sil_thres, every_i=1, qual_every_i=1,
                    tracking=False, mapping=False, device="cuda", load_semantics=False, wandb_run=None,
                    wandb_step=None, wandb_save_qual=False, online_time_idx=None, global_logging=True):
    if i % every_i == 0 or i == 1:
        if wandb_run is not None:
            if tracking:
                stage = "Tracking"
            elif mapping:
                stage = "Mapping"
            else:
                stage = "Current Frame Optimization"
        if not global_logging:
            stage = "Per Iteration " + stage

        if tracking:
            # Get list of gt poses
            gt_w2c_list = data['iter_gt_w2c_list']
            valid_gt_w2c_list = []

            # Get latest trajectory
            latest_est_w2c = data['w2c']
            latest_est_w2c_list = []
            latest_est_w2c_list.append(latest_est_w2c)
            valid_gt_w2c_list.append(gt_w2c_list[0])
            for idx in range(1, iter_time_idx + 1):
                # Check if gt pose is not nan for this time step
                if torch.isnan(gt_w2c_list[idx]).sum() > 0:
                    continue
                interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
                interm_cam_trans = params['cam_trans'][..., idx].detach()
                intermrel_w2c = torch.eye(4).to(device).float()
                intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
                intermrel_w2c[:3, 3] = interm_cam_trans
                latest_est_w2c = intermrel_w2c
                latest_est_w2c_list.append(latest_est_w2c)
                valid_gt_w2c_list.append(gt_w2c_list[idx])

            # Get latest gt pose
            gt_w2c_list = valid_gt_w2c_list
            iter_gt_w2c = gt_w2c_list[-1]
            # Get euclidean distance error between latest and gt pose
            iter_pt_error = torch.sqrt(
                (latest_est_w2c[0, 3] - iter_gt_w2c[0, 3]) ** 2 + (latest_est_w2c[1, 3] - iter_gt_w2c[1, 3]) ** 2 + (
                            latest_est_w2c[2, 3] - iter_gt_w2c[2, 3]) ** 2)
            if iter_time_idx > 0:
                # Calculate relative pose error
                rel_gt_w2c = relative_transformation(gt_w2c_list[-2], gt_w2c_list[-1])
                rel_est_w2c = relative_transformation(latest_est_w2c_list[-2], latest_est_w2c_list[-1])
                rel_pt_error = torch.sqrt(
                    (rel_gt_w2c[0, 3] - rel_est_w2c[0, 3]) ** 2 + (rel_gt_w2c[1, 3] - rel_est_w2c[1, 3]) ** 2 + (
                                rel_gt_w2c[2, 3] - rel_est_w2c[2, 3]) ** 2)
            else:
                rel_pt_error = torch.zeros(1).float()

            # Calculate ATE RMSE
            ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
            ate_rmse = np.round(ate_rmse, decimals=6)
            if wandb_run is not None:
                tracking_log = {f"{stage}/Latest Pose Error": iter_pt_error,
                                f"{stage}/Latest Relative Pose Error": rel_pt_error,
                                f"{stage}/ATE RMSE": ate_rmse}

        # Get current frame Gaussians
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=False,
                                             camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_pts)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, data['w2c'],
                                                                     transformed_pts)
        depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (data['depth'] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)

        if load_semantics:
            semantic_rendervar = transformed_semantics2rendervar(params, transformed_pts, device=device)
            rastered_seg, _, _, = Renderer(raster_settings=data['cam'])(**semantic_rendervar)
            gt_seg = data['semantic_color']
            gt_id = data['semantic_id']
            # seg_psnr = calc_psnr(seg, data['semantic_color']).mean()
            rastered_seg = recolor_semantic_img(rastered_seg, gt_seg, gt_id, color_map)

            miou = calc_miou_v2(pred_color=rastered_seg.permute(1,2,0),target_color=gt_seg,target_id=gt_id)
        else:
            rastered_seg = None
            gt_seg = None
            miou = 0

        if tracking:
            psnr = calc_psnr(im * presence_sil_mask, data['im'] * presence_sil_mask).mean()
        else:
            psnr = calc_psnr(im, data['im']).mean()

        if tracking:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth']) * presence_sil_mask) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']) * presence_sil_mask)
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
        else:
            diff_depth_rmse = torch.sqrt((((rastered_depth - data['depth'])) ** 2))
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()
            diff_depth_l1 = torch.abs((rastered_depth - data['depth']))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        if not (tracking or mapping):
            progress_bar.set_postfix({
                                         f"Time-Step: {iter_time_idx} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | mIoU: {miou:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)
        elif tracking:
            progress_bar.set_postfix({
                                         f"Time-Step: {iter_time_idx} | Rel Pose Error: {rel_pt_error.item():.{7}} | Pose Error: {iter_pt_error.item():.{7}} | ATE RMSE": f"{ate_rmse.item():.{7}}"})
            progress_bar.update(every_i)
        elif mapping:
            progress_bar.set_postfix({
                                         f"Time-Step: {online_time_idx} | Frame {data['id']} | PSNR: {psnr:.{7}} | Depth RMSE: {rmse:.{7}} | mIoU: {miou:.{7}} | L1": f"{depth_l1:.{7}}"})
            progress_bar.update(every_i)

        if wandb_run is not None:
            wandb_log = {f"{stage}/PSNR": psnr,
                         f"{stage}/Depth RMSE": rmse,
                         f"{stage}/Depth L1": depth_l1,
                         f"{stage}/mIoU": miou,
                         f"{stage}/step": wandb_step}
            if tracking:
                wandb_log = {**wandb_log, **tracking_log}
            wandb_run.log(wandb_log)

        if wandb_save_qual and (i % qual_every_i == 0 or i == 1):
            # Silhouette Mask
            presence_sil_mask = presence_sil_mask.detach().cpu().numpy()

            # Log plot to wandb
            if not mapping:
                fig_title = f"Time-Step: {iter_time_idx} | Iter: {i} | Frame: {data['id']}"
            else:
                fig_title = f"Time-Step: {online_time_idx} | Iter: {i} | Frame: {data['id']}"
            plot_rgbd_silhouette(data['im'], data['depth'], im, rastered_depth, presence_sil_mask, diff_depth_l1,
                                 psnr, depth_l1, fig_title, seg=gt_seg, seg_id=gt_id, color_map=color_map, rastered_seg=rastered_seg, wandb_run=wandb_run,
                                 wandb_step=wandb_step, wandb_title=f"{stage} Qual Viz")


def calc_miou_v2(pred_color: torch.Tensor, target_color: torch.Tensor, target_id: torch.Tensor) -> float:
    """
    Compute mean Intersection over Union (mIoU) between a predicted mask and a target mask.
    Only considers classes present in the target mask.

    Args:
        pred (torch.Tensor): Predicted semantic mask of shape (H, W).
        target (torch.Tensor): Target semantic mask of shape (H, W).

    Returns:
        float: Mean IoU score.
    """
    pred_flat = pred_color.reshape(-1,3).to(target_color.dtype)  # (H*W, C)
    target_flat = target_color.reshape(-1,3).to(pred_flat.device)  # (H*W,)
    target_id_flat = target_id.reshape(-1).to(pred_color.device)

    # pred_flat = torch.round(pred_flat * 1e8) / 1e8
    # target_flat = torch.round(target_flat * 1e8) / 1e8

    # Only consider valid pixels (non-zero target)
    valid_mask = (target_id_flat != 0)

    pred = pred_flat[valid_mask].type(torch.float64)
    target = target_flat[valid_mask].type(torch.float64)

    unique_colors = torch.unique(target, dim=0).type(torch.float64)
    iou_per_color = []

    for color in unique_colors:
        target_matches = torch.all(target == color, dim=1)
        pred_matches = torch.all(pred == color, dim=1)

        # Calculate intersection and union
        intersection = (target_matches & pred_matches).sum().float()
        union = (pred_matches | target_matches).sum().float()

        if union == 0:
            iou = torch.tensor(float('nan'))  # Class not present in prediction and ground truth
        else:
            iou = intersection / union

        iou_per_color.append(iou)

    iou_per_color = torch.stack(iou_per_color)
    miou = torch.nanmean(iou_per_color).item()
    return miou

@torch.no_grad()
def eval_semantic(slam_model, dataset, final_params, num_frames, eval_dir, sil_thres,
         mapping_iters, add_new_gaussians, wandb_run=None, wandb_save_qual=False, eval_every=1, save_frames=False,
         ignore_first_frame=False):
    print("Evaluating Final Parameters ...")
    miou_g_list = []
    miou_p_list = []
    miou_g_curr_list = []
    top1_g_list = []
    top3_g_list = []
    top5_g_list = []
    mAP_g_list = []
    f1_g_list = []

    plot_dir = os.path.join(eval_dir, "semantic_plots")
    os.makedirs(plot_dir, exist_ok=True)
    if save_frames:
        seman_dir = os.path.join(eval_dir, "seman")
        os.makedirs(seman_dir, exist_ok=True)

    gt_w2c_list = []
    num_frames = len(dataset)
    # sem_colormap = create_class_colormap(slam_model.n_cls)
    sem_colormap = label_colormap(slam_model.n_cls)

    for time_idx in tqdm(range(num_frames)):
        # Get RGB-D Data & Camera Parameters
        color, _, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]

        seman_gt_id = dataset.get_semantic_map(time_idx)
        seman_gt_color = apply_colormap(seman_gt_id[0].cpu().long().numpy(), sem_colormap) / 255.
        seman_gt_color = torch.from_numpy(seman_gt_color).to(slam_model.device)
        seg_img = color.clone().to(slam_model.semantic_device)
        seman_pseudo_id, seman_pseudo_color = slam_model.semantic_annotation(seg_img)
        seman_pseudo_id = seman_pseudo_id.to(slam_model.device)
        seman_pseudo_color = seman_pseudo_color.to(slam_model.device)
        n_cls = slam_model.n_cls

        if time_idx == 0:
            # Process Camera Parameters
            first_frame_w2c = torch.linalg.inv(pose)
            # Setup Camera
            cam = setup_camera(color.shape[1], color.shape[0], intrinsics.cpu().numpy(),
                               first_frame_w2c.detach().cpu().numpy())

        # Skip frames if not eval_every
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        # Get current frame Gaussians
        # transformed_gaussians = transform_to_frame(final_params, time_idx,
        #                                            gaussians_grad=False,
        #                                            camera_grad=False)
        transformed_gaussians = transform_to_frame(final_params, time_idx,
                                                   gaussians_grad=False,
                                                   camera_grad=False,
                                                   rel_w2c=gt_w2c
                                                   )

        # Define current frame data
        curr_data = {'cam': cam, 'im': color,
                     'seman_gt_id': seman_gt_id[0].long(), 'seman_gt_color': seman_gt_color,
                     'seman_pseudo_id':seman_pseudo_id.long(), 'seman_pseudo_color': seman_pseudo_color,
                     'id': time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c}
        # Initialize Render Variables
        semantic_rendervar = transformed_semantics2rendervar(final_params, transformed_gaussians, device=slam_model.device)
        depth_sil_rendervar = transformed_params2depthplussilhouette(final_params, curr_data['w2c'],
                                                                     transformed_gaussians)
        depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
        rastered_seg, _, depth = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
        rastered_seg = torch.clamp(rastered_seg, min=0, max=1)

        recolored_seg_curr = recolor_semantic_img(rastered_seg.clone(), curr_data['seman_gt_color'],
                                                  curr_data['seman_gt_id'], color_map=None)
        miou_g_curr = calc_miou_v2(pred_color=recolored_seg_curr.permute(1, 2, 0),
                                   target_color=curr_data['seman_gt_color'],
                                   target_id=curr_data['seman_gt_id'])
        miou_g_curr_list.append(miou_g_curr)

        color_map = torch.from_numpy(sem_colormap.copy() / 255.).to(rastered_seg.device)
        recolored_seg, rendered_id, rendered_logits = recolor_semantic_img_v2(rastered_seg.clone(), curr_data['seman_gt_color'], curr_data['seman_gt_id'], color_map.clone())
        #_, _, rendered_logits = recolor_semantic_img_v2(rastered_seg.clone(),curr_data['seman_gt_color'],curr_data['seman_gt_id'],color_map=None)

        miou_g = calc_miou(pred=rendered_id.long(), target=curr_data['seman_gt_id'].long())
        miou_p = calc_miou(pred=curr_data['seman_pseudo_id'].long(), target=curr_data['seman_gt_id'].long())
        miou_g_list.append(miou_g)
        miou_p_list.append(miou_p)

        f1_g = calc_f1(pred=rendered_id.long(), target=curr_data['seman_gt_id'].long())
        f1_g_list.append(f1_g)

        topks_g = calc_topk_acc(pred_logits=rendered_logits.permute(1,2,0), target=curr_data['seman_gt_id'].long(), topk=(1, 3, 5))
        mAP_g = calc_mAP(pred_logits=rendered_logits.permute(1,2,0), target=curr_data['seman_gt_id'].long())
        top1_g_list.append(topks_g[0])
        top3_g_list.append(topks_g[1])
        top5_g_list.append(topks_g[2])
        mAP_g_list.append(mAP_g)

        gt_seman_rgb = curr_data['seman_gt_color'].cpu().numpy()
        pseudo_seman_rgb = curr_data['seman_pseudo_color'].cpu().numpy()
        rastered_seman_rgb = rastered_seg.cpu().numpy()
        recolored_seman_rgb = recolored_seg.cpu().numpy()
        recolored_seman_rgb_curr = recolored_seg_curr.cpu().numpy()

        # Plot the Ground Truth, Pseudo and Rasterized semantic RGB
        fig_title = "Time Step: {}".format(time_idx)
        plot_name = "%04d" % time_idx
        target_res = (256, 256)
        mode = 'nearest'

        # save original one
        # save_path = os.path.join(plot_dir, f"{plot_name}_gt.png")
        # cv2.imwrite(save_path, cv2.cvtColor(gt_seman_rgb, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
        save_path = os.path.join(plot_dir, f"{plot_name}_render_original.png")
        original_render_rgb = torch.from_numpy(rastered_seman_rgb).permute(1, 2, 0).cpu().numpy() * 255
        original_render_rgb = original_render_rgb.astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(original_render_rgb, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        save_path = os.path.join(plot_dir, f"{plot_name}_recolor_curr_miou_{miou_g_curr:.4f}.png")
        render_seman_rgb_curr = torch.from_numpy(recolored_seman_rgb_curr).permute(1, 2, 0).cpu().numpy() * 255
        render_seman_rgb_curr = render_seman_rgb_curr.astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(render_seman_rgb_curr, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        save_path = os.path.join(plot_dir, f"{plot_name}_recolor_global_miou_{miou_g:.4f}.png")
        render_seman_rgb_global = torch.from_numpy(recolored_seman_rgb).permute(1, 2, 0).cpu().numpy() * 255
        render_seman_rgb_global = render_seman_rgb_global.astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(render_seman_rgb_global, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

        gt_seman_rgb = torch.from_numpy(gt_seman_rgb).permute(2, 0, 1)  # 3, H ,W
        gt_seman = resize_tensor(gt_seman_rgb, *target_res, mode=mode)
        gt_seman_np = gt_seman.permute(1, 2, 0).cpu().numpy() * 255

        pseudo_seman_rgb = torch.from_numpy(pseudo_seman_rgb).permute(2, 0, 1)  # 3, H ,W
        pseudo_seman = resize_tensor(pseudo_seman_rgb, *target_res, mode=mode)
        pseudo_seman_np = pseudo_seman.permute(1, 2, 0).cpu().numpy() * 255

        rastered_seman_rgb = torch.from_numpy(rastered_seman_rgb)  # 3, H ,W
        rastered_seman = resize_tensor(rastered_seman_rgb, *target_res, mode=mode)
        rastered_seman_np = rastered_seman.permute(1, 2, 0).cpu().numpy() * 255

        recolored_seman_rgb = torch.from_numpy(recolored_seman_rgb)  # 3, H ,W
        recolored_seman = resize_tensor(recolored_seman_rgb, *target_res, mode=mode)
        recolored_seman_np = recolored_seman.permute(1, 2, 0).cpu().numpy() * 255

        recolored_seman_rgb_curr = torch.from_numpy(recolored_seman_rgb_curr)  # 3, H ,W
        recolored_seman_curr = resize_tensor(recolored_seman_rgb_curr, *target_res, mode=mode)
        recolored_seman_np_curr = recolored_seman_curr.permute(1, 2, 0).cpu().numpy() * 255

        final_image = np.hstack([gt_seman_np,pseudo_seman_np,rastered_seman_np, recolored_seman_np,recolored_seman_np_curr]).astype(np.uint8)
        save_path = os.path.join(plot_dir, f"{plot_name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    #
    # Compute Average Metrics
    miou_g_list = np.array(miou_g_list)
    miou_g_curr_list = np.array(miou_g_curr_list)
    miou_p_list = np.array(miou_p_list)
    top1_g_list = np.array(top1_g_list)
    top3_g_list = np.array(top3_g_list)
    top5_g_list = np.array(top5_g_list)
    mAP_g_list = np.array(mAP_g_list)
    f1_g_list = np.array(f1_g_list)


    avg_miou_g = miou_g_list.mean()
    avg_miou_g_curr = miou_g_curr_list.mean()
    avg_miou_p = miou_p_list.mean()
    avg_top1_g = top1_g_list.mean()
    avg_top3_g = top3_g_list.mean()
    avg_top5_g = top5_g_list.mean()
    avg_mAP_g = mAP_g_list.mean()
    avg_f1_g = f1_g_list.mean()

    print("Average MIOU with GT: {:.2f}".format(avg_miou_g * 100))
    print("Average MIOU with GT (current): {:.2f}".format(avg_miou_g_curr * 100))
    print("Average top1 acc with GT: {:.2f}".format(avg_top1_g * 100))
    print("Average top3 acc with GT: {:.2f}".format(avg_top3_g * 100))
    print("Average top5 acc with GT: {:.2f}".format(avg_top5_g * 100))
    print("Average top5 acc with GT: {:.2f}".format(avg_top5_g * 100))
    print("Average mAP with GT: {:.2f}".format(avg_mAP_g * 100))
    print("Average F1 with GT: {:.2f}".format(avg_f1_g * 100))

    print("Average MIOU with Pseudo: {:.2f}".format(avg_miou_p * 100))

    # Save metric lists as text files
    with open(os.path.join(eval_dir, "semantic_result.txt"), 'w') as f:
        lines = []
        lines.append(f"miou_g: {avg_miou_g * 100}\n")
        lines.append(f"miou_g_curr: {avg_miou_g_curr * 100}\n")
        lines.append(f"top1_g: {avg_top1_g * 100}\n")
        lines.append(f"top3_g: {avg_top3_g * 100}\n")
        lines.append(f"top5_g: {avg_top5_g * 100}\n")
        lines.append(f"mAP_g: {avg_mAP_g * 100}\n")
        lines.append(f"f1_g: {avg_f1_g * 100}\n")

        lines.append(f"miou_p: {avg_miou_p * 100}\n")
        f.writelines(lines)
