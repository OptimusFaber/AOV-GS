import os
import sys
sys.path.append(os.getcwd())

import argparse
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.naruto.cfg_loader import load_cfg
from src.slam import init_SLAM_model
from src.utils.general_utils import InfoPrinter


def argument_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--stage", type=str, default='final')
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--topk_threshold", type=int, default=16)
    parser.add_argument("--pixel_threshold", type=float, default=0.1)
    parser.add_argument("--min_top1_gap", type=float, default=0.2)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_heatmaps", action='store_true')
    return parser.parse_args()


def analyze_topk_predictions(pred_logits, target, topk_sizes=[1, 3, 5, 16]):
    H, W, C = pred_logits.shape
    pred_logits_flat = pred_logits.view(-1, C)
    target_flat = target.view(-1)
    
    valid_mask = (target_flat != 0)
    if valid_mask.sum() == 0:
        return None
    
    pred_logits_valid = pred_logits_flat[valid_mask]
    target_valid = target_flat[valid_mask]
    
    max_k = max(topk_sizes)
    topk_values, topk_indices = pred_logits_valid.topk(max_k, dim=1, largest=True, sorted=True)
    
    results = {}
    for k in topk_sizes:
        topk_k = topk_indices[:, :k]
        correct_mask = topk_k.eq(target_valid.unsqueeze(1)).any(dim=1)
        failure_mask = ~correct_mask
        
        results[f'top{k}'] = {
            'accuracy': correct_mask.float().mean().item(),
            'failure_mask': failure_mask.cpu().numpy(),
            'failure_count': failure_mask.sum().item(),
            'failure_ratio': failure_mask.float().mean().item()
        }
    
    failure_mask_2d = torch.zeros(H * W, dtype=torch.bool)
    failure_mask_2d[valid_mask] = torch.from_numpy(results[f'top{topk_sizes[0]}']['failure_mask'])
    failure_mask_2d = failure_mask_2d.view(H, W).cpu().numpy()
    
    target_positions = []
    for i, target_cls in enumerate(target_valid):
        position = (topk_indices[i] == target_cls).nonzero(as_tuple=True)
        if len(position[0]) > 0:
            target_positions.append(position[0][0].item() + 1)
        else:
            target_positions.append(max_k + 1)
    
    results['target_positions'] = np.array(target_positions)
    results['failure_mask_2d'] = failure_mask_2d
    results['topk_indices'] = topk_indices.cpu().numpy()
    results['topk_values'] = topk_values.cpu().numpy()
    results['valid_mask_2d'] = valid_mask.view(H, W).cpu().numpy()
    
    return results


def create_failure_visualization(image, gt_semantic, pred_semantic, pred_logits, 
                                 failure_analysis, colormap, class_names=None, frame_idx=0):
    H, W = gt_semantic.shape
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title(f'Frame {frame_idx}: Original', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    gt_colored = apply_colormap(gt_semantic, colormap)
    ax2.imshow(gt_colored)
    ax2.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    pred_colored = apply_colormap(pred_semantic, colormap)
    ax3.imshow(pred_colored)
    ax3.set_title('Prediction (Top-1)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    failure_mask = failure_analysis['failure_mask_2d']
    ax4.imshow(failure_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax4.set_title(f'Failure Mask ({failure_mask.mean()*100:.1f}%)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, :2])
    valid_mask = failure_analysis['valid_mask_2d']
    target_positions_2d = np.zeros((H, W))
    target_positions_2d[valid_mask] = failure_analysis['target_positions']
    
    im5 = ax5.imshow(target_positions_2d, cmap='RdYlGn_r', vmin=1, vmax=17)
    ax5.set_title('Correct Class Position', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2:])
    positions = failure_analysis['target_positions']
    bins = np.arange(1, 19)
    hist, _ = np.histogram(positions, bins=bins)
    ax6.bar(bins[:-1], hist, width=0.8, edgecolor='black', linewidth=1)
    ax6.set_xlabel('Position', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Pixels', fontsize=11, fontweight='bold')
    ax6.set_title('Position Distribution', fontsize=12, fontweight='bold')
    ax6.set_xticks(bins[:-1])
    ax6.grid(axis='y', alpha=0.3)
    
    for i, (bin_start, count) in enumerate(zip(bins[:-1], hist)):
        if count > 0:
            ax6.text(bin_start, count, str(int(count)), ha='center', va='bottom', fontsize=9)
    
    failure_pixels = failure_mask.flatten()
    if failure_pixels.sum() > 0:
        pred_classes_at_failures = pred_semantic.flatten()[failure_pixels]
        unique_pred, counts = np.unique(pred_classes_at_failures, return_counts=True)
        top_confused_classes = unique_pred[np.argsort(-counts)][:4]
        
        for idx, cls in enumerate(top_confused_classes):
            ax = fig.add_subplot(gs[2 + idx // 2, idx % 2])
            cls_heatmap = pred_logits[:, :, int(cls)]
            im = ax.imshow(cls_heatmap.cpu().numpy(), cmap='hot')
            class_name = class_names[int(cls)] if class_names else f"Class {int(cls)}"
            ax.set_title(f'{class_name} ({counts[np.where(unique_pred==cls)[0][0]]} px)', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2:])
    topk_accs = []
    topk_labels = []
    for k in [1, 3, 5, 16]:
        if f'top{k}' in failure_analysis:
            topk_accs.append(failure_analysis[f'top{k}']['accuracy'] * 100)
            topk_labels.append(f'Top-{k}')
    
    bars = ax11.bar(topk_labels, topk_accs, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'],
                    edgecolor='black', linewidth=2)
    ax11.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax11.set_title('Top-K Accuracies', fontsize=12, fontweight='bold')
    ax11.set_ylim([0, 105])
    ax11.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, topk_accs):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.1f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    total_pixels = valid_mask.sum()
    failure_count = failure_mask.sum()
    failure_ratio = failure_mask.mean() * 100
    top1_acc = failure_analysis['top1']['accuracy'] * 100
    top16_acc = failure_analysis['top16']['accuracy'] * 100
    acc_gap = top16_acc - top1_acc
    
    info_text = f"""Frame {frame_idx}
    
Pixels: {total_pixels}
Top-1 failures: {failure_count} ({failure_ratio:.1f}%)
Top-16 failures: {failure_analysis['top16']['failure_count']}

Top-1 Acc: {top1_acc:.1f}%
Top-16 Acc: {top16_acc:.1f}%
Gap: {acc_gap:.1f}%

{" Adaptive top-k could help" if acc_gap > 20 else ""}
    """
    
    ax12.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Failure Case - Frame {frame_idx}', fontsize=16, fontweight='bold', y=0.995)
    return fig


def apply_colormap(semantic_ids, colormap):
    H, W = semantic_ids.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for cls_id in np.unique(semantic_ids):
        if cls_id < len(colormap):
            mask = semantic_ids == cls_id
            rgb[mask] = colormap[int(cls_id)]
    
    return rgb


def load_class_names(class_info_file):
    if os.path.exists(class_info_file):
        with open(class_info_file, 'r') as f:
            return json.load(f)
    return None


def label_colormap(n_classes):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(n_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def find_latest_checkpoint(result_dir, stage='final'):
    import glob
    import re
    
    splatam_dir = os.path.join(result_dir, 'splatam')
    if not os.path.exists(splatam_dir):
        return None
    
    stage_params = os.path.join(splatam_dir, stage, 'params.npz')
    if os.path.exists(stage_params):
        return 0, stage_params
    
    param_files = glob.glob(os.path.join(splatam_dir, 'params*.npz'))
    steps = []
    for f in param_files:
        match = re.search(r'params(\d+)\.npz', f)
        if match:
            steps.append(int(match.group(1)))
    
    if steps:
        latest_step = max(steps)
        latest_path = os.path.join(splatam_dir, f'params{latest_step}.npz')
        return latest_step, latest_path
    
    return None


def main():
    args = argument_parsing()
    
    print("="*80)
    print("Failure Cases Analysis")
    print("="*80)
    
    print("\n[1/5] Loading configuration...")
    main_cfg = load_cfg(args)
    
    if args.result_dir:
        main_cfg.dirs.result_dir = args.result_dir
    
    checkpoint_info = find_latest_checkpoint(main_cfg.dirs.result_dir, args.stage)
    if checkpoint_info is None:
        print(f"Error: No checkpoints found in {main_cfg.dirs.result_dir}/splatam/")
        return
    
    latest_step, checkpoint_path = checkpoint_info
    
    if args.step is None:
        args.step = latest_step
        print(f"Auto-detected latest checkpoint: step={args.step}")
    else:
        if latest_step == 0:
            print(f"Using stage checkpoint: {args.stage}")
            args.step = 0
        else:
            requested_path = os.path.join(main_cfg.dirs.result_dir, 'splatam', f'params{args.step}.npz')
            if not os.path.exists(requested_path):
                print(f"Warning: Requested step {args.step} not found. Using latest: {latest_step}")
                args.step = latest_step
    
    print(f"Using checkpoint: step={args.step}, stage={args.stage}")
    
    print("[2/5] Initializing SLAM...")
    info_printer = InfoPrinter("FailureAnalysis")
    slam = init_SLAM_model(main_cfg, info_printer, None)
    slam.load_params_by_step(step=args.step, stage=args.stage)
    
    print("[3/5] Loading dataset...")
    dataset = slam.dataset_eval
    num_frames = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    
    class_info = None
    class_names = None
    if hasattr(main_cfg.slam, 'class_info_file') and os.path.exists(main_cfg.slam.class_info_file):
        class_info = load_class_names(main_cfg.slam.class_info_file)
        if class_info:
            class_names = [class_info[str(i)]['name'] if str(i) in class_info else f"Class {i}" 
                          for i in range(slam.n_cls)]
    
    failure_dir = os.path.join(slam.eval_dir + f"_{args.stage}", "failure_cases")
    os.makedirs(failure_dir, exist_ok=True)
    
    print(f"[4/5] Analyzing {num_frames} frames...")
    print(f"    Top-k: {args.topk_threshold}, Gap: {args.min_top1_gap*100}%, Pixel: {args.pixel_threshold*100}%")
    
    colormap = label_colormap(slam.n_cls)
    params = slam.params
    variables = slam.variables
    
    from src.slam.semsplatam.modified_ver.splatam.eval_helper import (
        setup_camera, transform_to_frame, transformed_params2rendervar,
        transformed_params2semrendervar, Renderer, SEMRenderer
    )
    
    failure_cases = []
    with torch.no_grad():
        for time_idx in tqdm(range(num_frames)):
            color, _, intrinsics, pose = dataset[time_idx]
            gt_w2c = torch.linalg.inv(pose)
            intrinsics = intrinsics[:3, :3]
            
            seman_gt = dataset.get_semantic_map(time_idx)
            seg_img = color.clone().to(slam.semantic_device)
            seman_pseudo, seman_pseudo_logits = slam.semantic_annotation(seg_img)
            seman_pseudo = seman_pseudo.to(slam.device)
            seman_pseudo_logits = seman_pseudo_logits.to(slam.device)
            n_cls = seman_pseudo_logits.shape[-1]
            
            if time_idx == 0:
                first_frame_w2c = torch.linalg.inv(pose)
                cam = setup_camera(color.shape[1], color.shape[0], intrinsics.cpu().numpy(),
                                 first_frame_w2c.detach().cpu().numpy(), num_channels=n_cls)
            
            transformed_gaussians = transform_to_frame(params, time_idx, gaussians_grad=False,
                                                      camera_grad=False, rel_w2c=gt_w2c)
            
            curr_data = {'cam': cam, 'im': color, 'seman_gt': seman_gt[0],
                        'seman_pseudo': seman_pseudo, 'seman_pseudo_logits': seman_pseudo_logits,
                        'id': time_idx, 'intrinsics': intrinsics,
                        'w2c': first_frame_w2c}
            
            rendervar = transformed_params2rendervar(params, transformed_gaussians)
            _, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
            seen = radius > 0
            
            seman_rendervar = transformed_params2semrendervar(params, variables, transformed_gaussians, seen)
            rastered_seman, _ = SEMRenderer(raster_settings=curr_data['cam'])(**seman_rendervar)
            
            rastered_seman = torch.nan_to_num(rastered_seman, nan=0.0)
            rastered_seman[rastered_seman < 0] = 0.0
            rastered_seman = rastered_seman.permute(1, 2, 0)
            rastered_cls_ids = rastered_seman.argmax(-1)
            
            topk_sizes = [1, 3, 5, args.topk_threshold]
            analysis = analyze_topk_predictions(rastered_seman, curr_data['seman_gt'].long(), topk_sizes)
            
            if analysis is None:
                continue
            
            top1_acc = analysis['top1']['accuracy']
            topk_acc = analysis[f'top{args.topk_threshold}']['accuracy']
            acc_gap = topk_acc - top1_acc
            failure_ratio = analysis['top1']['failure_ratio']
            
            is_failure = (acc_gap >= args.min_top1_gap and failure_ratio >= args.pixel_threshold)
            
            if is_failure:
                print(f"\n  Frame {time_idx}: Top-1={top1_acc*100:.1f}%, Top-{args.topk_threshold}={topk_acc*100:.1f}%, Gap={acc_gap*100:.1f}%")
                
                failure_cases.append({
                    'frame_idx': time_idx,
                    'top1_acc': top1_acc,
                    'topk_acc': topk_acc,
                    'acc_gap': acc_gap,
                    'failure_ratio': failure_ratio
                })
                
                color_np = color.cpu().numpy()
                if color_np.ndim == 3 and color_np.shape[0] == 3:
                    image = color_np.transpose(1, 2, 0)
                elif color_np.ndim == 3 and color_np.shape[2] == 3:
                    image = color_np
                else:
                    image = color_np
                
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                
                fig = create_failure_visualization(
                    image=image,
                    gt_semantic=curr_data['seman_gt'].cpu().numpy(),
                    pred_semantic=rastered_cls_ids.cpu().numpy(),
                    pred_logits=rastered_seman,
                    failure_analysis=analysis,
                    colormap=colormap,
                    class_names=class_names,
                    frame_idx=time_idx
                )
                
                save_path = os.path.join(failure_dir, f"failure_frame_{time_idx:04d}_gap_{acc_gap*100:.1f}.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    print(f"\n[5/5] Saving summary...")
    summary = {
        'config': {
            'topk_threshold': args.topk_threshold,
            'min_top1_gap': args.min_top1_gap,
            'pixel_threshold': args.pixel_threshold,
        },
        'total_frames': num_frames,
        'failure_cases_count': len(failure_cases),
        'failure_cases': failure_cases
    }
    
    summary_path = os.path.join(failure_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Total frames: {num_frames}")
    print(f"Failure cases: {len(failure_cases)}")
    if len(failure_cases) > 0:
        avg_gap = np.mean([fc['acc_gap'] for fc in failure_cases]) * 100
        print(f"Avg gap: {avg_gap:.1f}%")
    print(f"Results: {failure_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
