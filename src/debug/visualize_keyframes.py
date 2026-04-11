"""
Script for visualizing keyframes found by the robot during training.
Saves RGB images of keyframes and information about them.
"""

import os
import numpy as np
import torch
import cv2
import json
from pathlib import Path
import argparse
from tqdm import tqdm


def load_keyframes_from_checkpoint(checkpoint_dir):
    """
    Loads keyframes from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to results directory (e.g., results/Replica/office0/ActiveSem/run_0)
    
    Returns:
        keyframe_info: dict with keyframe information
    """
    keyframe_info = {
        'keyframe_time_indices': [],
        'keyframe_paths': [],
        'global_keyframe_indices': [],
    }
    
    # Search for saved keyframe indices
    checkpoint_files = list(Path(checkpoint_dir).glob("**/keyframe_time_indices*.npy"))
    if checkpoint_files:
        # Get latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        keyframe_indices = np.load(latest_checkpoint)
        keyframe_info['keyframe_time_indices'] = keyframe_indices.tolist()
        print(f"Loaded {len(keyframe_indices)} keyframes from {latest_checkpoint}")
    
    # Search for global keyframes (if saved separately)
    global_kf_files = list(Path(checkpoint_dir).glob("**/global_keyframe_time_indices*.npy"))
    if global_kf_files:
        latest_global = max(global_kf_files, key=lambda x: x.stat().st_mtime)
        global_indices = np.load(latest_global)
        keyframe_info['global_keyframe_indices'] = global_indices.tolist()
        print(f"Loaded {len(global_indices)} global keyframes from {latest_global}")
    
    return keyframe_info


def load_keyframes_from_dataset(dataset_dir, keyframe_indices):
    """
    Loads RGB images of keyframes from dataset.
    
    Args:
        dataset_dir: Path to data directory (data/Replica/office0)
        keyframe_indices: List of keyframe indices
    
    Returns:
        keyframes: dict with RGB images
    """
    keyframes = {}
    results_dir = os.path.join(dataset_dir, "results_habitat")
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found!")
        return keyframes
    
    # Find all RGB images
    rgb_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    
    print(f"Found {len(rgb_files)} RGB images in {results_dir}")
    print(f"Loading {len(keyframe_indices)} keyframes...")
    
    for idx in tqdm(keyframe_indices):
        if idx < len(rgb_files):
            rgb_path = os.path.join(results_dir, rgb_files[idx])
            if os.path.exists(rgb_path):
                img = cv2.imread(rgb_path)
                if img is not None:
                    keyframes[idx] = {
                        'rgb': img,
                        'path': rgb_path,
                        'filename': rgb_files[idx]
                    }
        else:
            print(f"Warning: index {idx} is out of bounds for available images ({len(rgb_files)})")
    
    return keyframes


def visualize_keyframes(keyframes, output_dir, keyframe_info, dataset_name="Replica", scene_name="office0"):
    """
    Visualizes keyframes and saves them to output directory.
    
    Args:
        keyframes: dict with RGB images of keyframes
        output_dir: Directory to save results
        keyframe_info: dict with keyframe information
        dataset_name: Dataset name
        scene_name: Scene name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    rgb_dir = os.path.join(output_dir, "keyframes_rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # Save RGB images of keyframes
    print(f"\nSaving {len(keyframes)} keyframes to {rgb_dir}...")
    for idx, kf_data in tqdm(keyframes.items()):
        output_path = os.path.join(rgb_dir, f"keyframe_{idx:04d}.jpg")
        cv2.imwrite(output_path, kf_data['rgb'])
    
    # Create HTML visualization
    html_path = os.path.join(output_dir, "keyframes_visualization.html")
    create_html_visualization(keyframes, html_path, keyframe_info, dataset_name, scene_name)
    
    # Save JSON with information
    json_path = os.path.join(output_dir, "keyframes_info.json")
    save_keyframes_info(keyframes, json_path, keyframe_info, dataset_name, scene_name)
    
    print(f"\nVisualization saved to {output_dir}")
    print(f"  - RGB images: {rgb_dir}")
    print(f"  - HTML visualization: {html_path}")
    print(f"  - JSON information: {json_path}")


def create_html_visualization(keyframes, output_path, keyframe_info, dataset_name, scene_name):
    """Creates HTML file for visualizing keyframes in browser."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Keyframes Visualization - {dataset_name}/{scene_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .info {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .keyframes-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .keyframe-card {{
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .keyframe-card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .keyframe-id {{
            font-weight: bold;
            color: #666;
            margin-top: 10px;
        }}
        .global-keyframe {{
            border: 3px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <h1>Keyframes Visualization</h1>
    <div class="info">
        <p><strong>Dataset:</strong> {dataset_name}</p>
        <p><strong>Scene:</strong> {scene_name}</p>
        <p><strong>Total Keyframes:</strong> {len(keyframes)}</p>
        <p><strong>Global Keyframes:</strong> {len(keyframe_info.get('global_keyframe_indices', []))}</p>
    </div>
    
    <div class="keyframes-grid">
"""
    
    global_kf_set = set(keyframe_info.get('global_keyframe_indices', []))
    
    for idx in sorted(keyframes.keys()):
        kf_data = keyframes[idx]
        is_global = idx in global_kf_set
        global_class = "global-keyframe" if is_global else ""
        global_label = " (Global)" if is_global else ""
        
        # Relative path to image
        img_rel_path = f"keyframes_rgb/keyframe_{idx:04d}.jpg"
        
        html_content += f"""
        <div class="keyframe-card {global_class}">
            <img src="{img_rel_path}" alt="Keyframe {idx}">
            <div class="keyframe-id">Frame {idx}{global_label}</div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def save_keyframes_info(keyframes, output_path, keyframe_info, dataset_name, scene_name):
    """Saves keyframe information to JSON."""
    info = {
        'dataset': dataset_name,
        'scene': scene_name,
        'total_keyframes': len(keyframes),
        'keyframe_indices': sorted(keyframes.keys()),
        'global_keyframe_indices': keyframe_info.get('global_keyframe_indices', []),
        'keyframe_details': {}
    }
    
    for idx, kf_data in keyframes.items():
        info['keyframe_details'][idx] = {
            'filename': kf_data['filename'],
            'path': kf_data['path'],
            'is_global': idx in set(keyframe_info.get('global_keyframe_indices', []))
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize keyframes from training results')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Path to results directory (e.g., results/Replica/office0/ActiveSem/run_0)')
    parser.add_argument('--dataset_dir', type=str, 
                        default=None,
                        help='Path to data directory (e.g., data/Replica/office0). If not specified, will be inferred from result_dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualization (default: result_dir/debug_keyframes)')
    
    args = parser.parse_args()
    
    # Determine dataset_dir if not specified
    if args.dataset_dir is None:
        # Try to infer from result_dir
        result_path = Path(args.result_dir)
        scene_name = result_path.parent.name
        dataset_name = result_path.parent.parent.name
        args.dataset_dir = f"data/{dataset_name}/{scene_name}"
        print(f"Auto-detected dataset_dir: {args.dataset_dir}")
    
    # Determine output_dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, "debug_keyframes")
    
    # Load keyframe information
    print("Loading keyframe information...")
    keyframe_info = load_keyframes_from_checkpoint(args.result_dir)
    
    if not keyframe_info['keyframe_time_indices']:
        print("Warning: keyframe indices not found in checkpoint. Trying to load from params.npz...")
        # Try to load from params.npz if available
        params_files = list(Path(args.result_dir).glob("**/params.npz"))
        if params_files:
            print("Found params.npz files, but keyframe indices should be saved separately.")
            print("Try running training with save_checkpoints=True to save keyframes.")
            return
        else:
            print("Failed to find keyframe information. Exiting.")
            return
    
    # Load RGB images of keyframes
    print(f"\nLoading RGB images from {args.dataset_dir}...")
    keyframes = load_keyframes_from_dataset(args.dataset_dir, keyframe_info['keyframe_time_indices'])
    
    if not keyframes:
        print("Failed to load keyframes. Check dataset_dir path.")
        return
    
    # Visualize
    scene_name = Path(args.dataset_dir).name
    dataset_name = Path(args.dataset_dir).parent.name
    visualize_keyframes(keyframes, args.output_dir, keyframe_info, dataset_name, scene_name)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
