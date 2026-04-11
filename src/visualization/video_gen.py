import numpy as np
import cv2
import mediapy as media
import os
from tqdm import tqdm


def save_training_video(gt_path,render_path, video_path, H, W):
    """Creates videos out of the images saved to disk."""
    gt_files = os.listdir(gt_path)
    gt_frames = [os.path.join(gt_path, f) for f in gt_files]
    gt_frames.sort()

    render_files = os.listdir(render_path)
    render_frames = [os.path.join(render_path, f) for f in render_files]
    render_frames.sort()

    n_frames = len(gt_frames)

    video_kwargs = {
        'shape': (int(H * 2), int(W)),
        'codec': 'h264',
        'fps': 10,
        'crf': 18,
    }

    video_file = os.path.join(video_path, f'training.mp4')
    with media.VideoWriter(video_file, **video_kwargs, input_format="rgb") as writer:
        for i in tqdm(range(n_frames), desc=f"Rendering video"):
            gt_frame = np.nan_to_num(cv2.imread(gt_frames[i])[:, :, ::-1])
            render_frame = np.nan_to_num(cv2.imread(render_frames[i])[:, :, ::-1])
            frame = np.vstack((gt_frame,render_frame))
            writer.add_image(frame)


def save_output_video(vis_path, video_path, H, W, output, near_depth, far_depth):
    """Creates videos out of the images saved to disk."""

    video_kwargs = {
        'shape': (int(H * 0.5), int((W * 3) * 0.5)),
        'codec': 'h264',
        'fps': 30,
        'crf': 18,
    }

    video_file = os.path.join(video_path, f'rendering.mp4')
    with media.VideoWriter(
            video_file, **video_kwargs, input_format="rgb") as writer:
        for output_view in tqdm(output, desc=f"Rendering video"):
            idx = output_view["idx"]
            image = output_view["image"]
            normal = output_view["normal"]
            depth = (np.tile(output_view["depth"].reshape(H, W, 1), (1, 1, 3)) - near_depth) / (far_depth - near_depth)
            image_frame = (np.clip(np.nan_to_num(image), 0., 1.) * 255.).astype(np.uint8)
            normal_frame = (np.clip(np.nan_to_num(normal), 0., 1.) * 255.).astype(np.uint8)
            depth_frame = (np.clip(np.nan_to_num(depth), 0., 1.) * 255.).astype(np.uint8)
            frame = np.concatenate((depth_frame, image_frame, normal_frame), axis=1)
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)),
                               interpolation=cv2.INTER_LINEAR)
            writer.add_image(frame)

if __name__ == '__main__':
    # set params ##
    HOME = "/mnt/Data2/"
    PROJ_DIR = f"{HOME}/liyan/ActiveGAMER"
    DATASET = "Replica"
    RESULT_DIR = f'{PROJ_DIR}/results'
    GT_DATA_DIR = f"{HOME}/slam_datasets/{DATASET}"

    scene = "office4"
    seed = 0
    method = "ActiveLang"
    slam = "splatam"
    vis_dir = f'{RESULT_DIR}/{DATASET}/{scene}/{method}/run_0/visualization/'
    gt_path = f'{vis_dir}/rgbd/'
    render_path = f'{vis_dir}/rendered_rgbd/'
    video_path = f'{vis_dir}/planning_path/'

    save_training_video(gt_path, render_path, video_path, H=320, W=640)
