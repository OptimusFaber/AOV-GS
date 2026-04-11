
import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_scheduler
import glob


from src.naruto.cfg_loader import argument_parsing, load_cfg
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step

## oneformer
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import json
import os
import wandb

import torch
import numpy as np
from PIL import Image
import random

def modify_metadata(class_info_file, processor):
    new_metadata = {}
    class_names = []
    thing_ids = []
    num_labels = 0

    # Ensure the file exists
    if not os.path.exists(class_info_file):
        raise FileNotFoundError(f"Error: '{class_info_file}' not found.")

    # Load class info JSON
    with open(class_info_file, "r") as f:
        class_info = json.load(f)

    # Process metadata
    for k, v in class_info.items():
        num_labels += 1
        new_metadata[k] = v["name"]
        class_names.append(v["name"])  # Ensure class names are stored
        if v.get("isthing", False):  # Use .get() to avoid KeyError
            thing_ids.append(int(k))

    # new_metadata[str(num_labels)] = num_labels
    new_metadata["num_labels"] = num_labels
    new_metadata['class_names'] = class_names
    new_metadata['thing_ids'] = thing_ids

    # Store in processor metadata
    if not hasattr(processor, "image_processor"):
        raise AttributeError("Error: 'processor' object has no attribute 'image_processor'")

    processor.image_processor.metadata = new_metadata
    processor.image_processor.num_labels = num_labels
    print("Metadata modified successfully.")


class CustomDataset(Dataset):
    def __init__(self, processor, img_save_dir):
        self.processor = processor
        self.img_save_dir = img_save_dir

    def __getitem__(self, idx):
        i = idx*10
        color_file = (f"{self.img_save_dir}/color_{i:04d}.jpg")
        image = Image.open(color_file)  # PIL image
        seman_file = f"{self.img_save_dir}/semantic_map_{i:04d}.npy"
        semantic_map = np.load(seman_file)
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        inputs = processor(images=image, segmentation_maps=semantic_map, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return idx, inputs

    def __len__(self):
        return 200


class CustomDatasetV2(Dataset):
    def __init__(self, processor, root_dirs):
        self.processor = processor
        self.color_paths = []
        self.seman_paths = []
        for root_dir in root_dirs:
            for idx in range(200):
                i = idx * 10
                self.color_paths.append(f"{root_dir}/rgb/color_{i:04d}.jpg")
                self.seman_paths.append(f"{root_dir}/semantic/semantic_map_{i:04d}.npy")
    def __getitem__(self, idx):
        image = Image.open(self.color_paths[idx])  # PIL image
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        inputs = processor(images=image, segmentation_maps=semantic_map, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return idx, inputs

    def get_semantic_gt(self,idx):
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map < 0] = 255
        semantic_map[semantic_map > 101] = 255
        return semantic_map
    def __len__(self):
        return len(self.color_paths)

    def get_scene_name(self,idx):
        color_path = self.color_paths[idx]
        parts = color_path.strip().split('/')
        for p in parts:
            if ('room' in p) or ('office' in p):
                current_scene = p
        return current_scene

    def get_frame_id(self,idx):
        color_path = self.color_paths[idx]
        parts = color_path.split('_')
        frame_id = int(parts[-1][:4])
        return frame_id

class CustomDatasetV3(Dataset):
    def __init__(self, processor, root_dirs):
        self.processor = processor
        self.color_paths = []
        self.seman_paths = []
        self.color_paths,self.seman_paths = self.get_filepaths(root_dirs)
        self.checkfiles()

    def get_filepaths(self,root_dirs):
        color_paths = []
        seman_paths = []
        for root_dir in root_dirs:
            color_files = os.path.join(root_dir, 'rgb/color_*.jpg')
            # seman_files = os.path.join(root_dir, 'semantic/semantic_map_*.npy')
            current_color_paths = glob.glob(color_files)
            current_semantic_paths = []
            for j in range(len(current_color_paths)):
                parts = current_color_paths[j].split('_')
                frame_id = int(parts[-1][:4])
                current_semantic_paths.append(os.path.join(root_dir, f'semantic/semantic_map_{frame_id:04d}.npy'))

            ## random sampling 500 frames
            if len(current_color_paths)>500:
                sample_indices = np.random.choice(len(current_color_paths), 500, replace=False)
                sampled_color_paths = [current_color_paths[i] for i in sample_indices]
                sampled_semantic_paths = [current_semantic_paths[i] for i in sample_indices]
                color_paths += sampled_color_paths
                seman_paths += sampled_semantic_paths
            else:
                color_paths += current_color_paths
                seman_paths += current_semantic_paths


        return color_paths, seman_paths


    def __getitem__(self, idx):
        image = Image.open(self.color_paths[idx])  # PIL image
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map <= 0] = 255
        semantic_map[semantic_map > 40] = 255
        inputs = processor(images=image, segmentation_maps=semantic_map, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
        return idx, inputs

    def get_semantic_gt(self,idx):
        semantic_map = np.load(self.seman_paths[idx])
        semantic_map[semantic_map <= 0] = 255
        semantic_map[semantic_map > 40] = 255
        return semantic_map

    def get_scene_name(self,idx):
        color_path = self.color_paths[idx]
        parts = color_path.strip().split('/')
        current_scene = parts[-4]
        return current_scene

    def get_frame_id(self,idx):
        color_path = self.color_paths[idx]
        parts = color_path.split('_')
        frame_id = int(parts[-1][:4])
        return frame_id

    def checkfiles(self):
        delete_indices = []
        for i in range(len(self.seman_paths)):
            semantic_map = np.load(self.seman_paths[i])
            semantic_map[semantic_map <= 0] = 255
            semantic_map[semantic_map > 40] = 255
            valid = (semantic_map<=40).sum()/semantic_map.size
            if valid < 0.5:
                delete_indices.append(i)
        if len(delete_indices)>0:
            color_paths = [item for idx, item in enumerate(self.color_paths) if idx not in delete_indices]
            seman_paths = [item for idx, item in enumerate(self.seman_paths) if idx not in delete_indices]
            self.color_paths = color_paths
            self.seman_paths = seman_paths
            print(f'Delete {len(delete_indices)} invalid samples!')


    def __len__(self):
        return len(self.color_paths)

def map_object_id_to_semlabel(object_ids,id2label):
    '''

    :param object_ids: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    :return: semantic labels: torch.tensor (H,W) # output from habitat-sim, including class id of each pixel
    '''
    id2label = torch.tensor(id2label)
    sem_labels = id2label[object_ids.long()]
    return sem_labels


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


def semantic_mask_to_rgb(mask: torch.Tensor, save_path: str):
    """
    Convert a semantic mask tensor (H, W) to an RGB image and save it.

    Args:
        mask (torch.Tensor): A tensor of shape (H, W) with semantic class indices.
        save_path (str): Path to save the RGB image.
    """
    # Get the number of unique classes
    unique_classes = torch.unique(mask).tolist()
    #num_classes = max(unique_classes) + 1  # Assuming classes start from 0

    # Generate a random colormap
    num_classes = 41
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


def write_dict_to_file(save_dict,save_file):
    with open(save_file) as f:
        for k,v in save_dict:
            f.write(f'{k}： {v.item():.4f}\n')
            f.close()


def compute_miou(pred_mask, true_mask):
    """
    Compute mean Intersection over Union (mIoU) using PyTorch tensors.

    Parameters:
        pred_mask (torch.Tensor): Predicted mask of shape (H, W), values are class indices.
        true_mask (torch.Tensor): Ground truth mask of shape (H, W), values are class indices.

    Returns:
        float: Mean IoU score.
    """
    classes = torch.unique(torch.cat((pred_mask, true_mask)))
    classes = classes[classes != 0]
    iou_per_class = []

    for cls in classes:
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()

        if union == 0:
            iou = torch.tensor(float('nan'))  # Class not present in prediction and ground truth
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    iou_per_class = torch.stack(iou_per_class)
    miou = torch.nanmean(iou_per_class).item()
    return miou


if __name__ == "__main__":

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="finetune-oneformer-replica",
        # Set the wandb project where this run will be logged.
        project="finetune_oneformer_mp3d",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 5e-5,
            "architecture": "oneformer",
            "dataset": "MP3D",
            "epochs": 5000,
        },
    )

    info_printer = InfoPrinter("Finetune Oneformer")
    timer = Timer()
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)

    info_printer("Loading oneformer Ade20K pretrained checkpoint...", 0, "Initialization")
    processor = AutoProcessor.from_pretrained(main_cfg.oneformer["checkpoint"])
    model = AutoModelForUniversalSegmentation.from_pretrained(main_cfg.oneformer["checkpoint"], is_training=True)
    version = 'finetune_mp3d'

    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
    class_info_file = './configs/MP3D/class_info_file.json'
    modify_metadata(class_info_file=class_info_file,processor=processor)

    info_printer("Prepare optimizer...", 0, "Initialization")

    finetune_scenes = ["GdvgFV5R1Z5","gZ6f7yhEvPG","HxpKQynjfin","pLe4wQe7qrG","YmJkqBEsHnH"]
    test_scenes = ["GdvgFV5R1Z5","gZ6f7yhEvPG","HxpKQynjfin","pLe4wQe7qrG","YmJkqBEsHnH"]

    num_classes = 41

    train_root_dirs = []
    for selected_scene in finetune_scenes:
        img_save_dir = f'./data/mp3d_sim_finetune/{selected_scene}/results_habitat/'
        train_root_dirs.append(img_save_dir)

    test_root_dirs = []
    for selected_scene in test_scenes:
        img_save_dir = f'./data/mp3d_sim_nvs/{selected_scene}/results_habitat/'
        test_root_dirs.append(img_save_dir)

    Trainset = CustomDatasetV3(processor=processor, root_dirs=train_root_dirs)
    Testset = CustomDatasetV3(processor=processor, root_dirs=test_root_dirs)

    train_dataloader = DataLoader(Trainset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(Testset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(Trainset, batch_size=1, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5,betas=(0.9, 0.999), weight_decay=0.01)

    per_scene_iter = 5
    total_step = 0
    ACCUM_STEPS = 10
    num_training_steps = per_scene_iter * len(Trainset)
    print(f"Total number of training images: {len(Trainset)}")

    # num_training_steps = per_scene_iter*200*len(finetune_scenes)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    info_printer("Preparing Dataset...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    total_step = 0
    info_printer.update_total_step(num_training_steps)
    # optimizer.zero_grad()

    for num_iter in range(per_scene_iter):

        ##################################################
        ### Training
        ##################################################
        model.train()
        for idx, batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            # forward pass
            outputs = model(**batch)
            # backward pass + optimize
            loss = outputs.loss

            current_scene = Trainset.get_scene_name(idx)
            info_printer.update_scene(main_cfg.general.dataset + " - " + current_scene)
            info_printer("Training...",total_step, f"Loss: {loss.item():.4f}")
            loss.backward()

            optimizer.step()
            # optimizer.zero_grad()
            lr_scheduler.step()

            run.log({"total_step": total_step, "loss": loss.item()})
            total_step += 1

        save_ckpt_dir = f'./data/checkpoint/oneformer/{version}/step_{total_step}'
        os.makedirs(save_ckpt_dir, exist_ok=True)
        model.save_pretrained(save_ckpt_dir)

        #####################################
        ####  evaluation
        #####################################

        model.eval()
        with torch.no_grad():
            scenes_acc = {}
            for selected_scene in test_scenes:
                scenes_acc[selected_scene] = {
                    'acc': 0,
                    'num_frames': 0,
                    'miou': 0,
                }

            for idx, batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                gt_semantic_map = Testset.get_semantic_gt(idx)
                gt_semantic_map[gt_semantic_map > (num_classes-1)] = 0
                gt_semantic_map = torch.from_numpy(gt_semantic_map).to(device)

                img_size = gt_semantic_map.shape
                semantic_segmentation = \
                processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size])[0]
                semantic_segmentation[semantic_segmentation > (num_classes-1)] = 0

                valid_pixels = img_size[0] * img_size[1] - (gt_semantic_map == 0).sum()
                corrects = (semantic_segmentation == gt_semantic_map) & (gt_semantic_map > 0)
                acc = corrects.sum() / valid_pixels

                miou = compute_miou(semantic_segmentation, gt_semantic_map)

                current_scene = Testset.get_scene_name(idx)
                scenes_acc[current_scene]['acc'] += acc.item()
                scenes_acc[current_scene]['num_frames'] += 1
                scenes_acc[current_scene]['miou'] += miou

                info_printer.update_scene(main_cfg.general.dataset + " - " + current_scene)
                info_printer("Evaluating...", total_step, f"Acc: {acc.item():.4f}, miou: {miou:.4f}")

            run.log({"total_step": total_step, "Acc": acc.item(), "miou": miou})

            for selected_scene in test_scenes:
                avg_acc = scenes_acc[selected_scene]['acc'] / scenes_acc[selected_scene]['num_frames']
                avg_miou = scenes_acc[selected_scene]['miou'] / scenes_acc[selected_scene]['num_frames']
                info_printer.update_scene(main_cfg.general.dataset + " - " + selected_scene)
                info_printer("Evaluating...", total_step, f"Scene Acc: {avg_acc:.4f}, Miou: {avg_miou:.4f}")

    #################################################
    # do per class semantic evaluation for all scenes
    #################################################

    class_info_file = './configs/MP3D/class_info_file.json'
    with open(class_info_file, "r") as f:
        class_info = json.load(f)
    with torch.no_grad():
        scenes_acc = {}
        for selected_scene in test_scenes:
            scenes_acc[selected_scene] = {
                'acc': 0,
                'num_frames': 0,
                'miou': 0,
            }
            per_class = {}

        for scene in scenes_acc.keys():
            per_cls_acc = {}
            for k,v in class_info.items():
                per_cls_acc[k] = {
                    'name': v['name'],
                    'acc' : 0,
                    'num_frames': 0,
                }
            scenes_acc[scene]['per_cls'] = per_cls_acc

        for idx, batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            gt_semantic_map = Testset.get_semantic_gt(idx)
            gt_semantic_map[gt_semantic_map > (num_classes-1)] = 0
            gt_semantic_map = torch.from_numpy(gt_semantic_map).to(device)

            img_size = gt_semantic_map.shape
            semantic_segmentation = \
                processor.post_process_semantic_segmentation(outputs, target_sizes=[img_size])[0]
            semantic_segmentation[semantic_segmentation > (num_classes-1)] = 0

            valid_pixels = img_size[0] * img_size[1] - (gt_semantic_map == 0).sum()
            corrects = (semantic_segmentation == gt_semantic_map) & (gt_semantic_map > 0)
            acc = corrects.sum() / valid_pixels

            miou = compute_miou(semantic_segmentation, gt_semantic_map)

            current_scene = Testset.get_scene_name(idx)
            frame_id = Testset.get_frame_id(idx)
            scenes_acc[current_scene]['acc'] += acc.item()
            scenes_acc[current_scene]['num_frames'] += 1
            scenes_acc[current_scene]['miou'] += miou

            save_dir = f'./data/checkpoint/oneformer/{version}/eval/{current_scene}/'
            os.makedirs(save_dir,exist_ok=True)
            semantic_mask_to_rgb(semantic_segmentation, f"{save_dir}/semantic_rgb_{frame_id:04d}.png")

            #### do per class statistic
            unique_classes = torch.unique(gt_semantic_map).tolist()
            for cls_id in unique_classes:
                if cls_id >0:
                    gt_cls_pixels = (gt_semantic_map==cls_id).sum()
                    corrects = (semantic_segmentation==gt_semantic_map) & (gt_semantic_map==cls_id)
                    cls_acc = corrects.sum()/gt_cls_pixels
                    scenes_acc[current_scene]['per_cls'][str(cls_id)]['acc'] += cls_acc.item()
                    scenes_acc[current_scene]['per_cls'][str(cls_id)]['num_frames'] += 1

        for scene in scenes_acc.keys():
            save_dict = {}
            avg_acc = scenes_acc[scene]['acc'] / scenes_acc[scene]['num_frames']
            avg_miou = scenes_acc[scene]['miou'] / scenes_acc[scene]['num_frames']
            save_dict['Avg_acc'] = avg_acc
            save_dict['num_frames'] = scenes_acc[scene]['num_frames']
            save_dict['Avg_miou'] = avg_miou
            for k,v in scenes_acc[scene]['per_cls'].items():
                if v['num_frames'] > 0:
                    save_dict[v['name']] = v['acc']/ v['num_frames']
                else:
                    save_dict[v['name']] = -1

            write_to_file = f'./data/checkpoint/oneformer/{version}/eval/{scene}/eval.txt'
            with open(write_to_file,'a') as f:
                for key in save_dict.keys():
                    f.write(f'{key}: {save_dict[key]:.4f}\n')
                f.close()

    info_printer("Evaluation Done!", total_step, f"Save all evaluation results!")

    run.finish()







