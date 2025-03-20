import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import stopit
import json
import matplotlib.pyplot as plt
import shutil

from datetime import date, datetime
from scipy import ndimage as ndi
from collections import defaultdict

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def set_nonzero_to_one(array):
    return np.where(array != 0, 1, 0)

def save_cell_counts(save_path, generated_counts, ground_truth_counts):
    data = defaultdict(dict)
    for filename, counts in generated_counts.items():
        data[filename]["generated"] = counts
    for filename, counts in ground_truth_counts.items():
        data[filename]["ground_truth"] = counts
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        sample_dir="",
        model_path="path/to/checkpoints/brca_m2c.pt"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

args = create_argparser().parse_args([])
args.num_channels = 256
args.num_res_blocks=2
args.num_head_channels=64
args.attention_resolutions='32,16,8'
args.class_cond=True
args.use_scale_shift_norm=True
args.resblock_updown=True
args.use_fp16=False
args.learn_sigma=True
args.diffusion_steps=1000
args.noise_schedule='cosine'
args.p2_gamma = 1
args.p2_k = 1
args.image_size = 256
args.timestep_respacing = 'ddim100'
args.use_ddim = True

dist_util.setup_dist()
logger.configure()

logger.log("creating model and diffusion...")
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.model_path, map_location="cpu")
)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model.to(device)
if args.use_fp16:
    model.convert_to_fp16()
model.eval()

sample_fn = (
    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
)

def count_dots(cell_map):

    labeled_array, num_cells = ndimage.label(cell_map)

    return num_cells

def count_cells_3c(cell_map):
    cell_0_count = count_dots(cell_map[:,:,0])
    cell_1_count = count_dots(cell_map[:,:,1])
    cell_2_count = count_dots(cell_map[:,:,2])
    return cell_0_count, cell_1_count, cell_2_count

def visualize_cell_dot_map(dot_map, file_name):
    assert dot_map.shape[2] == 3, "Input must have 3 channels"
    colors = [
        [1, 0, 0],      # Red for cell type 0
        [0, 1, 0],      # Green for cell type 1
        [0, 0, 1]       # Blue for cell type 2
    ]

    height, width, _ = dot_map.shape
    combined_image = np.zeros((height, width, 3))

    for i, color in enumerate(colors):
        channel = dot_map[:,:,i]
        for c in range(3):  # RGB channels
            combined_image[:,:,c] += channel * color[c]

    combined_image = np.clip(combined_image, 0, 1)
    plt.figure()
    plt.imshow(combined_image)
    plt.imsave(file_name, combined_image)
    plt.close()

def denoise_fun(inp):
    all_labels = np.zeros([256,256,3])
    unique_list = []
    for ik in range(3):
        image = (inp[0][ik] > 0.4).numpy().astype(int)
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        u_val = np.unique(labels)
        for ikii in u_val:
            if np.sum(labels == ikii) <= 5:
                labels[labels == ikii] = 0
        all_labels[:,:,ik] = labels
        unique_list.append(len(np.unique(labels)))

    return all_labels, np.sum(unique_list), unique_list

def get_cell_count_from_file(file_name):
    return int(file_name.split(".npy")[0].split("_")[-1])

results_root_path = "path/to/results/"
day = date.today()
current_time = datetime.now().strftime("%H-%M-%S")
results_save_path = os.path.join(results_root_path, str(day), current_time)
os.makedirs(results_save_path, exist_ok=True)

args_dict = vars(args)
with open(os.path.join(results_save_path, "hyperparams.json"), "w") as f:
    json.dump(args_dict, f, indent=4)

test_patch_path = "/path/to/test_dataset/"
test_patch_list = os.listdir(test_patch_path)

generated_cell_counts = {}
ground_truth_cell_counts = {}

for test_patch_item in test_patch_list:
    if test_patch_item.endswith(".npy"):
        test_patch_item_path = os.path.join(test_patch_path, test_patch_item)
        test_patch = np.load(test_patch_item_path)
        cell_count_0, cell_count_1, cell_count_2 = count_cells_3c(test_patch)
        cell_count_list = [cell_count_0, cell_count_1, cell_count_2]
        
        ground_truth_cell_counts[test_patch_item] = cell_count_list
        
        print(f"Generating layout with {cell_count_0} type-0, {cell_count_1} type-1, {cell_count_2} type-2 cells.")
        
        model_kwargs = {}
        model_kwargs["y"] = th.tensor([cell_count_list], device=dist_util.dev(), dtype=th.float32)

        sample = sample_fn(
            model,
            (1, 3, 256, 256),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        with stopit.ThreadingTimeout(5) as context_manager:
            labels_res, num_cell, cell_num_list = denoise_fun(sample.detach().cpu())

        if context_manager.state == context_manager.EXECUTED:
            save_npy_root_path = os.path.join(results_save_path, 'npy')
            save_img_root_path = os.path.join(results_save_path, 'img')
            os.makedirs(save_npy_root_path, exist_ok=True)
            os.makedirs(save_img_root_path, exist_ok=True)
            
            generated_cell_counts[test_patch_item] = cell_num_list
            
            test_patch_item_name = test_patch_item.split(".npy")[0]
            labels_res = set_nonzero_to_one(labels_res)
            
            # Save npy file
            np.save(os.path.join(save_npy_root_path, f"{test_patch_item_name}_gen_{num_cell}.npy"), labels_res)
            
            # Visualize and save image
            img_path = os.path.join(save_img_root_path, f"{test_patch_item_name}_gen_{num_cell}.png")
            visualize_cell_dot_map(labels_res, img_path)

        elif context_manager.state == context_manager.TIMED_OUT:
            print("DID NOT FINISH...")

# Save the cell counts
cell_counts_path = os.path.join(results_save_path, 'cell_counts.json')
save_cell_counts(cell_counts_path, generated_cell_counts, ground_truth_cell_counts)

print("Generation Done.")