import math
import random
import torch
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

def load_data(
    *,
    ann_dir, # point_den_3*3
    ddd_dir, # point_den_1*1
    batch_size,

    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    
    dataset = ImageDataset(
        ann_dir,
        ddd_dir,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def count_to_class(inpt):
    sppl = np.array([73.0, 148.0, 194.0, 276.0])
    return np.sum(inpt > sppl)

class ImageDataset(Dataset):
    def __init__(
        self,

        anndir,
        ddddir
    ):
        super().__init__()

        self.anndir = anndir # point_den_3*3
        self.ddddir = ddddir # point_den_1*1
        self.im_ls = os.listdir(ddddir)

    def __len__(self):
        return len(self.im_ls)

    def __getitem__(self, idx):

        ann_path = os.path.join( self.anndir, self.im_ls[idx].split('.npy')[0] + '.npy' )
        ddd_path = os.path.join( self.ddddir, self.im_ls[idx].split('.npy')[0] + '.npy' )

        # pil_im = np.array( Image.open( img_path ).convert('RGB') )

        pil_ann = np.load(ann_path)
        pil_ddd = np.load(ddd_path)
        cr_ann, count_0, count_1, count_2, count_all = random_crop_only_input( pil_ann, 256, 256)

        if random.random() < 0.5:
            cr_ann = cr_ann[:,::-1]

        if random.random() < 0.5:
            cr_ann = cr_ann[::-1,:]

        cr_ann = cr_ann.astype( np.float32 )
        cr_ann = torch.tensor(cr_ann).permute(2,0,1)
        count_list = [count_0, count_1, count_2]
        #print(count_list)

        out_dict = {}
        out_dict["y"] = torch.tensor(count_list, dtype=torch.float32)

        return cr_ann, out_dict

    

def count_dots(dot_map):
    rows, cols = dot_map.shape
    count = 0
    
    for i in range(rows - 2):
        for j in range(cols - 2):
            if np.all(dot_map[i:i+3, j:j+3] == 1):
                count += 1
    
    return count

def random_crop_only_input(annd, crop_width, crop_height):

    if crop_width > annd.shape[1] or crop_height > annd.shape[0]:
        raise ValueError("Crop dimensions are larger than the image dimensions." + str(annd.shape[1] ) + str(annd.shape[0]))

    x = random.randint(0, annd.shape[1] - crop_width)
    y = random.randint(0, annd.shape[0] - crop_height)

    cropped_ann = annd[y:y+crop_height, x:x+crop_width]

    count_0 = count_dots(cropped_ann[:, :, 0])
    count_1 = count_dots(cropped_ann[:, :, 1])
    count_2 = count_dots(cropped_ann[:, :, 2])

    count_all = count_0 + count_1 + count_2

    return cropped_ann, count_0, count_1, count_2, count_all


