import numpy as np
from pylab import *
import torch
from ripser import ripser
import cripser as cr
import ot
import FastGeodis
import cv2
from loss_functions.topoloss_pd import getTopoLoss

def binarize_map(map, threshold=0.7):
    return (map >= threshold).float()

def combine_channels(tensor):
    # Combine channels by taking the maximum value across all channels
    return torch.max(tensor, dim=0, keepdim=True)[0]

def calculate_edt(input):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure input is 2D
    if input.dim() == 3:
        input = input.squeeze(0)
    
    image = (input).to(torch.float32)
    image_pt = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    image_pt = image_pt.to(device)
    
    mask_pt = 1 - image_pt
    v = 1e10
    iterations = 2
    lamb = 0.0  # <-- Euclidean distance transform
    
    euclidean_dist = FastGeodis.generalised_geodesic2d(
        mask_pt, image_pt, v, lamb, iterations
    )
    #print(euclidean_dist.shape)
    #return euclidean_dist.squeeze(0)  # Remove batch dimension, keep channel dimension
    return euclidean_dist


def normalize_edt_map(edt_map, norm_type='global-norm'):
    assert edt_map.shape[0] == 1 and edt_map.shape[1] == 1, "Input should be of shape (1, 1, H, W)"
    
    if norm_type == 'global-norm':
        min_val = edt_map.min()
        max_val = edt_map.max()
        return (edt_map - min_val) / (max_val - min_val + 1e-6)
    
    elif norm_type == 'standard':
        mean = edt_map.mean()
        std = edt_map.std()
        return (edt_map - mean) / (std + 1e-6)
    
    elif norm_type == 'relative-scaling':
        global_max = edt_map.max()
        return edt_map / (global_max + 1e-6)
    
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

class TopoLossMSE2D_TotalCell(torch.nn.Module):
    def __init__(self, topo_weight, topo_window):
        super().__init__()
        #print(f"Topo weight: {topo_weight}")
        self.topo_weight = topo_weight
        self.topo_window = topo_window
    
    def forward(self, pred, target):
        assert pred.shape == target.shape

        pred = (pred + 1.0) / 2.0 # transfer from [-1, 1] to [0, 1]
        
        # Ensure pred and target are float32
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        
        batch_size = pred.size(0)
        loss_per_item = []

        for idx in range(batch_size):
            # Binarize prediction
            binary_pred = binarize_map(pred[idx])
            #print(binary_pred.shape) # (3, H, W)

            conbined_pred = combine_channels(binary_pred)
            combined_target = combine_channels(target[idx])

            edt_pred = calculate_edt(1-conbined_pred[0])
            edt_target = calculate_edt(1-combined_target[0])
            
            # Normalize EDT maps
            norm_edt_pred = normalize_edt_map(edt_pred, norm_type='global-norm')
            norm_edt_target = normalize_edt_map(edt_target, norm_type='global-norm')

            
            # Calculate topological loss for each channel
            item_loss = getTopoLoss(norm_edt_pred[0][0], norm_edt_target[0][0], self.topo_window)
            loss_per_item.append(item_loss)
        
        # Calculate mean loss over the batch
        mean_loss = torch.stack(loss_per_item).mean()
        
        # Apply topo weight
        final_loss = mean_loss * self.topo_weight
        
        return final_loss, mean_loss