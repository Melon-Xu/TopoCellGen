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

def normalize_edt_map(edt_map, norm_type='channel-wise-norm'):
    assert edt_map.shape[0] == 1 and edt_map.shape[1] == 3, "Input should be of shape (1, 3, H, W)"
    
    if norm_type == 'channel-wise-norm':
        min_vals = edt_map.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        max_vals = edt_map.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        return (edt_map - min_vals) / (max_vals - min_vals + 1e-6)
    
    elif norm_type == 'global-norm':
        min_val = edt_map.min()
        max_val = edt_map.max()
        return (edt_map - min_val) / (max_val - min_val + 1e-6)
    
    elif norm_type == 'channel-wise-stand':
        mean = edt_map.mean(dim=(2, 3), keepdim=True)
        std = edt_map.std(dim=(2, 3), keepdim=True)
        return (edt_map - mean) / (std + 1e-6)
    
    elif norm_type == 'relative-scaling':
        global_max = edt_map.max()
        return edt_map / global_max
    
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

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

    return euclidean_dist

def calculate_edt_3c(input):
    input = input.to(torch.float32)
    # Ensure input is 3D (C, H, W)
    if input.dim() == 4:
        input = input.squeeze(0)

    edt_channels = [calculate_edt(1 - input[i]) for i in range(3)]
    return torch.cat(edt_channels, dim=1)

class TopoLossMSE2D(torch.nn.Module):
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
            
            # Calculate EDT for both prediction and target
            edt_pred = calculate_edt_3c(binary_pred)
            edt_target = calculate_edt_3c(target[idx])

            #print(edt_pred.shape) # (1, 3, H, W)
            
            # Normalize EDT maps
            norm_edt_pred = normalize_edt_map(edt_pred)
            norm_edt_target = normalize_edt_map(edt_target)
            
            # Calculate topological loss for each channel
            item_loss = 0.
            for i in range(3):  # channel wise
                item_loss += getTopoLoss(norm_edt_pred[0][i], norm_edt_target[0][i], self.topo_window)
            
            item_loss /= 3  # Average over channels
            loss_per_item.append(item_loss)
        
        # Calculate mean loss over the batch
        mean_loss = torch.stack(loss_per_item).mean()
        
        # Apply topo weight
        final_loss = mean_loss * self.topo_weight
        
        return final_loss, mean_loss







