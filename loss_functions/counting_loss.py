import torch
import numpy as np
import torch.nn as nn

def binarize_map(map, threshold=0.7):
    return (map >= threshold).float()

def counting_loss_torch(gt_map, pred_map):
    assert gt_map.shape == pred_map.shape, "GT and prediction shapes should match"
    channel_num = gt_map.shape[1] # BRCA-M2C: 3; Lizard: 6

    losses = {}
    total_loss = 0.0
    batch_size = gt_map.shape[0]

    # binarize the prediction map
    binarized_pred_map = binarize_map(pred_map)

    for i in range(batch_size):
        for j in range(channel_num):
            gt_count = torch.sum(gt_map[i,j,:,:]) / 9.0
            
            pred_count = torch.sum(binarized_pred_map[i,j,:,:]) / 9.0

            channel_loss = torch.abs(gt_count - pred_count)

            losses[f'batch_{i}_channel_{j}'] = channel_loss.item()
            total_loss += channel_loss.item()

    total_loss /= (batch_size * 3.0)

    return total_loss, losses  


if __name__ == '__main__':

    # Example usage
    batch_size, channels, height, width = 6, 3, 256, 256

    # Create random tensors for demonstration
    predicted_clean = torch.rand(batch_size, channels, height, width)
    input_image = torch.zeros(batch_size, channels, height, width)

    # Calculate loss
    loss, losses = counting_loss_torch(input_image, predicted_clean)