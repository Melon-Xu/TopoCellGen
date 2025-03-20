import os
import json
import numpy as np
import torch
from datetime import datetime
from scipy import linalg
from scipy import ndimage
from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn


##########################################################################
# PART 1: Feature Extraction 
##########################################################################
def extract_features(file_path, model_path, feat_file="feature_eval_gen.pth"):
    """
    Load the pre-trained model, extract features from all .npy files under
    file_path/npy, and save them to file_path/features/feature_eval_gen.pth
    """
    # 1) Prepare input / output folders
    input_folder = os.path.join(file_path, "npy")
    save_feature_path = os.path.join(file_path, 'features')
    os.makedirs(save_feature_path, exist_ok=True)

    # 2) Load and modify model
    #    Example: using an fcn_resnet50 model with final layers truncated
    model = fcn_resnet50(num_classes=3)
    model.load_state_dict(torch.load(model_path))

    # Remove layers to get intermediate features
    model.backbone.layer3 = nn.Identity()
    model.backbone.layer4 = nn.Identity()
    model.classifier = nn.Identity()
    model = model.cuda()
    model.eval()

    # Additional post-processing to pool features to 1x1
    post_process = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.AdaptiveAvgPool2d(output_size=(1, 1))
    )

    # 3) Extract features
    feat_list = []
    file_list = os.listdir(input_folder)

    # OPTIONAL: If you always expect the exact number of .npy files, you can keep this:
    # assert len(file_list) == 1550, "Number of .npy files does not match expected count!"

    for file_name in file_list:
        if file_name.endswith('.npy'):
            file_path_ = os.path.join(input_folder, file_name)
            den_img = np.load(file_path_)
            # Move to GPU and reorder from (H,W,C) --> (C,H,W)
            tensor_img = torch.tensor(den_img).permute(2, 0, 1)[None, :].float().cuda()

            # Forward pass (through truncated model) + post-process
            with torch.no_grad():
                encoder_out = model(tensor_img)['out'][0]
                feat = post_process(encoder_out).squeeze().cpu().numpy()

            feat_list.append(feat)

    # 4) Save features
    torch.save(np.array(feat_list), os.path.join(save_feature_path, feat_file))
    print(f"Saved generated features to {os.path.join(save_feature_path, feat_file)}")


##########################################################################
# PART 2: FID Calculation
##########################################################################
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) +
            np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def mean_cov_process_no_class(feats):
    """Return mean & covariance of feats (2D: NxD)."""
    all_mean = np.mean(feats, axis=0)
    all_cov = np.cov(feats, rowvar=False)
    return all_mean, all_cov


##########################################################################
# PART 3: MSE Calculation
##########################################################################
def count_dots(cell_map):
    labeled_array, num_cells = ndimage.label(cell_map)
    return num_cells

def calculate_mse(file_path):
    """
    Calculate Mean Squared Error between ground truth and generated cell counts,
    as stored in file_path/cell_counts.json
    """
    json_file_path = os.path.join(file_path, 'cell_counts.json')
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    total_mse = 0
    class_mse = [0] * 3  # 3 classes for BRCA dataset
    sample_size = len(data)
    print(f"Sample size: {sample_size}")
    
    for filename, counts in data.items():
        gt_counts = counts['ground_truth']
        gen_counts = counts['generated']
        
        # Calculate MSE for this sample (total cell count)
        total_gt = sum(gt_counts)
        total_gen = sum(gen_counts)
        sample_mse = np.abs(total_gt - total_gen)
        total_mse += sample_mse
        
        # Calculate MSE per class
        for i in range(3):  # 3 classes for BRCA
            class_mse[i] += np.abs(gt_counts[i] - gen_counts[i])
    
    # Average MSE across all samples
    total_mse /= sample_size
    class_mse = [mse / sample_size for mse in class_mse]
    
    return total_mse, class_mse, sample_size


##########################################################################
# PART 4: Bringing it all together for evaluation
##########################################################################
def evaluate_and_save_results(file_path,
                              ref_feature_path="path/to/ref_features/reference_features.pth",
                              gen_model_path="path/to/test_fid_models/brcam2c_latest_2000_3channel.pth"):
    """
    1) Extract features for newly generated data using the generator model.
    2) Load reference features and generated features.
    3) Calculate MSE, FID, etc. and save results to file_path.
    """
    # 1) Extract features for generated samples
    extract_features(file_path, gen_model_path, feat_file="feature_eval_gen.pth")

    # 2) Calculate MSE from cell_counts.json
    total_mse, class_mse, sample_size = calculate_mse(file_path)

    # 3) Calculate FID
    real_feats = torch.load(ref_feature_path)
    fake_feats = torch.load(os.path.join(file_path, "features", "feature_eval_gen.pth"))

    real_mean, real_cov = mean_cov_process_no_class(real_feats)
    fake_mean, fake_cov = mean_cov_process_no_class(fake_feats)
    fid_score = calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)

    # 4) Prepare and save results
    results = f"""Evaluation Results:
                    Date and Time: {datetime.now()}
                    Sample Size: {sample_size}

                    Total MSE: {total_mse}
                    Class MSE:
                    - Class 1 (Type-0): {class_mse[0]}
                    - Class 2 (Type-1): {class_mse[1]}
                    - Class 3 (Type-2): {class_mse[2]}

                    FID Score: {fid_score}
                """
    result_file_path = os.path.join(file_path, 'evaluation_results.txt')
    with open(result_file_path, 'w') as f:
        f.write(results)
    
    print(f"Results saved to {result_file_path}")
    return total_mse, class_mse, fid_score


##########################################################################
# MAIN
##########################################################################
if __name__ == "__main__":
    # Example usage
    file_path = "path/to/your/generated_data" # e.g., path/to/2025-03-19/11-47-15

    # Path to the model used for feature extraction
    gen_model_path = "path/to/test_fid_models/brcam2c_latest_2000_3channel.pth"

    # Path to your precomputed reference features
    ref_feature_path = "path/to/ref_features/feature_eval_ref_brca.pth"

    total_mse, class_mse, fid_score = evaluate_and_save_results(
        file_path=file_path,
        ref_feature_path=ref_feature_path,
        gen_model_path=gen_model_path
    )

    print(f"Evaluation complete. FID: {fid_score}, Total MSE: {total_mse}")
