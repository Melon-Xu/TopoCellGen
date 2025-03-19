# Topoloss using CubicalRipser library

from __future__ import print_function, division

import numpy as np
from pylab import *
import torch
from ripser import ripser
import cripser as cr
import ot
import FastGeodis
from gudhi.wasserstein import wasserstein_distance


class TopoLossMSE2D(torch.nn.Module):
    """Weighted Topological loss
    """

    def __init__(self, topo_weight, topo_window):
        super().__init__()
        print("Topo weight: {}".format(topo_weight))
        self.topo_weight = topo_weight
        self.topo_window = topo_window

    def forward(self, pred, target):
        loss_val = 0.

        for idx in range(pred.size()[0]): # batchsize=N
            loss_val += getTopoLoss(pred[idx, 0, :, : ], target[idx, 0, :, : ], self.topo_window)

        loss_val *= self.topo_weight

        return loss_val


def getCriticalPoints_cr(likelihood):
        
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")
    pd_arr_lh = pd[pd[:, 0] == 1] # 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time

    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0

    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa

def get_matchings(lh, gt):
    
    _, matchings = wasserstein_distance(lh, gt, matching=True)

    dgm_to_diagonal = matchings[matchings[:,1] == -1, 0]
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0)

    return dgm_to_diagonal, off_diagonal_match


def compute_dgm_force(lh_dgm, gt_dgm):

    if lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    
    if (gt_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(gt_dgm[:, 1] - gt_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        idx_holes_to_remove, off_diagonal_match = get_matchings(lh_dgm, gt_dgm)
    
    return idx_holes_to_remove, off_diagonal_match


def getTopoLoss(pred_tensor, gt_tensor, topo_window):

    if pred_tensor.ndim != 2:
        print("incorrct dimension")
    
    likelihood = pred_tensor.clone()
    gt = gt_tensor.clone()

    likelihood = torch.squeeze(likelihood).cpu().detach().numpy()
    gt = torch.squeeze(gt).cpu().detach().numpy()

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    for y in range(0, likelihood.shape[0], topo_window):
        for x in range(0, likelihood.shape[1], topo_window):
            lh_patch = likelihood[y:min(y + topo_window, likelihood.shape[0]),
                         x:min(x + topo_window, likelihood.shape[1])]
            gt_patch = gt[y:min(y + topo_window, gt.shape[0]),
                         x:min(x + topo_window, gt.shape[1])]

            if(np.min(lh_patch) == 1 or np.max(lh_patch) == 0): continue
            if(np.min(gt_patch) == 1 or np.max(gt_patch) == 0): continue
            
            # Get the critical points of predictions and ground truth
            pd_lh, bcp_lh, dcp_lh, pairs_lh_pa = getCriticalPoints_cr(lh_patch)
            pd_gt, bcp_gt, dcp_gt, pairs_lh_gt = getCriticalPoints_cr(gt_patch)

            # If the pairs not exist, continue for the next loop
            if not(pairs_lh_pa): continue
            if not(pairs_lh_gt): continue

            idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh, pd_gt)

            idx_holes_to_remove = []
            off_diagonal_match = []

            if (len(idx_holes_to_remove_for_matching) > 0):
                for i in idx_holes_to_remove_for_matching:
                    index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
                    idx_holes_to_remove.append(index_pd_lh_removed)
            
            if len(off_diagonal_for_matching) > 0:
                for idx, (i, j) in enumerate(off_diagonal_for_matching):
                    index_pd_lh = np.where(np.all(pd_lh == pd_lh[i], axis=1))[0][0]
                    index_pd_gt = np.where(np.all(pd_gt == pd_gt[j], axis=1))[0][0]
                    off_diagonal_match.append((index_pd_lh, index_pd_gt))

            if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
                for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                            bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                        topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = pd_gt[j][0]
                    
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                        topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = pd_gt[j][1]
                
                for hole_indx in idx_holes_to_remove:
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(bcp_lh[hole_indx][0]), x + int(
                            bcp_lh[hole_indx][1])] = 1  # push to diagonal
                        if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = \
                                lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(bcp_lh[hole_indx][0]), x + int(bcp_lh[hole_indx][1])] = 1
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_weight_map[y + int(dcp_lh[hole_indx][0]), x + int(
                            dcp_lh[hole_indx][1])] = 1  # push to diagonal
                        if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                            0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                                likelihood.shape[1]):
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = \
                                lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                        else:
                            topo_cp_ref_map[y + int(dcp_lh[hole_indx][0]), x + int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((pred_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    return loss_topo