from __future__ import print_function, absolute_import
import numpy as np
import copy
import os
import os.path as osp

def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP

def evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL'):
    import os.path as osp
    # Define the expected dataset directory and file
    dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    
    # If the file does not exist, skip location evaluation.
    if not osp.exists(query_IDX_path):
        print("Warning: {} not found. Skipping location evaluation.".format(query_IDX_path))
        # Return zeroed CMC and mAP, or simply skip this evaluation.
        dummy_cmc = np.zeros(len(g_pids))
        dummy_mAP = 0.0
        return dummy_cmc, dummy_mAP

    # If the file exists, load it and continue as normal.
    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    # (Rest of your original evaluate_locations code follows here)
    in_cam = [330, 329, 507, 508, 509]
    out_cam = [436, 505, 336, 340, 639, 301]

    camera_file = '../test_track_scale.txt'
    camera_set = np.genfromtxt(camera_file, dtype='str')[:,0]
    q_locationids = camera_set[query_IDX]
    gallery_IDX = [i for i in range(camera_set.shape[0]) if i not in query_IDX]
    g_locationids = camera_set[gallery_IDX]
    for k in range(q_locationids.shape[0]):
        q_locationids[k] = 0 if int(q_locationids[k][9:12]) in in_cam else 1
    for k in range(g_locationids.shape[0]):
        g_locationids[k] = 0 if int(g_locationids[k][9:12]) in in_cam else 1

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)  # from small to large

    num_no_gt = 0  # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        location_index = np.argwhere(g_locationids == q_locationids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'SL':
            good_index = np.setdiff1d(good_index, location_index, assume_unique=True)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, location_index)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, location_index)
            junk_index = np.union1d(junk_index1, junk_index2)
    
        if good_index.size == 0:
            num_no_gt += 1
            continue
    
        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0] == 1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

def evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS'):
    """
    Compute CMC and mAP with scales.
    Args:
        distmat (numpy ndarray): distance matrix with shape (num_query, num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        mode: 'SS' for same size; 'DS' for different size.
    """
    import os.path as osp
    dataset_dir = '../../mevid'
    query_IDX_path = osp.join(dataset_dir, 'query_IDX.txt')
    
    # If the label file is missing, skip the scales evaluation.
    if not osp.exists(query_IDX_path):
        print("Warning: {} not found. Skipping scales evaluation.".format(query_IDX_path))
        dummy_cmc = np.zeros(len(g_pids))
        dummy_mAP = 0.0
        return dummy_cmc, dummy_mAP

    query_IDX = np.loadtxt(query_IDX_path).astype(np.int)
    scale_file = '../test_track_scale.txt'
    scale_set = np.genfromtxt(scale_file, dtype='str')[:, 1]
    q_scaleids = scale_set[query_IDX]
    gallery_IDX = [i for i in range(scale_set.shape[0]) if i not in query_IDX]
    g_scaleids = scale_set[gallery_IDX]

    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)  # from small to large

    num_no_gt = 0  # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        scale_index = np.argwhere(g_scaleids == q_scaleids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if mode == 'DS':
            good_index = np.setdiff1d(good_index, scale_index, assume_unique=True)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.intersect1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)
        else:
            good_index = np.intersect1d(good_index, scale_index)
            junk_index1 = np.intersect1d(query_index, camera_index)
            junk_index2 = np.setdiff1d(query_index, scale_index)
            junk_index = np.union1d(junk_index1, junk_index2)

        if good_index.size == 0:
            num_no_gt += 1
            continue

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0] == 1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query samples do not have groundtruth.".format(num_no_gt))

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP

# SC, CC and STD
def evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='SC'):
    """
    Compute CMC and mAP with clothes.
    
    Args:
        distmat (numpy ndarray): distance matrix (num_query x num_gallery).
        q_pids (numpy array): person IDs for query samples.
        g_pids (numpy array): person IDs for gallery samples.
        q_camids (numpy array): camera IDs for query samples.
        g_camids (numpy array): camera IDs for gallery samples.
        q_clothids (numpy array): cloth labels for query samples.
        g_clothids (numpy array): cloth labels for gallery samples.
        mode: 'SC' for same clothes, 'CC' for clothes-changing, or 'STD' for standard (ignore cloth).
    
    Returns:
        (CMC, mAP): The CMC curve and mean average precision.
    """
    assert mode in ['SC', 'CC', 'STD']
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)  # sorted indices, smallest first
    num_no_gt = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    for i in range(num_q):
        # Ground truth: gallery images with same identity.
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        
        if mode == 'STD':
            # Standard evaluation ignores cloth labels.
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            junk_index = np.intersect1d(query_index, camera_index)
        else:
            # First, compute the basic good set (same identity, different camera)
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            cloth_index = np.argwhere(g_clothids == q_clothids[i])
            if mode == 'SC':
                # Same Clothes: good if cloth label matches.
                good_index = np.intersect1d(good_index, cloth_index)
                junk_index1 = np.intersect1d(query_index, camera_index)
                junk_index2 = np.setdiff1d(query_index, cloth_index)
                junk_index = np.union1d(junk_index1, junk_index2)
            elif mode == 'CC':
                # Clothes-Changing: first check if there is any cloth variation.
                unique_cloths = np.unique(g_clothids[query_index])
                if unique_cloths.size == 1 and unique_cloths[0] == q_clothids[i]:
                    # No variation: fallback to standard (or SC) evaluation.
                    good_index = np.intersect1d(good_index, cloth_index)
                    junk_index1 = np.intersect1d(query_index, camera_index)
                    junk_index2 = np.setdiff1d(query_index, cloth_index)
                    junk_index = np.union1d(junk_index1, junk_index2)
                else:
                    # Remove gallery images with the same cloth label.
                    good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
                    junk_index1 = np.intersect1d(query_index, camera_index)
                    junk_index2 = np.intersect1d(query_index, cloth_index)
                    junk_index = np.union1d(junk_index1, junk_index2)
        if good_index.size == 0:
            num_no_gt += 1
            continue

        ap_tmp, cmc_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        CMC += cmc_tmp
        AP += ap_tmp

    if (num_q - num_no_gt) != 0:
        CMC = CMC / (num_q - num_no_gt)
        mAP = AP / (num_q - num_no_gt)
    else:
        mAP = 0

    return CMC, mAP


# SC and CC
# def evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_clothids, g_clothids, mode='SC'):
#     """
#     Compute CMC and mAP with clothes.
    
#     Args:
#         distmat (numpy ndarray): distance matrix (num_query x num_gallery).
#         q_pids (numpy array): person IDs for query samples.
#         g_pids (numpy array): person IDs for gallery samples.
#         q_camids (numpy array): camera IDs for query samples.
#         g_camids (numpy array): camera IDs for gallery samples.
#         q_clothids (numpy array): cloth labels for query samples.
#         g_clothids (numpy array): cloth labels for gallery samples.
#         mode: 'SC' for same clothes; 'CC' for clothes-changing.
    
#     Returns:
#         (CMC, mAP): The CMC curve and mean average precision.
#     """
#     assert mode in ['SC', 'CC']
#     num_q, num_g = distmat.shape
#     index = np.argsort(distmat, axis=1)  # sorted indices, smallest first
#     num_no_gt = 0
#     CMC = np.zeros(len(g_pids))
#     AP = 0

#     for i in range(num_q):
#         # Ground truth indices: gallery images of the same person
#         query_index = np.argwhere(g_pids == q_pids[i])
#         camera_index = np.argwhere(g_camids == q_camids[i])
#         cloth_index = np.argwhere(g_clothids == q_clothids[i])
#         # Basic good set: same identity but from a different camera.
#         good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        
#         if mode == 'SC':
#             # For same-clothes mode, require that the cloth label matches.
#             good_index = np.intersect1d(good_index, cloth_index)
#             junk_index1 = np.intersect1d(query_index, camera_index)
#             junk_index2 = np.setdiff1d(query_index, cloth_index)
#             junk_index = np.union1d(junk_index1, junk_index2)
#         else:  # mode == 'CC'
#             # Check if there is cloth variation in the ground truth.
#             unique_cloths = np.unique(g_clothids[query_index])
#             if unique_cloths.size == 1 and unique_cloths[0] == q_clothids[i]:
#                 # If there is no cloth variation, fallback to SC.
#                 good_index = np.intersect1d(good_index, cloth_index)
#                 junk_index1 = np.intersect1d(query_index, camera_index)
#                 junk_index2 = np.setdiff1d(query_index, cloth_index)
#                 junk_index = np.union1d(junk_index1, junk_index2)
#             else:
#                 # For clothes-changing mode, remove gallery images with the same cloth label.
#                 good_index = np.setdiff1d(good_index, cloth_index, assume_unique=True)
#                 junk_index1 = np.intersect1d(query_index, camera_index)
#                 junk_index2 = np.intersect1d(query_index, cloth_index)
#                 junk_index = np.union1d(junk_index1, junk_index2)
                
#         if good_index.size == 0:
#             num_no_gt += 1
#             continue
        
#         ap_tmp, cmc_tmp = compute_ap_cmc(index[i], good_index, junk_index)
#         CMC = CMC + cmc_tmp
#         AP += ap_tmp

#     if (num_q - num_no_gt) != 0:
#         CMC = CMC / (num_q - num_no_gt)
#         mAP = AP / (num_q - num_no_gt)
#     else:
#         mAP = 0

#     return CMC, mAP