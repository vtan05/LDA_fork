
import numpy as np 
from features.kinetic import extract_kinetic_features
from features.manual_motorica import extract_manual_features
from scipy import linalg
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import os


def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def quantized_metrics(predicted_bvh_root, gt_bvh_root):


    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []


    # for bvh in os.listdir(predicted_bvh_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_bvh_root, 'kinetic_features', bvh))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_bvh_root, 'manual_features_new', bvh)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_bvh_root, 'kinetic_features', bvh)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_bvh_root, 'manual_features_new', bvh)))

    pred_features_k = [np.load(os.path.join(predicted_bvh_root, 'kinetic_features', bvh)) for bvh in os.listdir(os.path.join(predicted_bvh_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_bvh_root, 'manual_features_new', bvh)) for bvh in os.listdir(os.path.join(predicted_bvh_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_bvh_root, 'kinetic_features', bvh)) for bvh in os.listdir(os.path.join(gt_bvh_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_bvh_root, 'manual_features_new', bvh)) for bvh in os.listdir(os.path.join(gt_bvh_root, 'manual_features_new'))]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 

#   T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize(gt_freatures_k)
    # gt_freatures_m = normalize(gt_freatures_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_g': fid_m, 'div_k': div_k, 'div_g' : div_m}
    return metrics


def calc_fid(kps_gen, kps_gt):

    # print(kps_gen.shape)
    # print(kps_gt.shape)

    # kps_gen = kps_gen[:19, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []

    bvh_files = [f for f in os.listdir(root) if f.lower().endswith(".bvh") and os.path.isfile(os.path.join(root, f))]

    for bvh in bvh_files:
        print(bvh)
        if os.path.isdir(os.path.join(root, bvh)):
            continue
        joint3d = process_motion(os.path.join(root, bvh))
        np.save(os.path.join(root, 'kinetic_features', bvh), extract_kinetic_features(joint3d.reshape(-1, 19, 3)))
        np.save(os.path.join(root, 'manual_features_new', bvh), extract_manual_features(joint3d.reshape(-1, 19, 3)))


# def to_relative(joint3d):
#     """
#     Convert absolute 3D joint positions to root-relative coordinates.

#     Args:
#         joint3d: np.ndarray of shape (T, J*3) or (T, J, 3)
#         zero_root: if True, zero out the root joint (index 0)

#     Returns:
#         np.ndarray of the same shape as input, but root-relative
#     """
#     if joint3d.ndim == 2:
#         T, C = joint3d.shape
#         assert C % 3 == 0, "Input must have channels multiple of 3"
#         J = C // 3
#         x = joint3d.reshape(T, J, 3)
#         flat = True
#     elif joint3d.ndim == 3:
#         T, J, _ = joint3d.shape
#         x = joint3d
#         flat = False
#     else:
#         raise ValueError("Input must have shape (T, J*3) or (T, J, 3)")

#     root = x[:, 0:1, :]  # (T, 1, 3)

#     # Subtract root from all joints
#     x_rel = x - root

#     x_rel[:, 0, :] -= x[:, 0, :]

#     return x_rel.reshape(T, -1) if flat else x_rel
    
# def hybrid_root_global_others_relative(joints, root_index=0, flatten_like_input=True):
#     """
#     Build a hybrid representation:
#       - Root joint stays in GLOBAL coordinates
#       - All other joints become RELATIVE to the root at the same frame

#     Parameters
#     ----------
#     joints : np.ndarray or torch.Tensor
#         Shape (T, J, 3) or (T, J*3).
#     root_index : int
#         Index of the root joint in the J dimension.
#     flatten_like_input : bool
#         If True and input was flat (T, J*3), return flat too. Otherwise return (T, J, 3).

#     Returns
#     -------
#     same type/shape as input (by default), with:
#       out[:, root_index, :] = original global root positions
#       out[:, other, :]      = (original[other] - original[root_index]) per frame
#     """
#     # Lazy imports to support both numpy and torch
#     try:
#         import torch
#         is_torch = isinstance(joints, torch.Tensor)
#     except Exception:
#         torch = None
#         is_torch = False

#     # --- reshape to (T, J, 3)
#     if joints.ndim == 2:
#         T, C = joints.shape
#         assert C % 3 == 0, "Channel dimension must be a multiple of 3"
#         J = C // 3
#         x = joints.view(T, J, 3) if is_torch else joints.reshape(T, J, 3)
#         was_flat = True
#     else:
#         x = joints
#         T, J, _ = x.shape
#         was_flat = False

#     # --- build hybrid
#     out = x.clone() if is_torch else x.copy()       # start from absolute positions
#     root = x[:, root_index:root_index+1, :]         # (T, 1, 3), global root
#     # make all non-root joints relative to the root (per-frame)
#     if root_index == 0:
#         out[:, 1:, :] = x[:, 1:, :] - root
#     elif root_index == J - 1:
#         out[:, :-1, :] = x[:, :-1, :] - root
#     else:
#         out[:, :root_index, :]  = x[:, :root_index, :]  - root
#         out[:, root_index+1:, :] = x[:, root_index+1:, :] - root
#     # keep the root joint itself in global coords (already true in `out[:, root_index]`)

#     # --- return shape like input if requested
#     if flatten_like_input and was_flat:
#         out = out.view(T, -1) if is_torch else out.reshape(T, -1)
#     return out


def process_motion(motion_path):

    fps = 30 # for finedance
    joints = ['Spine','Spine1','Neck','Head',
                'RightUpLeg','RightLeg','RightFoot',
                'LeftUpLeg','LeftLeg', 'LeftFoot',
                'RightShoulder','RightArm','RightForeArm','RightHand',
                'LeftShoulder','LeftArm','LeftForeArm','LeftHand']
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps)),
        ('jtsel', JointSelector(joints, include_root=True)),
        ('exp', MocapParameterizer('position')), 
        ('npf', Numpyfier())
    ])

    parser = BVHParser()
    parsed_data = parser.parse(motion_path)
    piped_data = data_pipe.fit_transform([parsed_data])
    joint_pos = np.reshape(piped_data, [-1, 19*3]) * 100  # to cm
    print("joint_pos:", joint_pos.shape)  # (T, 19*3)

    roott = joint_pos[:1, :3]  # the root Tx72 (Tx(24x3))
    print("roott:", roott.shape)          # (1, 3)
    joint3d = joint_pos - np.tile(roott, (1, 19))  # Calculate relative offset with respect to root

    # relative
    joint3d_relative = joint3d.copy()
    joint3d_relative = joint3d_relative.reshape(-1, 19, 3)
    joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
    return joint3d_relative
    # joint_pos = np.reshape(piped_data, [-1, 19, 3])

    # joint_rel = hybrid_root_global_others_relative(joint_pos, root_index=0)    # (T, 19, 3)
    # joint_rel = to_relative(joint_pos)
    # return joint_rel
    # print("joint_pos:", joint_pos.shape)  # (T, 19, 3)
    # roott = joint_pos[:, 0, :]
    # print("roott:", roott.shape)          # (T, 3)
    # joint_pos = joint_pos - roott[:, None, :]  # Calculate relative offset with respect to root
    # print("joint_pos (rel):", joint_pos.shape)
    # return joint_pos

if __name__ == '__main__':


    gt_root = '/host_data/van/DTM/data/finedance/motorica_bvh'
    pred_root = '/host_data/van/DTM/results/finedance'
    print('Calculating and saving features')
    calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)


    print('Calculating metrics')
    print(gt_root)
    print(pred_root)
    print(quantized_metrics(pred_root, gt_root))