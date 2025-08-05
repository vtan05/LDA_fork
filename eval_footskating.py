import os
import numpy as np
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline

# === Configuration ===
fps = 30
slide_threshold = 0.005       # Tangential velocity threshold for sliding
contact_threshold = 0.05      # Max Y from virtual ground to consider contact
print_debug = False           # Toggle per-frame debug logging

# BVH joint names used in your pipeline
joints = [
    'Hips', 'Spine','Spine1','Neck','Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightUpLeg', 'RightLeg', 'RightFoot',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot'
]

data_pipe = Pipeline([
    ('dwnsampl', DownSampler(tgt_fps=fps)),
    ('jtsel', JointSelector(joints, include_root=False)),
    ('exp', MocapParameterizer('position')),
    ('npf', Numpyfier())
])

def calc_foot_skating_ratio_bvh(joint_pos, joint_names, ground_height=0, slide_threshold=0.005, contact_threshold=0.05):
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    l_foot_idx = name_to_idx['LeftFoot']
    r_foot_idx = name_to_idx['RightFoot']

    pred_joint_xyz = joint_pos[:, [l_foot_idx, r_foot_idx], :]  # [T, 2, 3]
    pred_vel = np.zeros_like(pred_joint_xyz)
    pred_vel[:-1] = pred_joint_xyz[1:] - pred_joint_xyz[:-1]

    left_foot_y = joint_pos[:, l_foot_idx, 1]
    right_foot_y = joint_pos[:, r_foot_idx, 1]

    left_fc_mask = (left_foot_y <= ground_height + contact_threshold)
    right_fc_mask = (right_foot_y <= ground_height + contact_threshold)

    left_pred_vel = pred_vel[:, 0:1, :]
    right_pred_vel = pred_vel[:, 1:2, :]

    left_pred_vel[~left_fc_mask] = 0
    right_pred_vel[~right_fc_mask] = 0

    left_velocity_tangent = np.abs(left_pred_vel[:, :, [0, 2]]).mean(axis=-1)  # [T, 1]
    right_velocity_tangent = np.abs(right_pred_vel[:, :, [0, 2]]).mean(axis=-1)

    left_slide_frames = (left_velocity_tangent > slide_threshold).squeeze(-1)
    right_slide_frames = (right_velocity_tangent > slide_threshold).squeeze(-1)

    left_static_num = np.sum(left_fc_mask)
    right_static_num = np.sum(right_fc_mask)

    left_ratio = np.sum(left_slide_frames) / max(left_static_num, 1)
    right_ratio = np.sum(right_slide_frames) / max(right_static_num, 1)

    # Optional: Debug logging
    if print_debug:
        print("\n--- DEBUG FRAMEWISE ---")
        for i in range(len(left_foot_y)):
            print(f"[Frame {i:03}] L_Y={left_foot_y[i]:.3f}, VXZ={left_velocity_tangent[i][0]:.5f}, "
                  f"Contact={left_fc_mask[i]}, Slide={left_slide_frames[i]}")

    return float(left_ratio), float(right_ratio), left_fc_mask, right_fc_mask, left_slide_frames, right_slide_frames

# === Main Script ===
if __name__ == '__main__':
    bvh_dir = '/host_data/van/LDA/results/motorica'  # <-- Replace if needed

    left_ratio_list = []
    right_ratio_list = []

    parser = BVHParser()

    for file in os.listdir(bvh_dir):
        if not file.endswith('.bvh'):
            continue

        full_path = os.path.join(bvh_dir, file)
        print(f"\n📂 Parsing: {file}")
        parsed_data = parser.parse(full_path)

        piped_data = data_pipe.fit_transform([parsed_data])
        piped_data = piped_data.squeeze()
        joint_pos = piped_data.reshape(piped_data.shape[0], -1, 3)

        # ✅ Convert to meters
        joint_pos = joint_pos / 100.0

        # ✅ Use virtual ground height = minimum foot Y position
        min_left = joint_pos[:, joints.index('LeftFoot'), 1].min()
        min_right = joint_pos[:, joints.index('RightFoot'), 1].min()
        ground_height = min(min_left, min_right)

        # Optional: check foot height range
        print(f"LeftFoot Y range (meters): {joint_pos[:, joints.index('LeftFoot'), 1].min():.3f} → {joint_pos[:, joints.index('LeftFoot'), 1].max():.3f}")
        print(f"Computed virtual ground height: {ground_height:.3f}")

        left_ratio, right_ratio, lmask, rmask, lslide, rslide = calc_foot_skating_ratio_bvh(
            joint_pos, joints, ground_height, slide_threshold, contact_threshold
        )

        print(f"  Contact frames: L={np.sum(lmask)}, R={np.sum(rmask)}")
        print(f"  Slide frames:   L={np.sum(lslide)}, R={np.sum(rslide)}")
        print(f"  Skating ratio:  L={left_ratio:.4f}, R={right_ratio:.4f}")

        left_ratio_list.append(left_ratio)
        right_ratio_list.append(right_ratio)

    print("\n✅ FINAL RESULTS")
    print("Left foot skating ratio: ", np.mean(left_ratio_list))
    print("Right foot skating ratio:", np.mean(right_ratio_list))
    print("Average skating ratio:   ", (np.mean(left_ratio_list) + np.mean(right_ratio_list)) / 2)
    print("Evaluated BVH directory: ", bvh_dir)
