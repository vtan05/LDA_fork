import numpy as np
from . import utils as feat_utils

BVH_JOINT_NAMES = [
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"
]


def extract_manual_features(positions):
    """
    positions: (seq_len, n_joints, 3) numpy array
    """
    assert len(positions.shape) == 3
    features = []
    f = ManualFeatures(positions, joint_names=BVH_JOINT_NAMES)

    for _ in range(1, positions.shape[0]):
        pose_features = []

        # arm movement relative to hips & neck
        pose_features.append(f.f_nmove("Neck", "RightUpLeg", "LeftUpLeg", "RightHand", 1.8 * f.hl))
        pose_features.append(f.f_nmove("Neck", "LeftUpLeg", "RightUpLeg", "LeftHand", 1.8 * f.hl))

        # chest plane wrt wrists
        pose_features.append(f.f_nplane("Spine2", "Neck", "Neck", "RightHand", 0.2 * f.hl))
        pose_features.append(f.f_nplane("Spine2", "Neck", "Neck", "LeftHand", 0.2 * f.hl))

        # belly (Spine) movement wrt wrists
        pose_features.append(f.f_move("Spine", "Spine2", "Spine2", "RightHand", 1.8 * f.hl))
        pose_features.append(f.f_move("Spine", "Spine2", "Spine2", "LeftHand", 1.8 * f.hl))

        # elbow angles
        pose_features.append(f.f_angle("RightForeArm", "RightArm", "RightForeArm", "RightHand", [0, 110]))
        pose_features.append(f.f_angle("LeftForeArm", "LeftArm", "LeftForeArm", "LeftHand", [0, 110]))

        # shoulder–wrist plane
        pose_features.append(f.f_nplane("LeftShoulder", "RightShoulder", "LeftHand", "RightHand", 2.5 * f.sw))

        # wrist cross-movement
        pose_features.append(f.f_move("LeftHand", "RightHand", "RightHand", "LeftHand", 1.4 * f.hl))
        pose_features.append(f.f_move("RightHand", "Spine", "LeftHand", "Spine", 1.4 * f.hl))
        pose_features.append(f.f_move("LeftHand", "Spine", "RightHand", "Spine", 1.4 * f.hl))

        # wrist speed
        pose_features.append(f.f_fast("RightHand", 2.5 * f.hl))
        pose_features.append(f.f_fast("LeftHand", 2.5 * f.hl))

        # feet relative to hips
        pose_features.append(f.f_plane("Spine", "LeftUpLeg", "LeftFoot", "RightFoot", 0.38 * f.hl))
        pose_features.append(f.f_plane("Spine", "RightUpLeg", "RightFoot", "LeftFoot", 0.38 * f.hl))

        # ankle vertical plane
        pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "RightFoot", 1.2 * f.hl))
        pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "LeftFoot", 1.2 * f.hl))

        # hip–ankle plane
        pose_features.append(f.f_nplane("LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot", 2.1 * f.hw))

        # knee angles
        pose_features.append(f.f_angle("RightLeg", "RightUpLeg", "RightLeg", "RightFoot", [0, 110]))
        pose_features.append(f.f_angle("LeftLeg", "LeftUpLeg", "LeftLeg", "LeftFoot", [0, 110]))

        # ankle speed
        pose_features.append(f.f_fast("RightFoot", 2.5 * f.hl))
        pose_features.append(f.f_fast("LeftFoot", 2.5 * f.hl))

        # torso–arm & torso–leg angles
        pose_features.append(f.f_angle("Neck", "Spine", "RightShoulder", "RightForeArm", [25, 180]))
        pose_features.append(f.f_angle("Neck", "Spine", "LeftShoulder", "LeftForeArm", [25, 180]))
        pose_features.append(f.f_angle("Neck", "Spine", "RightUpLeg", "RightLeg", [50, 180]))
        pose_features.append(f.f_angle("Neck", "Spine", "LeftUpLeg", "LeftLeg", [50, 180]))

        # balance / orientation
        pose_features.append(f.f_plane("RightFoot", "Neck", "LeftFoot", "Spine", 0.5 * f.hl))
        pose_features.append(f.f_angle("Neck", "Spine", "zero", "y_unit", [70, 110]))
        pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "RightHand", -1.2 * f.hl))
        pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "LeftHand", -1.2 * f.hl))

        # root motion (Spine base as root)
        pose_features.append(f.f_fast("Spine", 2.3 * f.hl))

        features.append(pose_features)
        f.next_frame()

    features = np.array(features, dtype=np.float32).mean(axis=0)
    return features


class ManualFeatures:
    def __init__(self, positions, joint_names=BVH_JOINT_NAMES):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1

        # humerus length (shoulder → elbow)
        self.hl = feat_utils.distance_between_points(
            [0.2, 0.25, 0.0],   # approx LeftShoulder
            [0.45, 0.20, 0.0],  # approx LeftForeArm
        )
        # shoulder width
        self.sw = feat_utils.distance_between_points(
            [0.2, 0.25, 0.0],   # LeftShoulder
            [-0.2, 0.25, 0.0],  # RightShoulder
        )
        # hip width
        self.hw = feat_utils.distance_between_points(
            [0.06, -0.3, 0.0],  # LeftUpLeg
            [-0.06, -0.3, 0.0], # RightUpLeg
        )

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [0, min([y for (_, y, _) in self.positions[self.frame_num]]), 0]
        return self.positions[self.frame_num][self.joint_names.index(j)]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][self.joint_names.index(j)]

    # same feature functions as before...
    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, j3, j3_prev, range)

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, j4, j4_prev, range)

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return feat_utils.velocity_above_threshold(j1, j1_prev, threshold)
