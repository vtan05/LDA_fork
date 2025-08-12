import os
import torch
from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from utils.logging_mixin import LoggingMixin

# def get_joint_indices(joint_names):
#     """Generate indices for selected joints with alpha, beta, gamma suffixes."""
#     return [f"{joint}_{axis}" for joint in joint_names for axis in ['alpha', 'beta', 'gamma']]

if __name__ == "__main__":

    # Parse BVH file
    parser = BVHParser()
    bvh_file = r"/host_data/van/LDA/data/finedance/ybot_bvh/finedance_ClassicDunHuang_sFM_cAll_d02_mClassic_ch01_xuangupiao_051.bvh"
    parsed_data = parser.parse(bvh_file)
    print("Parsed BVH Data:", parsed_data)

    # Define joint names for processing
    selected_joints = [
        'Spine','Spine1','Spine2','Neck','Head',
        'RightUpLeg','RightLeg','RightFoot',
        'LeftUpLeg','LeftLeg', 'LeftFoot',
        'RightShoulder','RightArm','RightForeArm','RightHand',
        'LeftShoulder','LeftArm','LeftForeArm','LeftHand'
    ]

    index = ['RightFoot_alpha', 'RightFoot_beta', 'RightFoot_gamma',
            'RightLeg_alpha', 'RightLeg_beta', 'RightLeg_gamma', 'RightUpLeg_alpha',
            'RightUpLeg_beta', 'RightUpLeg_gamma', 'LeftFoot_alpha',
            'LeftFoot_beta', 'LeftFoot_gamma', 'LeftLeg_alpha', 'LeftLeg_beta',
            'LeftLeg_gamma', 'LeftUpLeg_alpha', 'LeftUpLeg_beta', 'LeftUpLeg_gamma',
            'RightHand_alpha', 'RightHand_beta', 'RightHand_gamma',
            'RightForeArm_alpha', 'RightForeArm_beta', 'RightForeArm_gamma',
            'RightArm_alpha', 'RightArm_beta', 'RightArm_gamma',
            'RightShoulder_alpha', 'RightShoulder_beta', 'RightShoulder_gamma',
            'LeftHand_alpha', 'LeftHand_beta', 'LeftHand_gamma',
            'LeftForeArm_alpha', 'LeftForeArm_beta', 'LeftForeArm_gamma',
            'LeftArm_alpha', 'LeftArm_beta', 'LeftArm_gamma', 'LeftShoulder_alpha',
            'LeftShoulder_beta', 'LeftShoulder_gamma', 'Head_alpha', 'Head_beta',
            'Head_gamma', 'Neck_alpha', 'Neck_beta', 'Neck_gamma', 'Spine2_alpha',
            'Spine2_beta', 'Spine2_gamma', 'Spine1_alpha',
            'Spine1_beta', 'Spine1_gamma', 'Spine_alpha', 'Spine_beta',
            'Spine_gamma', 'Hips_alpha', 'Hips_beta', 'Hips_gamma',
            'Hips_Yposition', 'reference_dXposition', 'reference_dZposition',
            'reference_dYrotation']

    # # Dynamically generate indices for selected joints
    # index = get_joint_indices(selected_joints) + [
    #     'Hips_Yposition', 'reference_dXposition', 'reference_dZposition', 'reference_dYrotation',
    #     'Hips_alpha', 'Hips_beta', 'Hips_gamma'
    # ]

    # Define data processing pipeline
    data_pipeline = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=24)),
        ('mir', MirrorFinedance(axis='X', append=True)),
        ('jtsel', JointSelector(selected_joints, include_root=True)),
        ('root', RootTransformer('pos_rot_deltas', position_smoothing=1, rotation_smoothing=1)),
        ('drop', ColumnDropper(['Hips_Xposition', 'Hips_Zposition'])),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('npf', Numpyfier(indices=index)),
        ('cnt', FeatureCounter())
    ])

    # Process the parsed data through the pipeline
    try:
        piped_data = data_pipeline.fit_transform([parsed_data])
        data = torch.from_numpy(piped_data)
        print(f"Tensor shape from BVH file: {data.shape}")
    except Exception as e:
        print(f"Error during pipeline processing: {e}")
        exit(1)

    '''
    Convert tensor back to BVH format
    '''
    logging = LoggingMixin()

    # Number of features after processing
    n_feats = data_pipeline['cnt'].n_features

    # Ensure RootTransformer is properly configured for inverse transformation
    data_pipeline['root'].separate_root = False

    # # Validate indices before inverse transformation
    # missing_indices = [idx for idx in index if idx not in data_pipeline['npf'].indices]
    # if missing_indices:
    #     print(f"Warning: Missing indices for inverse transformation: {missing_indices}")
    #     index = [idx for idx in index if idx in data_pipeline['npf'].indices]

    # Perform inverse transformation to reconstruct BVH data
    print(f"Inverse transforming data with shape: {data.shape} and n_feats: {n_feats}")
    bvh_data = data_pipeline.inverse_transform(data[1:2, :, :n_feats])

    # Save the reconstructed BVH file
    output_dir = "."
    output_filename = "test"
    logging.write_bvh(bvh_data, output_dir, output_filename)

    print(f"Reconstructed BVH file saved as: {os.path.join(output_dir, output_filename)}")
