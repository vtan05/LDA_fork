import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import shutil  
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *
from pymo.viz_tools import *

import joblib as jl


def extract_joint_angles(bvh_dir, file, dest_dir, pipeline_dir, error_dir, fps):
    p = BVHParser()

    if not os.path.exists(pipeline_dir):
        raise Exception("Pipeline dir for the motion processing ", pipeline_dir, " does not exist! Change -pipe flag value.")

    ff = os.path.join(bvh_dir, file + '.bvh')
    print("Processing file:", ff)
    data = p.parse(ff)
    # print_skel(data)

    index = ['Right_foot_alpha', 'Right_foot_beta', 'Right_foot_gamma',
            'Right_ankle_alpha', 'Right_ankle_beta', 'Right_ankle_gamma',
            'Right_knee_alpha', 'Right_knee_beta', 'Right_knee_gamma', 'Right_hip_alpha',
            'Right_hip_beta', 'Right_hip_gamma', 'Left_ankle_alpha',
            'Left_foot_alpha', 'Left_foot_beta', 'Left_foot_gamma',
            'Left_ankle_beta', 'Left_ankle_gamma', 'Left_knee_alpha', 'Left_knee_beta',
            'Left_knee_gamma', 'Left_hip_alpha', 'Left_hip_beta', 'Left_hip_gamma',
            'Right_wrist_alpha', 'Right_wrist_beta', 'Right_wrist_gamma',
            'Right_elbow_alpha', 'Right_elbow_beta', 'Right_elbow_gamma',
            'Right_shoulder_alpha', 'Right_shoulder_beta', 'Right_shoulder_gamma',
            'Right_collar_alpha', 'Right_collar_beta', 'Right_collar_gamma',
            'Left_wrist_alpha', 'Left_wrist_beta', 'Left_wrist_gamma',
            'Left_elbow_alpha', 'Left_elbow_beta', 'Left_elbow_gamma',
            'Left_shoulder_alpha', 'Left_shoulder_beta', 'Left_shoulder_gamma', 'Left_collar_alpha',
            'Left_collar_beta', 'Left_collar_gamma', 'Head_alpha', 'Head_beta',
            'Head_gamma', 'Neck_alpha', 'Neck_beta', 'Neck_gamma', 'Spine3_alpha',
            'Spine3_beta', 'Spine3_gamma', 'Spine2_alpha',
            'Spine2_beta', 'Spine2_gamma', 'Spine1_alpha', 'Spine1_beta',
            'Spine1_gamma', 'Pelvis_alpha', 'Pelvis_beta', 'Pelvis_gamma',
            'Pelvis_Yposition', 'reference_dXposition', 'reference_dZposition',
            'reference_dYrotation']

    def create_pipeline(include_mirror):
        steps = [
            ('dwnsampl', DownSampler(tgt_fps=fps)),
        ]

        # Insert mirror early
        if include_mirror:
            steps.append(('mir', MirrorFinedance(axis='X', append=True)))

        # Then do joint selection and other transforms
        steps += [
            ('jtsel', JointSelector([
                'Spine1','Spine2','Spine3','Neck','Head',
                'Right_hip','Right_knee','Right_ankle', 'Right_foot',
                'Left_hip','Left_knee', 'Left_ankle', 'Left_foot',
                'Right_collar','Right_shoulder','Right_elbow','Right_wrist',
                'Left_collar','Left_shoulder','Left_elbow','Left_wrist'
            ], include_root=True)),

            ('root', RootTransformer('pos_rot_deltas', position_smoothing=1, rotation_smoothing=1)),
            ('drop', ColumnDropper(['Pelvis_Xposition', 'Pelvis_Zposition'])),
            ('exp', MocapParameterizer('expmap')),
            ('cnst', ConstantsRemover()),
            ('npf', Numpyfier(indices=index)),
            ('cnt', FeatureCounter())
        ]
        return Pipeline(steps)

    try:
        data_pipeline = create_pipeline(include_mirror=False)
        out_data = data_pipeline.fit_transform([data])
        print(out_data[0].shape)
        n_feats = data_pipeline["cnt"].n_features

        if n_feats == 70:
            jl.dump(data_pipeline, os.path.join(pipeline_dir, 'data_pipe_no_mirror.sav'))
            
            print("Saving features for file (without mirror):", file)
            with open(os.path.join(dest_dir, "aist_" + file + ".expmap_30fps.pkl"), 'wb') as fp:
                df2 = pd.DataFrame(out_data[0], columns=index)
                df2.index = pd.Series([pd.Timedelta(seconds=(1/fps) * i) for i in range(len(df2.index))])
                pkl.dump(df2, fp)
                
    except Exception as e:
        print(f"Error processing {file} without mirror: {e}. Moving file to error directory.")
        error_file_path = os.path.join(bvh_dir, file + '.bvh')
        error_dest_path = os.path.join(error_dir, file + '.bvh')
        shutil.move(error_file_path, error_dest_path)


if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--bvh_dir', '-orig', default=r"/host_data/van/LDA/data/edge_aistpp/bvh",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default=r"/host_data/van/LDA/data/edge_aistpp/feat",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default=r"/host_data/van/LDA/data/edge_aistpp/feat",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--error_dir', '-err', default=r"/host_data/van/LDA/data/edge_aistpp/error_bvh",
                        help="Path where BVH files with errors will be moved")

    params = parser.parse_args()

    # Ensure the error directory exists
    if not os.path.exists(params.error_dir):
        os.makedirs(params.error_dir)

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    for r, d, f in os.walk(params.bvh_dir):
        for file in f:
            print(file)
            if '.bvh' in file:
                ff = os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    for file in files:
        extract_joint_angles(params.bvh_dir, file, params.dest_dir, params.pipeline_dir, params.error_dir, fps=30)