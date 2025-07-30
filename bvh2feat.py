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
            'Head_gamma', 'Neck_alpha', 'Neck_beta', 'Neck_gamma', 'Spine1_alpha',
            'Spine1_beta', 'Spine1_gamma', 'Spine_alpha', 'Spine_beta',
            'Spine_gamma', 'Hips_alpha', 'Hips_beta', 'Hips_gamma',
            'Hips_Yposition', 'reference_dXposition', 'reference_dZposition',
            'reference_dYrotation']

    def create_pipeline(include_mirror):
        steps = [
            ('dwnsampl', DownSampler(tgt_fps=fps)),
        ]

        # Insert mirror early
        if include_mirror:
            steps.append(('mir', Mirror(axis='X', append=True)))

        # Then do joint selection and other transforms
        steps += [
            ('jtsel', JointSelector([
                'Spine','Spine1','Neck','Head',
                'RightUpLeg','RightLeg','RightFoot',
                'LeftUpLeg','LeftLeg', 'LeftFoot',
                'RightShoulder','RightArm','RightForeArm','RightHand',
                'LeftShoulder','LeftArm','LeftForeArm','LeftHand'
            ], include_root=True)),

            ('root', RootTransformer('pos_rot_deltas', position_smoothing=1, rotation_smoothing=1)),
            ('drop', ColumnDropper(['Hips_Xposition', 'Hips_Zposition'])),
            ('exp', MocapParameterizer('expmap')),
            # ('cnst', ConstantsRemover()),
            ('cnt', FeatureCounter()),
            ('npf', Numpyfier(indices=index))
        ]
        return Pipeline(steps)

    try:
        data_pipeline = create_pipeline(include_mirror=True)
        out_data = data_pipeline.fit_transform([data])

        # the datapipe will append the mirrored files to the end
        assert len(out_data) == 2

        jl.dump(data_pipeline, os.path.join(pipeline_dir, 'data_pipe.sav'))
        
        print("Saving features for file:", file)    
        with open(os.path.join(dest_dir, file + ".expmap_30fps.pkl"), 'wb') as fp:
            print(out_data[0].shape)
            df0 = pd.DataFrame(out_data[0], columns=index)
            df0.index = pd.Series([pd.Timedelta(seconds=(1/fps) * i) for i in range(len(df0.index))])
            pkl.dump(df0, fp)
        with open(os.path.join(dest_dir, file + "_mirrored.expmap_30fps.pkl"), 'wb') as fp:
            print(out_data[1].shape)
            df1 = pd.DataFrame(out_data[1], columns=index)
            df1.index = pd.Series([pd.Timedelta(seconds=(1/fps) * i) for i in range(len(df1.index))])
            pkl.dump(df1, fp)

    except Exception as e:
        print(f"Error processing {file} with mirror: {e}. Trying without mirror.")
        try:
            data_pipeline = create_pipeline(include_mirror=False)
            out_data = data_pipeline.fit_transform([data])
            print(out_data[0].shape)

            jl.dump(data_pipeline, os.path.join(pipeline_dir, 'data_pipe_no_mirror.sav'))
            
            print("Saving features for file (without mirror):", file)
            with open(os.path.join(dest_dir, file + ".expmap_30fps.pkl"), 'wb') as fp:
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
    parser.add_argument('--bvh_dir', '-orig', default=r"/host_data/van/LDA/data/motorica/bvh",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default=r"/host_data/van/LDA/data/motorica/feat",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default=r"/host_data/van/LDA/data/motorica/feat",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--error_dir', '-err', default=r"/host_data/van/LDA/data/motorica/error_bvh",
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