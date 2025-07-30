
import os
import torch

#from hparams import get_hparams
from utils.logging_mixin import LoggingMixin
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.writers import *
from pymo.viz_tools import *
from sklearn.pipeline import Pipeline


if __name__ == "__main__":

    # hparams, conf_name = get_hparams()
    # assert os.path.exists(
    #     hparams.dataset_root
    # ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    '''
    bvh -> torch
    '''
    parser = BVHParser()
    parsed_data = parser.parse(r"/host_data/van/LDA/data/motorica/bvh/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh")
    #parsed_data = parser.parse(r"/host_data/van/Dance/LDA/data/motorica_dance/bvh/kthstreet_gLH_sFM_cAll_d01_mLH4_ch04.bvh")
    # print_skel(parsed_data)

    # * Original LDA: original data pipeline for LDA character
    # data_pipeline = jl.load(Path(hparams.dataset_root) / hparams.Data["datapipe_filename"])

    # * New data pipeline for YBot
    # every setup except for JointSelector follows original pipeline

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


    data_pipeline = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        ('mir', Mirror(axis='X', append=False)),
        ('jtsel', JointSelector([
                'Spine','Spine1','Neck','Head',
                'RightUpLeg','RightLeg','RightFoot',
                'LeftUpLeg','LeftLeg', 'LeftFoot',
                'RightShoulder','RightArm','RightForeArm','RightHand',
                'LeftShoulder','LeftArm','LeftForeArm','LeftHand'], include_root=True)),
        ('root', RootTransformer('pos_rot_deltas', position_smoothing=1, rotation_smoothing=1)),
        ('drop', ColumnDropper(['Hips_Xposition', 'Hips_Zposition'])),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('cnt', FeatureCounter()),
        ('npf', Numpyfier(indices=index))
    ])
    piped_data = data_pipeline.fit_transform([parsed_data])
    data = torch.from_numpy(piped_data)
    print(f"Tensor shape from bvh file: {data.shape}")

    '''
    torch -> bvh
    '''
    logging = LoggingMixin()

    # # * Original LDA: load data pipeline for LDA character
    # # bvh_data = logging.feats_to_bvh(data, hparams)

    # * New data pipeline for YBot
    # n_feats = data.shape[2]  
    # print(n_feats)
      
    # data_pipeline["root"].separate_root=False
    bvh_data=data_pipeline.inverse_transform(data)
    logging.write_bvh(bvh_data, ".", "test_mirror")
    
    print(f"Tensor shape from bvh file: {data.shape}")