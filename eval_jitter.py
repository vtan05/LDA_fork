import os

# import torch
import numpy as np
from tqdm import tqdm
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline

def is_bvh(fname: str) -> bool:
    if fname[-4:] == ".bvh":
        return True
    return False

def is_npy(fname: str) -> bool:
    if fname[-4:] == ".npy":
        return True
    return False

def list_bvh_in_dir(dir):
    '''
    List all files in dir.

    Args:
        dir: directory to retreive file names from
    '''

    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(f"{dir}/{f}") and is_bvh(f)]
    files.sort()
    return files

def list_npy_in_dir(dir):
    '''
    List all files in dir.

    Args:
        dir: directory to retreive file names from
    '''

    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(f"{dir}/{f}") and is_npy(f)]
    files.sort()
    return files

def measure_jitter(joint_pos, fps):
    jitter = (joint_pos[3:] - 3 * joint_pos[2:-1] + 3 * joint_pos[1:-2] - joint_pos[:-3]) * (fps ** 3) 
    jitter = np.linalg.norm(jitter, axis=2) # [297, 19]
    jitter = jitter.mean() 
    return jitter

def measure_jitter_npy(dir:str, fps:int):
    print('Computing jitter metric:')
    print(f' - {dir}')
    file_list = list_npy_in_dir(dir)
    total_jitter = np.zeros([len(file_list)]) # one jitter metric for one motion data

    jitter_bar = tqdm(range(len(file_list)))
    for i in jitter_bar:
        fname = file_list[i]
        full_data_dir = f"{dir}/{fname}"

        joint_pos = np.load(full_data_dir)
        joint_pos = joint_pos * 0.01 # Convert to meters
        jitter = measure_jitter(joint_pos, fps)
        total_jitter[i] = jitter

    jitter_mean = total_jitter.mean()
    print(f"Total mean of jitter of {len(file_list)} motions: {jitter_mean}")

def measure_jitter_bvh(data_pipe, dir:str, fps:int):
    print('Computing jitter metric:')
    print(f' - {dir}')
    file_list = list_bvh_in_dir(data_dir)
    total_jitter = np.zeros([len(file_list)]) # one jitter metric for one motion data

    jitter_bar = tqdm(range(len(file_list)))
    for i in jitter_bar:
        fname = file_list[i]
        full_data_dir = f"{data_dir}/{fname}"
        parser = BVHParser()
        parsed_data = parser.parse(full_data_dir)

        piped_data = data_pipe.fit_transform([parsed_data])
        piped_data = piped_data.squeeze() 
        nof = piped_data.shape[0]
        joint_pos = np.reshape(piped_data, [nof, -1, 3])
        joint_pos = joint_pos * 0.01 # Convert to meters

        jitter = measure_jitter(joint_pos, fps)
        total_jitter[i] = jitter
    jitter_mean = total_jitter.mean()
    
    # ! If there is an outlier, exclude manually
    # max_index = np.argmax(total_jitter) # 111, val: 168787429.9756961
    # print(max_index)
    # print(total_jitter[max_index])
    # total_jitter_deleted = np.delete(total_jitter, [77, 78])
    # jitter_mean = total_jitter_deleted.mean()
        
    print(f"Total mean of jitter of {len(file_list)} motions: {jitter_mean}")


if __name__ == "__main__":
    ''' 
    Preprocess setup
    '''
    # fps = 30 # for motorica
    fps = 24 # for finedance
    joints = ['Hips', 'Spine','Spine1','Neck','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps)),
        #('mir', Mirror(axis='X', append=True)),
        #('rev', ReverseTime(append=True)),
        ('jtsel', JointSelector(joints, include_root=False)),
        # ('root', RootTransformer('pos_rot_deltas', position_smoothing=4, rotation_smoothing=4)),
        ('exp', MocapParameterizer('position')), 
        # ('cnst', ConstantsRemover()),
        ('npf', Numpyfier())
    ])

    '''
    Compute jitter metric for all motions and get mean
    '''
    data_dir = "/host_data/van/LDA/results/finedance"
    # bvh
    measure_jitter_bvh(data_pipe, data_dir, fps)
    
    # npy
    # measure_jitter_npy(data_dir, fps)
    