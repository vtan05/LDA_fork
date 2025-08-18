import os
import numpy as np
import librosa

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from pathlib import Path
import re


def ba_score(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


def cal_motion_beat(motion_file_path):
    parser = BVHParser()
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=24)),
        ('jtsel', JointSelector(['Spine','Spine1','Spine2','Neck','Head',
                'RightUpLeg','RightLeg','RightFoot',
                'LeftUpLeg','LeftLeg', 'LeftFoot',
                'RightShoulder','RightArm','RightForeArm','RightHand',
                'LeftShoulder','LeftArm','LeftForeArm','LeftHand'], include_root=True)),
        ('root', RootTransformer('pos_rot_deltas', position_smoothing=1, rotation_smoothing=1)),
        ('param', MocapParameterizer('position')),
        ('cnst', ConstantsRemover()),
        ('npf', Numpyfier())
    ])

    parsed_data = parser.parse(motion_file_path)
    joint_pos = data_pipe.fit_transform([parsed_data])
    print(f"Joint positions shape: {joint_pos.shape}")
    joint_pos = np.array(joint_pos).reshape(-1, 20, 3)

    kinetic_vel = np.mean(np.sqrt(np.sum((joint_pos[1:] - joint_pos[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)

    return motion_beats, len(kinetic_vel)


def bvh_to_wav_name(motion_name: str) -> str:
    """
    finedance_..._036_0_ClassicHanTang.bvh
    -> finedance_..._036_0.wav
    (drops last underscore segment if it's letters-only)
    """
    p = Path(motion_name)
    stem = p.stem
    parts = stem.split("_")
    if len(parts) > 1 and re.fullmatch(r"[A-Za-z]+", parts[-1]):  # last segment like 'ClassicHanTang'
        stem = "_".join(parts[:-1])
    return stem + ".wav"


def calc_ba_score(directory):
    ba_scores = []

    motion_files = [f for f in os.listdir(directory) if f.endswith('.bvh')]

    for motion_file in motion_files:
        motion_path = os.path.join(directory, motion_file)
        dance_beats, length = cal_motion_beat(motion_path)

        music_file = bvh_to_wav_name(motion_file)
        music_path = os.path.join(directory, music_file)
        print(f"Processing music file: {music_file}")

        if not os.path.exists(music_path):
            print(f"Corresponding music file {music_file} not found for motion file {motion_file}. Skipping.")
            continue

        FPS = 24
        HOP_LENGTH = 512
        SR = FPS * HOP_LENGTH
        music, _ = librosa.load(music_path, sr=SR)
        envelope = librosa.onset.onset_strength(y=music, sr=SR)  # (seq_len,)
        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
        )
        start_bpm = librosa.beat.tempo(y=music)[0]
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=SR,
            hop_length=HOP_LENGTH,
            start_bpm=start_bpm,
            tightness=100,
        )

        ba_scores.append(ba_score(beat_idxs, dance_beats))

    return np.mean(ba_scores) if ba_scores else 0


if __name__ == '__main__':
    directory = r"/host_data/van/LDA/results/finedance"
    print(calc_ba_score(directory))
