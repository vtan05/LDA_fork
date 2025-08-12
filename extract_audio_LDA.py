import os
import pandas as pd
import numpy as np
import madmom
import pickle

from scipy.io.wavfile import write


def extract(sig):
    stft = madmom.audio.stft.STFT(sig)
    spec = madmom.audio.spectrogram.Spectrogram(stft)

    sf = madmom.features.onsets.spectral_flux(spec)
    sf_feat = np.mean(sf, axis=0).reshape(1, 1)

    dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(sig)
    chroma = chroma.reshape(1, 12)
    chroma_feat = np.mean(chroma, axis=0)[0].reshape(1, 1) 

    proc = madmom.features.downbeats.RNNDownBeatProcessor()
    beat = proc(sig)
    beat_feat = np.mean(beat, axis=0)[0].reshape(1, 1)

    features = [sf_feat, chroma_feat, beat_feat]
    features_array = np.concatenate(features, axis=1)

    return features_array


def process_wav_files(directory, out_directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            wav_path = os.path.join(directory, filename)
            print(f"Processing {wav_path}")

            sig = madmom.audio.signal.Signal(wav_path, num_channels=1, sample_rate=44100)

            fs_30 = madmom.audio.signal.FramedSignal(sig, fps=30)

            feat = np.array([])
            for i in range(0, len(fs_30)):
                features = extract(fs_30[i])
                feat = np.concatenate([feat, features], axis=0) if feat.size > 1 else features
            print(feat.shape)

            dataframe = pd.DataFrame(feat, columns=[
                'Chroma_0', 'Spectralflux_0', 'Beatactivation_0'
            ])
            dataframe.replace([np.inf, -np.inf], 0, inplace=True)
            dataframe = dataframe.fillna(dataframe.mean())
            dataframe.index = pd.Series([pd.Timedelta(seconds=(1/sig.sample_rate) * i) for i in range(len(dataframe.index))])

            output_file = os.path.join(out_directory, f"{os.path.splitext(filename)[0]}.audio29_30fps.pkl")
            with open(output_file, 'wb') as file:
                pickle.dump(dataframe, file)
            
            output_file = os.path.join(out_directory, f"{os.path.splitext(filename)[0]}_mirrored.audio29_30fps.pkl")
            with open(output_file, 'wb') as file:
                pickle.dump(dataframe, file)


if __name__ == '__main__':

    directory = r"/host_data/van/LDA/data/motorica/wav" # Replace with the path to your directory containing WAV files
    out_directory = r"/host_data/van/LDA/data/motorica/feat"

    process_wav_files(directory, out_directory)
