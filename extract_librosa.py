import os
import pandas as pd
import numpy as np
import madmom
import pickle

# --------- Feature extraction for one (framed) audio segment ----------
def extract_per_frame(sig_frame, sr=44100):
    """
    sig_frame: either madmom.audio.signal.Signal or a 1D numpy array
    Returns [Beat_0, Chroma_0, Chroma_1, MFCC_1, MFCC_3, MFCC_6, Beatactivation_0, Spectralflux_0]
    as a (1, 8) numpy array.
    """
    # Normalize input to a madmom Signal
    if isinstance(sig_frame, madmom.audio.signal.Signal):
        s = sig_frame
    else:
        s = madmom.audio.signal.Signal(sig_frame, num_channels=1, sample_rate=sr)

    # Safeguards (short frames can cause processors to fail)
    def safe_mean(x, default=0.0):
        try:
            x = np.asarray(x)
            if x.size == 0:
                return default
            return float(np.mean(x))
        except Exception:
            return default

    # --- Spectrogram for MFCC + spectral flux ---
    try:
        stft = madmom.audio.stft.STFT(s)                         # complex STFT
        spec = madmom.audio.spectrogram.Spectrogram(stft)        # magnitude
    except Exception:
        spec = None

    # MFCCs (we’ll pick coefficients 1,3,6; skip 0 which is loudness-like)
    mfcc_1 = mfcc_3 = mfcc_6 = 0.0
    if spec is not None:
        try:
            from cepstrogram import MFCC
            mfcc = MFCC(spec, num_bands=20)                      # [T, 20]
            mfcc_mean = np.mean(mfcc, axis=0)                    # [20]
            if mfcc_mean.shape[0] >= 7:
                mfcc_1 = float(mfcc_mean[1])
                mfcc_3 = float(mfcc_mean[3])
                mfcc_6 = float(mfcc_mean[6])
        except Exception:
            pass

    # Spectral flux (continuous onset proxy)
    spectral_flux = 0.0
    if spec is not None:
        try:
            sf = madmom.features.onsets.spectral_flux(spec)      # [T]
            spectral_flux = safe_mean(sf, 0.0)
        except Exception:
            pass

    # Deep chroma (take first two bins)
    chroma_0 = chroma_1 = 0.0
    try:
        dcp = madmom.audio.chroma.DeepChromaProcessor()
        chroma = dcp(s)                                          # [T, 12]
        if chroma.ndim == 2 and chroma.shape[1] >= 2:
            chroma_mean = np.mean(chroma, axis=0)                # [12]
            chroma_0 = float(chroma_mean[0])
            chroma_1 = float(chroma_mean[1])
    except Exception:
        pass

    # Beat / downbeat activations from RNNDownBeatProcessor
    # We’ll map:
    #   Beatactivation_0 := mean of column 0
    #   Beat_0           := mean of column 1
    beatactivation_0 = 0.0
    beat_0 = 0.0
    try:
        proc = madmom.features.downbeats.RNNDownBeatProcessor()
        beat_mat = proc(s)                                       # [T, 2]
        if beat_mat.ndim == 2 and beat_mat.shape[1] == 2:
            beatactivation_0 = safe_mean(beat_mat[:, 0], 0.0)
            beat_0 = safe_mean(beat_mat[:, 1], 0.0)
    except Exception:
        pass

    # Order exactly as requested
    feat = np.array([[beat_0, chroma_0, chroma_1,
                      mfcc_1, mfcc_3, mfcc_6,
                      beatactivation_0, spectral_flux]], dtype=np.float32)
    return feat


# --------- Batch processing of WAVs at 60 fps ----------
def process_wav_files(directory, out_directory, sr=44100, fps=60):
    os.makedirs(out_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if not filename.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(directory, filename)
        print(f"Processing {wav_path}")

        # Load mono signal at fixed sample rate
        sig = madmom.audio.signal.Signal(wav_path, num_channels=1, sample_rate=sr)

        # Frame the signal at target fps (≈ hop = sr/fps)
        fs = madmom.audio.signal.FramedSignal(sig, fps=fps)

        # Extract per-frame features
        feats = []
        for i in range(len(fs)):
            frame = fs[i]  # ndarray for the i-th frame
            feat = extract_per_frame(frame, sr=sr)  # (1, 8)
            feats.append(feat)

        feat = np.vstack(feats) if len(feats) else np.zeros((0, 8), dtype=np.float32)
        print("Feature shape:", feat.shape)

        # Build DataFrame
        cols = [
            'Beat_0',
            'Chroma_0', 'Chroma_1',
            'MFCC_1', 'MFCC_3', 'MFCC_6',
            'Beatactivation_0',
            'Spectralflux_0'
        ]
        df = pd.DataFrame(feat, columns=cols)

        # Cleanups
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.fillna(0)

        # 60 fps time index
        times = np.arange(len(df)) / float(fps)
        df.index = pd.to_timedelta(times, unit='s')

        # Save (both names like your original script)
        stem = os.path.splitext(filename)[0]
        out1 = os.path.join(out_directory, f"{stem}.audio8_60fps.pkl")
        out2 = os.path.join(out_directory, f"{stem}_mirrored.audio8_60fps.pkl")
        for out_path in (out1, out2):
            with open(out_path, 'wb') as f:
                pickle.dump(df, f)
        print(f"Saved: {out1} and {out2}")


if __name__ == '__main__':
    directory = r"/host_data/van/LDA/data/edge_aistpp/wavs"
    out_directory = r"/host_data/van/music_feat/aist/librosa"
    process_wav_files(directory, out_directory)
