import os
import torch
import librosa
import pickle
import numpy as np
from tqdm import tqdm
from muq import MuQ

# === Configuration ===
input_dir = "/host_data/van/LDA/data/edge_aistpp/wavs"
output_dir = "/host_data/van/music_feat/aist/muq"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_sr = 24000           # MuQ expects 24 kHz mono
muq_fps = 25.0             # MuQ outputs ~25 Hz (40 ms)
motion_fps = 60.0          # Your motion model's frame rate
resample_embeddings = True # Set False to keep MuQ's native 25 Hz

# === Load MuQ ===
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

def resample_time_series(arr, src_hz, dst_hz):
    """
    arr: [T, D] numpy array
    Returns resampled [T', D] using per-dim linear interpolation.
    """
    T, D = arr.shape
    T_dst = int(round(T * dst_hz / src_hz))
    if T <= 1 or T_dst <= 1 or T == T_dst:
        return arr
    x_src = np.linspace(0.0, 1.0, T, endpoint=True)
    x_dst = np.linspace(0.0, 1.0, T_dst, endpoint=True)
    out = np.empty((T_dst, D), dtype=arr.dtype)
    for d in range(D):
        out[:, d] = np.interp(x_dst, x_src, arr[:, d])
    return out

# === Process All WAV Files ===
for fname in tqdm(sorted(os.listdir(input_dir))):
    if not fname.lower().endswith(".wav"):
        continue

    path = os.path.join(input_dir, fname)
    try:
        # Load mono at 24 kHz
        wav, sr = librosa.load(path, sr=audio_sr, mono=True)
        wavs = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # [1, T]

        # Extract MuQ last hidden layer
        with torch.no_grad():
            output = muq(wavs, output_hidden_states=True)
            last = output.last_hidden_state  # [B, T, D]
        print(f"{fname} | Total number of layers: {len(output.hidden_states)}")
        print(f"{fname} | Feature shape (last layer): {tuple(last.shape)}")

        # Remove batch dim -> [T, D]
        emb_np = last.squeeze(0).detach().cpu().numpy()

        # Optional resample to motion FPS
        if resample_embeddings:
            emb_np = resample_time_series(emb_np, src_hz=muq_fps, dst_hz=motion_fps)

        # Save as .pkl
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".pkl")
        with open(out_path, "wb") as f:
            pickle.dump(emb_np, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved: {out_path} | Shape: {emb_np.shape}")

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
