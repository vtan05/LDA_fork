import os
import shutil

# === Set your directories ===
wav_dir = "/host_data/van/LDA/data/finedance/music_wav"
bvh_dir = "/host_data/van/LDA/data/finedance/ybot_bvh"
unpaired_wav_dir = "/host_data/van/LDA/data/finedance/unpaired_wav"
unpaired_bvh_dir = "/host_data/van/LDA/data/finedance/unpaired_bvh"

# === Create unpaired folders if they don't exist ===
os.makedirs(unpaired_wav_dir, exist_ok=True)
os.makedirs(unpaired_bvh_dir, exist_ok=True)

# === Extract IDs from filenames ===
wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith(".bvh")]

wav_ids = set(os.path.splitext(f)[0] for f in wav_files)
bvh_ids = set(os.path.splitext(f)[0].split("_")[-1] for f in bvh_files)

# === Find unpaired files ===
unpaired_wavs = [f for f in wav_files if f.split('.')[0] not in bvh_ids]
unpaired_bvhs = [f for f in bvh_files if f.split("_")[-1].split(".")[0] not in wav_ids]

# === Move unpaired wavs ===
for fname in unpaired_wavs:
    src = os.path.join(wav_dir, fname)
    dst = os.path.join(unpaired_wav_dir, fname)
    shutil.move(src, dst)
    print(f"Moved unpaired wav: {fname}")

# === Move unpaired bvhs ===
for fname in unpaired_bvhs:
    src = os.path.join(bvh_dir, fname)
    dst = os.path.join(unpaired_bvh_dir, fname)
    shutil.move(src, dst)
    print(f"Moved unpaired bvh: {fname}")
