import os
import json
import shutil

# === Directory paths ===
json_dir = "/host_data/van/LDA/data/finedance/label_json"
wav_dir = "/host_data/van/LDA/data/finedance/music_wav"
bvh_dir = "/host_data/van/LDA/data/finedance/ybot_bvh"

# === Helper function ===
def get_id_from_filename(fname):
    """Extracts 3-digit ID from filename like '001.wav' or 'ybot_001.bvh'"""
    if fname.endswith(".wav"):
        return os.path.splitext(fname)[0]
    elif fname.endswith(".bvh"):
        return os.path.splitext(fname)[0].split("_")[-1]
    else:
        return None

# === Process each JSON ===
for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue
    
    json_path = os.path.join(json_dir, json_file)
    with open(json_path, "r") as f:
        meta = json.load(f)

    # Get ID (e.g., 001 from 001.json)
    id_num = os.path.splitext(json_file)[0].zfill(3)

    # Format new filename
    new_base = f"finedance_{meta['style1']}{meta['style2']}_sFM_cAll_d02_m{meta['style1']}_ch01_{meta['name']}_{id_num}"

    # Rename WAV
    wav_old = os.path.join(wav_dir, f"{id_num}.wav")
    wav_new = os.path.join(wav_dir, f"{new_base}.wav")
    if os.path.exists(wav_old):
        os.rename(wav_old, wav_new)
        print(f"Renamed WAV: {wav_old} -> {wav_new}")
    else:
        print(f"Missing WAV: {wav_old}")

    # Rename BVH
    bvh_old = os.path.join(bvh_dir, f"ybot_{id_num}.bvh")
    bvh_new = os.path.join(bvh_dir, f"{new_base}.bvh")
    if os.path.exists(bvh_old):
        os.rename(bvh_old, bvh_new)
        print(f"Renamed BVH: {bvh_old} -> {bvh_new}")
    else:
        print(f"Missing BVH: {bvh_old}")
