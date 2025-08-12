import os
import torch
import torchaudio
import torchaudio.transforms as T
import pickle
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from torch import nn

audio_folder = "/host_data/van/LDA/data/edge_aistpp/wavs"
output_folder = "/host_data/van/music_feat/aist/mert"
os.makedirs(output_folder, exist_ok=True)

# Load MERT‑v1‑330M
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
model.eval()

# Aggregator across layers
class MERTLayerAggregator(nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))
    def forward(self, hs):
        w = torch.softmax(self.weights, dim=0)  # [24]
        return torch.einsum("ltd,l->td", hs, w)  # [T, 1024]

aggregator = MERTLayerAggregator()
aggregator.eval()

motion_fps = 60
audio_sr = processor.sampling_rate  # 24000 Hz
mert_fps = 75.0  # feature rate specified for MERT‑v1 models :contentReference[oaicite:11]{index=11}
resample_ratio = motion_fps / mert_fps

for fname in tqdm(sorted(os.listdir(audio_folder))):
    if not fname.lower().endswith(".wav"):
        continue
    wav, sr = torchaudio.load(os.path.join(audio_folder, fname))
    wav = wav[0]
    if sr != audio_sr:
        wav = T.Resample(orig_freq=sr, new_freq=audio_sr)(wav)

    inputs = processor(wav.numpy(), sampling_rate=audio_sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        all_layers = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # [L, T, D]
        num_layers = all_layers.shape[0]
        aggregator = MERTLayerAggregator(num_layers=num_layers).eval()
        emb = aggregator(all_layers)  # [T, D]

    emb_np = emb.cpu().numpy()
    emb_resampled = zoom(emb_np, (resample_ratio, 1))  # [T_motion, 1024]

    out_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(emb_resampled, f)
    print(f"Saved {out_path} shape={emb_resampled.shape}")
