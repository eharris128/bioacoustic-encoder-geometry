"""Quick timing test: stream NatureLM samples and extract AVES activations."""

import time
import numpy as np
import torch
from datasets import load_dataset

from aves import load_feature_extractor

# --- Config ---
CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
N_SAMPLES = 5  # just a few for timing

# --- Load model ---
print("Loading AVES model...")
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)

# --- Stream dataset ---
print(f"\nStreaming {N_SAMPLES} samples from NatureLM-audio-training...")
t_stream_start = time.time()
ds = load_dataset(
    "EarthSpeciesProject/NatureLM-audio-training",
    split="train",
    streaming=True,
)

samples = []
for i, item in enumerate(ds):
    if i >= N_SAMPLES:
        break
    samples.append(item)
    elapsed = time.time() - t_stream_start
    audio = item["audio"]
    sr = audio["sampling_rate"]
    duration = len(audio["array"]) / sr
    print(f"  Sample {i}: sr={sr}, duration={duration:.1f}s, "
          f"source={item.get('source_dataset', '?')}, "
          f"task={item.get('task', '?')}")
    print(f"    metadata keys: {list(item.keys())}")

t_stream = time.time() - t_stream_start
print(f"\nStreaming {N_SAMPLES} samples took {t_stream:.1f}s")

# --- Extract activations ---
print(f"\nExtracting activations...")
for i, item in enumerate(samples):
    audio_array = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]

    # Convert to tensor
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

    # Resample if needed
    if sr != 16000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)

    print(f"  Sample {i}: {len(audio_tensor)} samples ({len(audio_tensor)/16000:.1f}s at 16kHz)")

    t0 = time.time()
    layer_outputs = model.extract_features(audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor, layers=None)
    t_extract = time.time() - t0

    # Mean-pool to get one vector per layer
    mean_pooled = torch.stack([layer.squeeze(0).mean(dim=0) for layer in layer_outputs])
    print(f"    Extraction: {t_extract:.2f}s, shape per layer: {layer_outputs[0].shape}, "
          f"mean-pooled: {mean_pooled.shape}")

print(f"\nDone. Total wall time: {time.time() - t_stream_start:.1f}s")
