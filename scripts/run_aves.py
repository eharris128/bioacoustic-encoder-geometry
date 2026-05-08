"""Minimal AVES inference script — loads the model and extracts embeddings from an example audio file."""

import time
from aves import load_feature_extractor
from aves.utils import load_audio

CONFIG_PATH = "./aves/config/default_cfg_aves-base-all.json"
MODEL_PATH = "./models/aves-base-all.torchaudio.pt"
AUDIO_PATH = "./aves/example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav"

print("Loading model...")
t0 = time.time()
model = load_feature_extractor(
    config_path=CONFIG_PATH,
    model_path=MODEL_PATH,
    device="cpu",
    for_inference=True,
)
print(f"Model loaded in {time.time() - t0:.1f}s")

print(f"\nLoading audio: {AUDIO_PATH}")
audio = load_audio(AUDIO_PATH, mono=True, mono_avg=False)
print(f"Audio tensor shape: {audio.shape}")

print("\nExtracting embeddings (last layer)...")
t0 = time.time()
features = model.extract_features(audio, layers=-1)
print(f"Done in {time.time() - t0:.1f}s")
print(f"Output shape: {features.shape}")
print(f"  → {features.shape[-2]} frames x {features.shape[-1]}-dim embeddings")

# Also extract from all layers to show what's available for interpretability
print("\nExtracting embeddings from ALL layers...")
t0 = time.time()
all_features = model.extract_features(audio, layers=None)
print(f"Done in {time.time() - t0:.1f}s")
if isinstance(all_features, list):
    print(f"Got {len(all_features)} layers, each shape: {all_features[0].shape}")
else:
    print(f"All-layer output shape: {all_features.shape}")
