import librosa
import numpy
import sklearn
import matplotlib
import torch
import load_feature_extractor

# constants
AUDIO_PATH = "audio/zebra_finch/your_recording.wav"
LAYER_TARGET = 11
N_NOISE_LEVELS = 10
SNR_MIN = 5      # heavily degraded
SNR_MAX = 40     # near clean

# Audio files: {key: (filepath, label)}  0 = bullfinch, 1 = hawfinch
AUDIO_FILES = {
    "bullfinch_XC1077468": ("audio/bullfinch/XC1077468 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",           0),
    "bullfinch_XC965743":  ("audio/bullfinch/XC965743 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",            0),
    "bullfinch_XC938052":  ("audio/bullfinch/XC938052 - Eurasian Bullfinch - Pyrrhula pyrrhula.mp3",            0),
    "bullfinch_XC805629":  ("audio/bullfinch/XC805629 - Eurasian Bullfinch - Pyrrhula pyrrhula rosacea.mp3",    0),
    "hawfinch_XC944735":   ("audio/hawfinch/XC944735 - Hawfinch - Coccothraustes coccothraustes.wav",           1),
    "hawfinch_XC1087947":  ("audio/hawfinch/XC1087947 - Hawfinch - Coccothraustes coccothraustes.mp3",         1),
    "hawfinch_XC1086752":  ("audio/hawfinch/XC1086752 - Hawfinch - Coccothraustes coccothraustes.wav",         1),
    "hawfinch_XC1084204":  ("audio/hawfinch/XC1084204 - Hawfinch - Coccothraustes coccothraustes.wav",         1),
    "hawfinch_XC1083076":  ("audio/hawfinch/XC1083076 - Hawfinch - Coccothraustes coccothraustes.mp3",         1),
}

def load_audio_files(audio_dict, sr=22050):
    """Load all audio files and return list of (waveform, sample_rate, label) tuples."""
    loaded = {}
    for key, (path, label) in audio_dict.items():
        waveform, sample_rate = librosa.load(path, sr=sr, mono=True)
        loaded[key] = (waveform, sample_rate, label)
        print(f"Loaded {key}: {waveform.shape}, sr={sample_rate}, label={label}")
    return loaded

# Example usage:
# data = load_audio_files(AUDIO_FILES)
