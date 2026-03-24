import librosa
import numpy
import sklearn
import maplotlib
import torch
import load_feature_extractor 

# constants
AUDIO_PATH = "audio/zebra_finch/your_recording.wav"
LAYER_TARGET = 11
N_NOISE_LEVELS = 10
SNR_MIN = 5      # heavily degraded
SNR_MAX = 40     # near clean
