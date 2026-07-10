"""
analysis/extract_acoustic_targets.py — per-time-slice acoustic features aligned
row-for-row to the pooled EAT time-slices from pool_patches_to_timeslices.py.

This step ONLY extracts targets. No probing / regression.

======================================================================
Alignment (load-bearing)
======================================================================
pool_patches_to_timeslices.py produced activations/bullfinch_timeslices_pooled.npz:
2365 rows, one per 160 ms full-spectrum EAT time-slice, with a parallel index
(slice_rec_idx, slice_time_step, slice_t_start_s, rec_names) and per-clip
valid_time. Here we compute acoustic features over EXACTLY the same window,
in EXACTLY the same order, so Y[i] describes the audio behind pooled row i.

We reproduce the model's audio, not the raw file:
  * Same loader as the cache: data.loader.load_audio_file resamples every clip
    to TARGET_SAMPLE_RATE (16 kHz) mono — this is the identical audio-prep that
    fed the avex activation cache (bullfinch_within_layer_cluster.py ->
    extract_all_clips -> load_audio_file). We reuse it; we do not reimplement.
  * Same windowing as the pooling step: the cache fills up to the full 1024
    fbank frames (~10.24 s) from the clip start and zero-pads the tail (verified
    last step against the layer-0 padding boundary). So we take the 16 kHz
    waveform from sample 0 and pad/truncate to exactly valid_time * 2560 samples
    (2560 = 0.160 s * 16 kHz), giving valid_time non-overlapping 160 ms windows.
    Slice time_step=t therefore maps to samples [t*2560, (t+1)*2560), i.e.
    [t*0.160, (t+1)*0.160] s — matching slice_t_start_s in the pooled file.

Per clip we assert n_slices == valid_time (from the pooled file) AND that the
number of 160 ms windows we can cut equals it; we error on any mismatch. We
also assert the (rec_idx, time_step) order we build equals the pooled index
element-for-element, so the row alignment is guaranteed, not assumed.

======================================================================
Features per slice (K = 9), all over the 160 ms window via librosa
======================================================================
Framing: within each clip we run librosa on the processed waveform with
hop_length = HOP (256) and center=True, which yields exactly
SLICE_FRAMES = 2560 / 256 = 10 finer frames per 160 ms slice. We aggregate the
10 frames of each slice into one representative value:
  * mean  for rms_energy, spectral_centroid, spectral_bandwidth,
    spectral_rolloff, zero_crossing_rate, spectral_flatness, voiced_prob;
  * nanmean for f0 (NaN if the whole slice is unvoiced);
  * voiced_flag = (mean of pyin's per-frame voiced flags >= 0.5).

F0 / voicing is kept honest: librosa.pyin returns NaN on unvoiced/silent
frames. We do NOT fill — f0 is saved with NaNs preserved, voiced_mask records
which slices are voiced, and the voiced fraction is reported. All other
features are defined on every slice. On a voiced slice (voiced_mask True, i.e.
>= 5 of 10 frames voiced) f0 is always finite; a handful of minority-voiced
slices carry a finite f0 but voiced_mask False — the probe should select on
voiced_mask.

CAVEAT on voiced_prob: pyin's per-frame voiced_probability is the probability
mass on voiced pitch states, which is small in ABSOLUTE terms even for
confidently voiced frames (voiced frames here average ~0.18, unvoiced ~0.017).
It is a soft confidence on a compressed scale, NOT a calibrated P(voiced) to
threshold at 0.5. Use voiced_flag / voiced_mask (the pyin Viterbi decision) for
voicing; treat voiced_prob only as a relative confidence.

CPU-only, seed 42. Reuses constants + audio-prep from the collect script /
data.loader. Follows CLAUDE.md.

Usage:
    python -W ignore analysis/extract_acoustic_targets.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa  # noqa: E402

# Reuse constants + audio-prep — do not hardcode / reimplement.
from collect_esp_aves2_activations import PATCH_SIZE, TARGET_SAMPLE_RATE  # noqa: E402
from bullfinch_within_layer_cluster import AUDIO_DIR, collect_clips  # noqa: E402
from data.loader import load_audio_file  # noqa: E402

SEED = 42

# --- slice / framing geometry (derived, see header) ------------------------
FBANK_FRAME_SHIFT_S = 0.010                              # collect fbank frame_shift=10 ms
SECONDS_PER_TIME_PATCH = PATCH_SIZE * FBANK_FRAME_SHIFT_S  # 16 * 0.010 = 0.160 s
SAMPLES_PER_SLICE = int(round(SECONDS_PER_TIME_PATCH * TARGET_SAMPLE_RATE))  # 2560
HOP = 256                                                # librosa hop
assert SAMPLES_PER_SLICE % HOP == 0, "hop must divide the slice length"
SLICE_FRAMES = SAMPLES_PER_SLICE // HOP                  # 10 finer frames per slice

# librosa windowing
N_FFT = 512          # spectral / time-domain frame length
PYIN_FRAME_LENGTH = 2048
F0_FMIN = 200.0      # bullfinch song fundamentals sit ~1-3 kHz; 200 Hz floor is safe
F0_FMAX = 6000.0

FEATURE_NAMES = [
    "rms_energy",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "zero_crossing_rate",
    "spectral_flatness",
    "f0",
    "voiced_flag",
    "voiced_prob",
]
F0_COL = FEATURE_NAMES.index("f0")

POOLED_NPZ = Path("activations/bullfinch_timeslices_pooled.npz")
OUT_NPZ = Path("activations/bullfinch_acoustic_targets.npz")
OUT_CSV = Path("activations/bullfinch_acoustic_targets.csv")


def processed_waveform(path: Path, valid_time: int) -> np.ndarray:
    """16 kHz mono waveform, from sample 0, padded/truncated to the model's span.

    Reproduces the cache's audio: valid_time * 2560 samples (clip start, zero-pad
    the tail exactly as the model padded the mel). Reuses data.loader.load_audio_file.
    """
    wav = load_audio_file(str(path)).squeeze(0).cpu().numpy().astype(np.float32)
    target = valid_time * SAMPLES_PER_SLICE
    if wav.shape[0] < target:
        wav = np.pad(wav, (0, target - wav.shape[0]))
    else:
        wav = wav[:target]
    assert wav.shape[0] == target
    assert wav.shape[0] // SAMPLES_PER_SLICE == valid_time, (
        f"{path.name}: cut {wav.shape[0] // SAMPLES_PER_SLICE} windows != valid_time {valid_time}"
    )
    return wav


def _by_slice(frame_vals: np.ndarray, valid_time: int) -> np.ndarray:
    """Truncate a per-frame array to valid_time*SLICE_FRAMES and reshape to
    (valid_time, SLICE_FRAMES) so each row holds one slice's finer frames."""
    n = valid_time * SLICE_FRAMES
    if frame_vals.shape[0] < n:
        raise RuntimeError(f"expected >= {n} frames, got {frame_vals.shape[0]}")
    return frame_vals[:n].reshape(valid_time, SLICE_FRAMES)


def clip_features(wav: np.ndarray, valid_time: int) -> dict[str, np.ndarray]:
    """Per-slice features for one processed clip. Returns {name: (valid_time,)}."""
    sr = TARGET_SAMPLE_RATE
    rms = librosa.feature.rms(y=wav, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    cent = librosa.feature.spectral_centroid(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    bw = librosa.feature.spectral_bandwidth(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    roll = librosa.feature.spectral_rolloff(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    flat = librosa.feature.spectral_flatness(y=wav, n_fft=N_FFT, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(wav, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    f0, vflag, vprob = librosa.pyin(
        wav, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr,
        frame_length=PYIN_FRAME_LENGTH, hop_length=HOP, center=True,
    )
    vflag = vflag.astype(np.float32)

    out = {
        "rms_energy": _by_slice(rms, valid_time).mean(axis=1),
        "spectral_centroid": _by_slice(cent, valid_time).mean(axis=1),
        "spectral_bandwidth": _by_slice(bw, valid_time).mean(axis=1),
        "spectral_rolloff": _by_slice(roll, valid_time).mean(axis=1),
        "zero_crossing_rate": _by_slice(zcr, valid_time).mean(axis=1),
        "spectral_flatness": _by_slice(flat, valid_time).mean(axis=1),
        "f0": np.nanmean(_by_slice(f0, valid_time), axis=1),  # NaN if slice fully unvoiced
        "voiced_flag": (_by_slice(vflag, valid_time).mean(axis=1) >= 0.5).astype(np.float32),
        "voiced_prob": _by_slice(vprob, valid_time).mean(axis=1),
    }
    return out


def main() -> None:
    np.random.seed(SEED)  # convention; pyin is deterministic

    if not POOLED_NPZ.exists():
        raise FileNotFoundError(f"{POOLED_NPZ} not found — run pool_patches_to_timeslices.py first.")
    pooled = np.load(POOLED_NPZ, allow_pickle=True)
    slice_rec_idx = pooled["slice_rec_idx"].astype(np.int64)
    slice_time_step = pooled["slice_time_step"].astype(np.int64)
    slice_t_start_s = pooled["slice_t_start_s"].astype(np.float64)
    rec_names = [str(n) for n in pooled["rec_names"]]
    valid_time = pooled["valid_time"].astype(np.int64)
    n_rows = slice_rec_idx.shape[0]
    n_recs = len(rec_names)
    print(f"[pooled] {POOLED_NPZ}  rows={n_rows}  recordings={n_recs}")
    print(f"[grid] slice=160 ms={SAMPLES_PER_SLICE} samples @16kHz; "
          f"hop={HOP} -> {SLICE_FRAMES} librosa frames/slice (mean-aggregated)")

    # Reconstruct the exact clip order behind the cache (rec_idx i == clip i).
    clip_paths = collect_clips(AUDIO_DIR)
    if [p.stem for p in clip_paths] != rec_names:
        raise RuntimeError("collect_clips() order does not match pooled rec_names.")

    # --- extract per clip, in pooled row order --------------------------------
    per_feat: dict[str, list[np.ndarray]] = {k: [] for k in FEATURE_NAMES}
    built_rec_idx: list[np.ndarray] = []
    built_time_step: list[np.ndarray] = []
    print("\n[extract]")
    for i, path in enumerate(clip_paths):
        v = int(valid_time[i])
        wav = processed_waveform(path, v)
        feats = clip_features(wav, v)
        for k in FEATURE_NAMES:
            if feats[k].shape[0] != v:
                raise RuntimeError(f"{path.name}: feature {k} has {feats[k].shape[0]} slices != valid_time {v}")
            per_feat[k].append(feats[k])
        built_rec_idx.append(np.full(v, i, dtype=np.int64))
        built_time_step.append(np.arange(v, dtype=np.int64))
        vf = float(feats["voiced_flag"].mean())
        print(f"  [{i + 1:2d}/{n_recs}] {rec_names[i][:44]:<44s} slices={v:2d}  voiced_frac={vf:.2f}")

    # --- assemble Y in pooled row order + verify alignment --------------------
    Y = np.column_stack([np.concatenate(per_feat[k]) for k in FEATURE_NAMES]).astype(np.float32)
    built_rec_idx_arr = np.concatenate(built_rec_idx)
    built_time_step_arr = np.concatenate(built_time_step)

    if Y.shape[0] != n_rows:
        raise RuntimeError(f"Y has {Y.shape[0]} rows != pooled {n_rows}")
    if not np.array_equal(built_rec_idx_arr, slice_rec_idx):
        raise RuntimeError("row order (rec_idx) does not match the pooled index.")
    if not np.array_equal(built_time_step_arr, slice_time_step):
        raise RuntimeError("row order (time_step) does not match the pooled index.")
    if not np.allclose(slice_t_start_s, slice_time_step * SECONDS_PER_TIME_PATCH):
        raise RuntimeError("pooled t_start_s disagrees with time_step * 0.160.")

    voiced_mask = Y[:, FEATURE_NAMES.index("voiced_flag")] >= 0.5

    # --- save (gitignored activations/) ---------------------------------------
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        Y=Y,
        feature_names=np.array(FEATURE_NAMES),
        voiced_mask=voiced_mask,
        slice_rec_idx=slice_rec_idx.astype(np.int32),
        slice_time_step=slice_time_step.astype(np.int32),
        slice_t_start_s=slice_t_start_s.astype(np.float32),
        rec_names=np.array(rec_names),
    )
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording_id", "time_step", "t_start_seconds", *FEATURE_NAMES])
        for j in range(n_rows):
            w.writerow([
                rec_names[slice_rec_idx[j]],
                int(slice_time_step[j]),
                f"{slice_t_start_s[j]:.3f}",
                *[("" if np.isnan(Y[j, c]) else f"{Y[j, c]:.5g}") for c in range(len(FEATURE_NAMES))],
            ])

    # --- sanity checks --------------------------------------------------------
    print("\n[sanity]")
    print(f"  row count == 2365: {Y.shape[0]} -> {Y.shape[0] == 2365}")
    print(f"  Y shape: {Y.shape}  (K={len(FEATURE_NAMES)})")
    print("  per-feature NaN counts (should be 0 except f0):")
    for c, name in enumerate(FEATURE_NAMES):
        print(f"    {name:<20s} nan={int(np.isnan(Y[:, c]).sum())}")
    print(f"  voiced fraction: {voiced_mask.mean():.3f} "
          f"({int(voiced_mask.sum())}/{n_rows} slices voiced)")
    print("  per-feature min / mean / max (f0 over voiced slices only):")
    for c, name in enumerate(FEATURE_NAMES):
        col = Y[:, c]
        finite = col[np.isfinite(col)]
        print(f"    {name:<20s} min={finite.min():.4g}  mean={finite.mean():.4g}  max={finite.max():.4g}")

    # sane-range asserts (not exhaustive, catch gross alignment/units errors)
    def col(name):
        return Y[:, FEATURE_NAMES.index(name)]
    assert (col("rms_energy") >= 0).all(), "rms must be >= 0"
    assert (col("zero_crossing_rate") >= 0).all() and (col("zero_crossing_rate") <= 1).all()
    assert np.nanmax(col("spectral_centroid")) < TARGET_SAMPLE_RATE / 2, "centroid must be < Nyquist"
    assert np.nanmax(col("spectral_rolloff")) <= TARGET_SAMPLE_RATE / 2
    f0v = col("f0")[voiced_mask]
    assert np.all((f0v >= F0_FMIN - 1) & (f0v <= F0_FMAX + 1)), "voiced f0 out of [fmin,fmax]"
    print("  range asserts: passed")

    # --- eyeball one example clip+slice ---------------------------------------
    ex_rec, ex_t = 0, 5
    j = int(np.where((slice_rec_idx == ex_rec) & (slice_time_step == ex_t))[0][0])
    t0 = ex_t * SECONDS_PER_TIME_PATCH
    print(f"\n[example] clip '{rec_names[ex_rec][:40]}' slice time_step={ex_t}  "
          f"window [{t0:.3f}, {t0 + SECONDS_PER_TIME_PATCH:.3f}] s  "
          f"(samples [{ex_t * SAMPLES_PER_SLICE}, {(ex_t + 1) * SAMPLES_PER_SLICE}))")
    for c, name in enumerate(FEATURE_NAMES):
        val = Y[j, c]
        print(f"    {name:<20s} {'NaN (unvoiced)' if np.isnan(val) else f'{val:.5g}'}")

    print(f"\n[saved] {OUT_NPZ}")
    print(f"[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
