"""
analysis/pool_patches_to_timeslices.py — pool EAT's 2D patch tokens down to
1D time-slices, for acoustic-feature probing on the Bullfinch clips.

This step ONLY pools. No acoustic features, no probing.

======================================================================
LOAD-BEARING CHECK — patch flatten order (VERIFIED, not assumed)
======================================================================
EAT tokenizes each clip as 513 tokens = 1 CLS + 512 patches, where the 512
patches are a (time x freq) grid = 64 time-patches x 8 freq-patches
(NUM_MELS=128 / PATCH_SIZE=16 = 8 freq bands; TARGET_FBANK_FRAMES=1024 /
PATCH_SIZE=16 = 64 time steps; frame_shift=10ms so each time-patch spans
16 frames = 160 ms).

The 512 patches are flattened TIME-MAJOR: token p (0..511) = t*8 + f, so
reshape(512) -> (64 time, 8 freq).  Confirmed from the EAT source itself
(worstchan/EAT-base_epoch30_pretrain, the base architecture every ESP-AVES2
checkpoint is built on), not from the token count alone:

  * configuration_eat.py: `img_size=(1024, 128)  # (target_length, mel_bins)`.
    So the spectrogram fed to the patch conv is an image of shape
    (B, C=1, H=1024=TIME, W=128=FREQ). collect_esp_aves2_activations.py's
    waveform_to_eat_input builds exactly this: fbank -> (1, frames<=1024, 128)
    -> unsqueeze -> (1, 1, time, freq).

  * model_core.py, PatchEmbed_new.forward:
        x = self.proj(x)               # Conv2d(k=16, stride=16)
        x = x.flatten(2).transpose(1, 2)
    Conv2d over (B,1,1024,128) -> (B, 768, 64, 8) with dim2=64 TIME-patches,
    dim3=8 FREQ-patches. `flatten(2)` is row-major (C-order) over (64, 8), so
    the flat patch index is p = t*8 + f (time is the OUTER/major axis).
    transpose -> (B, 512, 768).

  * eat_model.py, build_sincos_pos_embed / encode: the positional grid is
    `(max_length, W)=(TIME, FREQ)` and the CLS ("extra") token is prepended,
    i.e. CLS sits at token index 0 and patches occupy 1..512.

  => reshape(512) -> (64, 8) is (TIME, FREQ). We mean-pool over axis=1 (the 8
     freq sub-bands) to get one 768-d vector per 160-ms time step.
  If we had assumed freq-major (8, 64) we would pool across TIME instead of
  frequency and silently corrupt every downstream feature — hence this note.

======================================================================
Where the activations come from
======================================================================
The Bullfinch cache activations/bullfinch_layers_raw.npz stores, per layer,
(total_frames, 768) = (37 recordings x 512 patch tokens, 768). The CLS token
was already stripped upstream (data/loader.extract_all_layers strips index 0
from every transformer block), and each recording's 512 patch rows are stored
CONTIGUOUSLY in native patch order p = t*8 + f (verified: 37 contiguous runs,
exactly 512 rows each). So the cache gives us the 512 patches per clip in the
verified time-major order — we do NOT re-run the model.

The cache does NOT store each clip's true length, so padded time-patches are
still present. We recover the valid length per clip and keep only
valid_time = ceil(min(original_frames, 1024) / 16) time-patches, dropping the
zero-padded tail, using the project's canonical
valid_token_count_from_frames.

IMPORTANT — original_frames is computed on the FULL clip (NOT center-cropped
to 10 s). The cache was extracted by avex, which fills up to the full 1024
fbank frames (~10.24 s); it does NOT 10 s-crop the way
collect_esp_aves2_activations does. Feeding waveform_to_eat_input a 10 s crop
gives 998 frames -> valid_time 63 and wrongly drops one REAL slice from every
clip >= 10.24 s. Passing the full waveform (waveform_to_eat_input still
truncates at 1024) reproduces avex's boundary exactly.

This is verified, not assumed: at layer 0 the zero-mel padding makes a
padded time-patch's 8 freq sub-patches numerically IDENTICAL (across-freq
std == 0), giving an unambiguous per-clip boundary directly from the cache.
The full-clip original_frames rule matches that empirical boundary for all
37 clips (36 fill all 64 patches; XC788066, ~9.7 s, keeps 61). The script
re-checks this agreement at run time and errors on any mismatch. (Interior
in-clip silence also has low freq-std but is NOT trailing, so the boundary
uses the trailing padded run, not a global threshold.)

CPU-only, seed 42. Follows CLAUDE.md conventions.

Usage:
    source venv/bin/activate
    python -W ignore analysis/pool_patches_to_timeslices.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse constants + the canonical fbank/valid-token logic — do not hardcode.
from collect_esp_aves2_activations import (
    MODEL_SPECS,
    NUM_MELS,
    PATCH_SIZE,
    TARGET_FBANK_FRAMES,
    TOKENS_PER_SAMPLE,
    valid_token_count_from_frames,
    waveform_to_eat_input,
)
from bullfinch_within_layer_cluster import AUDIO_DIR, ACTIVATIONS_PATH, collect_clips
from data.loader import NUM_LAYERS_TOTAL, load_audio_file

SEED = 42

# Patch-grid geometry, derived from the shared constants (see the header proof).
N_FREQ_PATCHES = NUM_MELS // PATCH_SIZE          # 128 // 16 = 8
N_TIME_PATCHES = TARGET_FBANK_FRAMES // PATCH_SIZE  # 1024 // 16 = 64
N_PATCH_TOKENS = N_TIME_PATCHES * N_FREQ_PATCHES    # 64 * 8 = 512
SECONDS_PER_TIME_PATCH = PATCH_SIZE * 0.010          # 16 fbank frames * 10 ms = 0.160 s

# The EAT-family checkpoint the Bullfinch cache was extracted with.
NORM_SPEC = MODEL_SPECS["eat_all"]

OUT_NPZ = Path("activations/bullfinch_timeslices_pooled.npz")
OUT_INDEX_CSV = Path("activations/bullfinch_timeslices_index.csv")


def valid_time_patches_for_frames(original_frames: int) -> int:
    """Number of real (non-padded) time-patches, via the canonical convention.

    valid_token_count_from_frames returns 1 + valid_time * N_FREQ_PATCHES
    (the +1 is the CLS token), so we back out valid_time from it to keep a
    single source of truth for the padding rule.
    """
    valid_tokens = valid_token_count_from_frames(original_frames)
    return (valid_tokens - 1) // N_FREQ_PATCHES


def clip_original_frames(path: Path) -> int:
    """Pre-pad fbank frame count for a clip, matching the avex cache boundary.

    Uses the FULL clip (no 10 s crop): waveform_to_eat_input truncates at
    TARGET_FBANK_FRAMES (1024), so this reproduces avex's up-to-1024-frame
    behavior. See the module header for why 10 s-cropping is wrong here.
    """
    waveform = load_audio_file(str(path)).squeeze(0)  # (n_samples,) mono, 16 kHz
    _, original_frames = waveform_to_eat_input(
        waveform, norm_mean=NORM_SPEC.norm_mean, norm_std=NORM_SPEC.norm_std
    )
    return int(original_frames)


def empirical_valid_time(layer0: np.ndarray, rows: np.ndarray) -> int:
    """Valid time-patch count read directly from the cache (padding boundary).

    Zero-mel padding makes a padded time-patch's 8 freq sub-patches identical
    at layer 0 (across-freq std == 0). Returns 64 minus the trailing run of
    such patches (interior silence is ignored — it is not trailing).
    """
    grid = layer0[rows].reshape(N_TIME_PATCHES, N_FREQ_PATCHES, layer0.shape[1]).astype(np.float32)
    freq_std = grid.std(axis=1).mean(axis=1)  # (64,) across-freq variation per time-step
    trailing_pad = 0
    for k in range(N_TIME_PATCHES - 1, -1, -1):
        if freq_std[k] < 1e-3:
            trailing_pad += 1
        else:
            break
    return N_TIME_PATCHES - trailing_pad


def main() -> None:
    np.random.seed(SEED)  # convention; no randomness is used in pooling

    # Structural invariant from the header proof.
    assert 1 + N_TIME_PATCHES * N_FREQ_PATCHES == TOKENS_PER_SAMPLE == 513, (
        f"1 + {N_TIME_PATCHES}x{N_FREQ_PATCHES} != {TOKENS_PER_SAMPLE}"
    )
    print(f"[grid] 1 CLS + {N_TIME_PATCHES} time x {N_FREQ_PATCHES} freq "
          f"= {TOKENS_PER_SAMPLE} tokens  (time-major, reshape 512 -> (64, 8))")

    # --- load the flattened per-clip patch cache -------------------------------
    if not ACTIVATIONS_PATH.exists():
        raise FileNotFoundError(
            f"{ACTIVATIONS_PATH} not found — run bullfinch_within_layer_cluster.py first."
        )
    cache = np.load(ACTIVATIONS_PATH, allow_pickle=True)
    rec_idx = cache["rec_idx"].astype(np.int64)          # (total_frames,)
    rec_names = [str(n) for n in cache["rec_names"]]      # len == n_recordings
    n_recs = len(rec_names)
    print(f"[cache] {ACTIVATIONS_PATH}  rows={rec_idx.shape[0]}  recordings={n_recs}")

    # --- reconstruct the exact clip order that produced the cache --------------
    # rec_idx was assigned in collect_clips() iteration order (0..n-1); rebuild
    # that order so clip i's audio maps to the cache rows where rec_idx == i.
    clip_paths = collect_clips(AUDIO_DIR)
    if len(clip_paths) != n_recs:
        raise RuntimeError(
            f"collect_clips() returned {len(clip_paths)} paths but cache has "
            f"{n_recs} recordings; cannot align clips to cached rows."
        )
    path_stems = [p.stem for p in clip_paths]
    if path_stems != rec_names:
        raise RuntimeError(
            "collect_clips() ordering does not match cached rec_names — "
            "the clip<->row mapping would be wrong.\n"
            f"  first mismatch at index "
            f"{next(i for i, (a, b) in enumerate(zip(path_stems, rec_names)) if a != b)}"
        )

    # Per-clip contiguous row spans in the cache, in patch order.
    rows_for_clip: list[np.ndarray] = []
    for i in range(n_recs):
        rows = np.where(rec_idx == i)[0]
        if rows.shape[0] != N_PATCH_TOKENS:
            raise RuntimeError(
                f"clip {i} ({rec_names[i]}) has {rows.shape[0]} cached rows, "
                f"expected {N_PATCH_TOKENS}."
            )
        rows_for_clip.append(rows)

    # --- per-clip valid time-patch count (padding drop) ------------------------
    print("\n[padding] valid time-patches per clip "
          "(full-clip fbank frames -> ceil(min(frames,1024)/16)), "
          "cross-checked vs the layer-0 padding boundary:")
    layer0 = cache["layer_00"]
    original_frames = np.zeros(n_recs, dtype=np.int64)
    valid_time = np.zeros(n_recs, dtype=np.int64)
    for i, path in enumerate(clip_paths):
        frames = clip_original_frames(path)
        v = valid_time_patches_for_frames(frames)
        emp = empirical_valid_time(layer0, rows_for_clip[i])
        if v != emp:
            raise RuntimeError(
                f"clip {i} ({rec_names[i]}): audio-derived valid_time={v} disagrees "
                f"with the layer-0 cache boundary emp={emp}. The frame convention no "
                f"longer matches how the cache was extracted — investigate before pooling."
            )
        original_frames[i] = frames
        valid_time[i] = v
        print(f"  [{i + 1:2d}/{n_recs}] {rec_names[i][:44]:<44s} "
              f"frames={frames:4d}  valid_time={v:2d}  padded_dropped={N_TIME_PATCHES - v:2d}"
              f"  (cache-check ok)")

    if int(valid_time.max()) > N_TIME_PATCHES:
        raise RuntimeError(f"a clip kept {valid_time.max()} > {N_TIME_PATCHES} time-steps")

    total_slices = int(valid_time.sum())

    # --- pool freq patches, drop padded tail, for every layer ------------------
    print(f"\n[pool] mean over {N_FREQ_PATCHES} freq patches, keep valid time "
          f"for all {NUM_LAYERS_TOTAL} layers -> {total_slices} time-slices total")
    pooled_by_layer: dict[str, np.ndarray] = {}
    example_before_after: dict[str, tuple] = {}
    for layer in range(NUM_LAYERS_TOTAL):
        X = cache[f"layer_{layer:02d}"]  # (total_frames, 768) float16
        out = np.empty((total_slices, X.shape[1]), dtype=np.float16)
        write = 0
        for i in range(n_recs):
            patches = X[rows_for_clip[i]]                        # (512, 768), p = t*8+f
            grid = patches.reshape(N_TIME_PATCHES, N_FREQ_PATCHES, X.shape[1])  # (64, 8, 768)
            valid = grid[: valid_time[i]]                       # (V, 8, 768)
            pooled = valid.astype(np.float32).mean(axis=1).astype(np.float16)  # (V, 768)
            out[write : write + valid_time[i]] = pooled
            if layer == 6 and i == 0:  # one example clip, mid layer
                example_before_after = {
                    "clip": rec_names[i],
                    "patches": patches.shape,
                    "grid": grid.shape,
                    "valid": valid.shape,
                    "pooled": pooled.shape,
                }
            write += valid_time[i]
        assert write == total_slices, f"layer {layer}: wrote {write} != {total_slices}"
        pooled_by_layer[f"layer_{layer:02d}"] = out

    # --- parallel index: one row per pooled time-slice -------------------------
    slice_rec_idx = np.empty(total_slices, dtype=np.int32)
    slice_time_step = np.empty(total_slices, dtype=np.int32)
    slice_t_start_s = np.empty(total_slices, dtype=np.float32)
    w = 0
    for i in range(n_recs):
        v = int(valid_time[i])
        steps = np.arange(v, dtype=np.int32)
        slice_rec_idx[w : w + v] = i
        slice_time_step[w : w + v] = steps
        slice_t_start_s[w : w + v] = steps.astype(np.float32) * SECONDS_PER_TIME_PATCH
        w += v

    # --- save (gitignored activations/) ---------------------------------------
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        **pooled_by_layer,
        slice_rec_idx=slice_rec_idx,
        slice_time_step=slice_time_step,
        slice_t_start_s=slice_t_start_s,
        rec_names=np.array(rec_names),
        valid_time=valid_time.astype(np.int32),
        original_frames=original_frames.astype(np.int32),
    )
    with OUT_INDEX_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording_id", "time_step", "t_start_seconds"])
        for j in range(total_slices):
            writer.writerow([
                rec_names[slice_rec_idx[j]],
                int(slice_time_step[j]),
                f"{slice_t_start_s[j]:.3f}",
            ])

    # --- sanity checks ---------------------------------------------------------
    print("\n[sanity]")
    print(f"  1 + {N_TIME_PATCHES}x{N_FREQ_PATCHES} == {TOKENS_PER_SAMPLE}: "
          f"{1 + N_TIME_PATCHES * N_FREQ_PATCHES == TOKENS_PER_SAMPLE}")
    pooled_rows = pooled_by_layer["layer_00"].shape[0]
    print(f"  total pooled rows == sum(valid_time): "
          f"{pooled_rows} == {total_slices}  -> {pooled_rows == total_slices}")
    print(f"  no clip kept > {N_TIME_PATCHES} time-steps: "
          f"max={int(valid_time.max())}  -> {int(valid_time.max()) <= N_TIME_PATCHES}")
    print(f"  index rows == pooled rows: {total_slices} == {pooled_rows}  "
          f"-> {total_slices == pooled_rows}")
    e = example_before_after
    print(f"  example clip '{e['clip'][:40]}' (layer idx 6): "
          f"patches {e['patches']} -> grid {e['grid']} -> valid {e['valid']} "
          f"-> pooled {e['pooled']}")

    print("\n[summary]")
    print(f"  recordings                : {n_recs}")
    print(f"  original tokens per clip  : {TOKENS_PER_SAMPLE} "
          f"(1 CLS + {N_PATCH_TOKENS} patches); cache stores {N_PATCH_TOKENS} patches (CLS pre-stripped)")
    print(f"  valid time-steps kept     : total {total_slices}  "
          f"(min {int(valid_time.min())}, max {int(valid_time.max())}, "
          f"mean {valid_time.mean():.1f} per clip)")
    print(f"  padded time-steps dropped : total {n_recs * N_TIME_PATCHES - total_slices}")
    print(f"  pooled shape per layer    : ({total_slices}, {pooled_by_layer['layer_00'].shape[1]})  "
          f"x {NUM_LAYERS_TOTAL} layers")
    print(f"  saved                     : {OUT_NPZ}")
    print(f"  saved                     : {OUT_INDEX_CSV}")


if __name__ == "__main__":
    main()
