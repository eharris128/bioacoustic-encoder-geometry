"""Step 3a (pilot): bio↔non-bio audio mixing in `sl_eat_bio_ssl_all`.

For 5 bio clips A_i and 5 non-bio clips B_j, generate audio mixtures
`M_ij(α) = (1-α)·A_i + α·B_j` for α ∈ {0, 0.25, 0.5, 0.75, 1}, run each
through `sl_eat_bio_ssl_all`, and probe whether the L9 representation
slides linearly between bio and non-bio centroids or shows a threshold /
off-manifold behavior.

Diagnostics (all on L9 mean-pooled activations):
  - **Bio-axis projection.** Project pooled activation onto the unit
    vector `(c_bio - c_nonbio) / ||·||`. Plot vs α. Linear feature → line.
  - **Top-10 subspace energy.** Project onto the top-10 bio-only PCA basis
    (B_bio) and top-10 non-bio-only PCA basis (B_nonbio) computed from
    the §4 frame-level data. Plot energy in each vs α.
  - **Cosine to bio centroid vs non-bio centroid.**

Closes the first-pass diagnostic on RESULTS.md §9.7.

Output: artifacts/comparisons/<manifest>/nway_eat_all4/audio_mixing_pilot/

Usage:
    python step3a_audio_mixing_pilot.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

from collect_esp_aves2_activations import (
    DEFAULT_LAYER_NAMES, MODEL_SPECS, NUM_BLOCKS, TARGET_SAMPLE_RATE,
    TOKENS_PER_SAMPLE, EMBED_DIM, WINDOW_DURATION_S, BASE_EAT_MODEL_ID,
    crop_window, load_eat_model, load_manifest, register_hooks,
    valid_token_count_from_frames, waveform_to_eat_input,
    ManifestParquetAudioResolver, ensure_mono,
)
from step2_bootstrap_cis import (
    BASE_SEED, NATURE_SOURCES, load_layer_tensor, top_k_basis_via_cov,
)


MANIFEST_ID = "naturelm_by_source_100each_20260418T171459Z"
DEFAULT_MANIFEST = Path(f"artifacts/manifests/{MANIFEST_ID}.jsonl")
DEFAULT_ROADMAP_DIR = Path(f"artifacts/roadmap_part1/{MANIFEST_ID}")
DEFAULT_NWAY_DIR = Path(f"artifacts/comparisons/{MANIFEST_ID}/nway_eat_all4")
DEFAULT_OUTPUT_DIR = DEFAULT_NWAY_DIR / "audio_mixing_pilot"

FOCAL_MODEL = "sl_eat_bio_ssl_all"
FOCAL_LAYER = 9
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BIO = 5
N_NONBIO = 5
TOP_K = 10
FRAMES_PER_ITEM = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--roadmap_dir", type=Path, default=DEFAULT_ROADMAP_DIR)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=FOCAL_MODEL)
    p.add_argument("--layer_idx", type=int, default=FOCAL_LAYER)
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    p.add_argument("--n_bio", type=int, default=N_BIO)
    p.add_argument("--n_nonbio", type=int, default=N_NONBIO)
    p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--frames_per_item", type=int, default=FRAMES_PER_ITEM)
    p.add_argument("--device", type=str, default="cpu",
                   help="torch device for the model + forward passes (e.g. cuda).")
    return p.parse_args()


def select_clips(records: list[dict], n_bio: int, n_nonbio: int, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    bio = [r for r in records if r.get("source_dataset") in NATURE_SOURCES]
    non = [r for r in records if r.get("source_dataset") not in NATURE_SOURCES]
    bio_idx = rng.choice(len(bio), n_bio, replace=False)
    non_idx = rng.choice(len(non), n_nonbio, replace=False)
    return [bio[i] for i in bio_idx], [non[i] for i in non_idx]


def fetch_waveform(record: dict, resolver: ManifestParquetAudioResolver) -> torch.Tensor:
    waveform, sample_rate, _, _ = resolver.fetch_audio(record)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    target = int(TARGET_SAMPLE_RATE * WINDOW_DURATION_S)
    waveform = crop_window(waveform, target_num_samples=target, selection="center")
    if waveform.numel() < target:
        waveform = torch.nn.functional.pad(waveform, (0, target - waveform.numel()))
    return waveform


def compute_subspaces_and_centroids(
    args: argparse.Namespace,
) -> dict:
    """Compute, for the focal model + layer, top-10 bio-only and non-bio-only
    PCA bases plus per-class centroids from the existing 600-sample shards."""
    shard_dir = args.roadmap_dir / args.model / "shards"
    print(f"  loading {args.model} L{args.layer_idx} for subspace setup", flush=True)
    layer_tensor, sample_meta = load_layer_tensor(shard_dir, args.layer_idx)
    valid_token_counts = np.array(
        [int(s.get("valid_token_count", layer_tensor.shape[1])) for s in sample_meta]
    )
    sources = np.array([s.get("source_dataset", "") for s in sample_meta])
    is_bio = np.array([s in NATURE_SOURCES for s in sources])
    rng = np.random.default_rng(BASE_SEED)
    frames_per_item = args.frames_per_item
    n_items, t_max, d = layer_tensor.shape
    out = np.empty((n_items, frames_per_item, d), dtype=np.float32)
    for i in range(n_items):
        valid = max(int(min(valid_token_counts[i], t_max)), 1)
        if valid >= frames_per_item:
            f_idx = rng.choice(valid, frames_per_item, replace=False)
        else:
            f_idx = rng.choice(valid, frames_per_item, replace=True)
        out[i] = layer_tensor[i, f_idx, :]
    frames = out.reshape(-1, d).astype(np.float64)
    is_bio_per_frame = np.repeat(is_bio, frames_per_item)
    bio_frames = frames[is_bio_per_frame]
    nonbio_frames = frames[~is_bio_per_frame]
    basis_bio = top_k_basis_via_cov(bio_frames, args.top_k)
    basis_nonbio = top_k_basis_via_cov(nonbio_frames, args.top_k)

    # Pooled centroids per item (mean over its 50 frames)
    pooled = out.mean(axis=1)  # (n_items, D)
    c_bio = pooled[is_bio].mean(axis=0).astype(np.float64)
    c_nonbio = pooled[~is_bio].mean(axis=0).astype(np.float64)
    bio_axis = c_bio - c_nonbio
    bio_axis_unit = bio_axis / max(np.linalg.norm(bio_axis), 1e-12)
    return {
        "basis_bio": basis_bio,
        "basis_nonbio": basis_nonbio,
        "c_bio": c_bio,
        "c_nonbio": c_nonbio,
        "bio_axis_unit": bio_axis_unit,
    }


def project_diagnostics(pooled: np.ndarray, ctx: dict) -> dict:
    pooled = pooled.astype(np.float64)
    bio_axis_proj = float(pooled @ ctx["bio_axis_unit"])
    energy_bio = float(np.linalg.norm(pooled @ ctx["basis_bio"]) ** 2)
    energy_nonbio = float(np.linalg.norm(pooled @ ctx["basis_nonbio"]) ** 2)
    pn = pooled / max(np.linalg.norm(pooled), 1e-12)
    cb = ctx["c_bio"] / max(np.linalg.norm(ctx["c_bio"]), 1e-12)
    cn = ctx["c_nonbio"] / max(np.linalg.norm(ctx["c_nonbio"]), 1e-12)
    cos_to_bio = float(pn @ cb)
    cos_to_nonbio = float(pn @ cn)
    return {
        "bio_axis_projection": bio_axis_proj,
        "energy_in_bio_top10": energy_bio,
        "energy_in_nonbio_top10": energy_nonbio,
        "energy_ratio_bio_over_total": energy_bio / max(energy_bio + energy_nonbio, 1e-12),
        "cos_to_bio_centroid": cos_to_bio,
        "cos_to_nonbio_centroid": cos_to_nonbio,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio mixing pilot — {args.model} L{args.layer_idx}", flush=True)
    print(f"  alphas = {args.alphas}, n_bio={args.n_bio}, n_nonbio={args.n_nonbio}", flush=True)

    # Step A: compute reference subspaces + centroids from the existing shards
    ctx = compute_subspaces_and_centroids(args)
    np.savez(
        args.output_dir / f"reference_subspaces_{args.model}_L{args.layer_idx:02d}.npz",
        **ctx,
    )

    # Step B: pick bio + non-bio source clips from the manifest
    rng = np.random.default_rng(BASE_SEED)
    records = load_manifest(args.manifest)
    bio_records, non_records = select_clips(records, args.n_bio, args.n_nonbio, rng)
    print(f"  selected {len(bio_records)} bio + {len(non_records)} non-bio clips", flush=True)
    for r in bio_records:
        print(f"    bio:  {r.get('source_dataset')}/{r.get('file_name')}", flush=True)
    for r in non_records:
        print(f"    non:  {r.get('source_dataset')}/{r.get('file_name')}", flush=True)

    # Step C: load the focal model + register the layer hook
    print(f"\nLoading {args.model} on device={args.device!r}...", flush=True)
    model = load_eat_model(args.model, device=args.device)
    layer_name = DEFAULT_LAYER_NAMES[args.layer_idx]
    hooks, hook_outputs = register_hooks(model, [layer_name])

    # Step D: pre-fetch all source waveforms once
    spec = MODEL_SPECS[args.model]
    resolver = ManifestParquetAudioResolver(bio_records + non_records)
    bio_waves: list[torch.Tensor] = []
    non_waves: list[torch.Tensor] = []
    for r in bio_records:
        bio_waves.append(fetch_waveform(r, resolver))
    for r in non_records:
        non_waves.append(fetch_waveform(r, resolver))
    print(f"  fetched all {len(bio_waves) + len(non_waves)} source waveforms", flush=True)

    # Step E: forward pass on every (bio, non-bio, alpha) mixture
    records_out: list[dict] = []
    n_pairs = args.n_bio * args.n_nonbio * len(args.alphas)
    counter = 0
    t_start = time.time()
    for i, (br, bw) in enumerate(zip(bio_records, bio_waves)):
        for j, (nr, nw) in enumerate(zip(non_records, non_waves)):
            for alpha in args.alphas:
                mixed = (1.0 - alpha) * bw + alpha * nw
                eat_input, n_frames = waveform_to_eat_input(
                    mixed, norm_mean=spec.norm_mean, norm_std=spec.norm_std
                )
                eat_input = eat_input.to(args.device)
                hook_outputs.clear()
                with torch.no_grad():
                    _ = model.extract_features(eat_input)
                tensor = hook_outputs[layer_name].detach().cpu().numpy()  # (1, T, D)
                if tensor.shape != (1, TOKENS_PER_SAMPLE, EMBED_DIM):
                    raise RuntimeError(f"Unexpected hook shape: {tensor.shape}")
                valid = valid_token_count_from_frames(n_frames)
                if valid > 1:
                    pooled = tensor[0, 1:valid, :].mean(axis=0)
                else:
                    pooled = tensor[0, 0, :]
                diag = project_diagnostics(pooled, ctx)
                records_out.append({
                    "bio_idx": i,
                    "nonbio_idx": j,
                    "alpha": alpha,
                    "bio_file": br.get("file_name", ""),
                    "nonbio_file": nr.get("file_name", ""),
                    "bio_source": br.get("source_dataset", ""),
                    "nonbio_source": nr.get("source_dataset", ""),
                    "valid_token_count": valid,
                    **diag,
                })
                counter += 1
                if counter % 5 == 0 or counter == n_pairs:
                    print(
                        f"  [{counter:3d}/{n_pairs}] α={alpha:.2f} bio_axis_proj={diag['bio_axis_projection']:+.3f} "
                        f"energy_ratio_bio={diag['energy_ratio_bio_over_total']:.3f} "
                        f"({(time.time()-t_start)/counter:.1f}s/sample)",
                        flush=True,
                    )

    for h in hooks.values():
        h.remove()

    df = pd.DataFrame.from_records(records_out)
    df.to_csv(args.output_dir / "mixing_diagnostics.csv", index=False)

    # Aggregate by alpha
    agg = df.groupby("alpha").agg(
        bio_axis_proj_mean=("bio_axis_projection", "mean"),
        bio_axis_proj_std=("bio_axis_projection", "std"),
        energy_ratio_bio_mean=("energy_ratio_bio_over_total", "mean"),
        energy_ratio_bio_std=("energy_ratio_bio_over_total", "std"),
        cos_to_bio_mean=("cos_to_bio_centroid", "mean"),
        cos_to_bio_std=("cos_to_bio_centroid", "std"),
        cos_to_nonbio_mean=("cos_to_nonbio_centroid", "mean"),
        cos_to_nonbio_std=("cos_to_nonbio_centroid", "std"),
    ).reset_index()
    agg.to_csv(args.output_dir / "mixing_summary_by_alpha.csv", index=False)
    print("\n=== Mixing summary by alpha ===", flush=True)
    print(agg.round(4).to_string(index=False), flush=True)

    # Plots: each diagnostic vs alpha, with one line per (bio, non-bio) pair + mean
    for metric, ylabel in [
        ("bio_axis_projection", "projection onto (c_bio - c_nonbio)/||·||"),
        ("energy_ratio_bio_over_total", "energy in bio top-10 / (bio + nonbio)"),
        ("cos_to_bio_centroid", "cos(activation, bio centroid)"),
        ("cos_to_nonbio_centroid", "cos(activation, non-bio centroid)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (i, j), g in df.groupby(["bio_idx", "nonbio_idx"]):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g[metric], color="0.7", alpha=0.5, linewidth=0.8)
        # Mean ± 1 std overlay
        m = df.groupby("alpha")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(m["alpha"], m["mean"], yerr=m["std"],
                    color="tab:red", marker="o", linewidth=2.0, capsize=4,
                    label="mean ± 1 std")
        ax.set_xlabel("α (0 = pure bio, 1 = pure non-bio)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{args.model} L{args.layer_idx} — {metric} vs α")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(args.output_dir / f"mixing_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Diagnostic: linearity test on bio_axis_projection. If linear feature,
    # mean(α=0.5) should equal 0.5 * (mean(α=0) + mean(α=1)).
    means = agg.set_index("alpha")["bio_axis_proj_mean"].to_dict()
    if 0.0 in means and 1.0 in means and 0.5 in means:
        midpoint_predicted = 0.5 * (means[0.0] + means[1.0])
        midpoint_observed = means[0.5]
        deviation = midpoint_observed - midpoint_predicted
        print(
            f"\nLinearity test (bio_axis_projection):"
            f"\n  predicted midpoint at α=0.5 = 0.5 * ({means[0.0]:.4f} + {means[1.0]:.4f}) = {midpoint_predicted:+.4f}"
            f"\n  observed at α=0.5 = {midpoint_observed:+.4f}"
            f"\n  deviation = {deviation:+.4f} ({100 * abs(deviation) / max(abs(means[1.0] - means[0.0]), 1e-9):.1f}% of full range)",
            flush=True,
        )

    print(f"\nSaved mixing pilot to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
