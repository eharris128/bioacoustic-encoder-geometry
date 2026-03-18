# Repository Guidelines

## Project Structure & Module Organization

This repository is a script-first AVES interpretability project. Top-level `*.py` files are the primary analyses, for example `run_aves.py`, `explore_layers.py`, `probe_species.py`, and `compare_hubert.py`. Generated figures are usually written to the repo root as `*.png`; cluster audio/plots go under `cluster_output/`. Automation wrappers live in `ops/researchctl_jobs/` and write structured outputs to `artifacts/runs/<job>/<run_id>/`. Local model/data dependencies live in `models/`, `audio/`, and the cloned `aves/` repo; these paths are gitignored and should stay local.

## Build, Test, and Development Commands

Create the environment from the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install esp-aves torchcodec matplotlib scikit-learn
```

Run analyses directly from the project root, for example:

```bash
python run_aves.py
python explore_clusters.py
python probe_species.py
python -W ignore explore_temporal.py
```

Use the wrappers when you want isolated workspaces and archived artifacts:

```bash
python ops/researchctl_jobs/probe_species_job.py --dry-run
python ops/researchctl_jobs/explore_clusters_job.py
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation, module docstrings, and clear top-level constants such as `MODEL_PATH` and `LAYER`. Prefer small, standalone scripts over framework-heavy abstractions. Follow existing naming: exploratory scripts use `explore_<topic>.py`, comparisons use `compare_<topic>.py`, and saved figures use lowercase snake_case names. Keep random seeds fixed at `42` when sampling or clustering, and save plots with `dpi=150` and `bbox_inches="tight"` to match the existing outputs.

## Testing Guidelines

There is no formal automated test suite yet. Validate changes by running the affected script end to end and confirming the expected PNGs, audio clips, or `artifacts/runs/.../result.json` files are produced. For new analysis code, document required inputs and include a reproducible smoke-test command in the PR.

## Commit & Pull Request Guidelines

Match the existing commit style: short, imperative, capitalized subjects such as `Add acoustic feature probing and cluster profiling`. Keep commits focused on one analysis or automation change. PRs should explain the research question, list required local assets (`aves/`, `audio/`, `models/`), and attach representative plots when visual outputs change. Do not commit large local data, model weights, virtualenv files, or `.researchctl/workspaces/` contents.
