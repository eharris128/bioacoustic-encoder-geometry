# Repository Guidelines

## Project Structure & Module Organization

Script-first interpretability project on the ESP-AVES2 EAT family. Top-level
`*.py` files are the analyses — `collect_esp_aves2_activations.py` extracts
residual-stream activations, `step2_*.py` / `step3*.py` / `step5_*.py` etc.
compute geometry. Generated CSVs and plots write to
`artifacts/comparisons/<manifest>/nway_eat_all4/<subdir>/`. Activation
shards land under `artifacts/roadmap_part1/<manifest>/<model>/shards/`
and are gitignored. The committed manifest lives in `artifacts/manifests/`.

## Build, Test, and Development Commands

Create the environment from the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio transformers huggingface_hub safetensors \
            pyarrow matplotlib scikit-learn scipy timm
```

Run analyses directly from the project root:

```bash
python collect_esp_aves2_activations.py --manifest <jsonl> --models <key,...>
python -W ignore step2_tier1_frame_level.py
python -W ignore step3c_veitch_hierarchy.py
```

Scripts read from shards on disk and write to `artifacts/comparisons/...`.
There is no test suite — validation is "did the expected CSVs and PNGs
appear, and do the numbers match what `RESULTS.md` claims?"

## Coding Style & Naming Conventions

Python with 4-space indentation, module docstrings, top-level constants
such as `MANIFEST_PATH` and `MODELS`. Prefer small, standalone scripts
over framework-heavy abstractions. Existing names: extraction
(`collect_*.py`), geometry stages (`step2_*.py`, `step3_*.py`, `step5_*.py`),
sample manifests (`sample_naturelm_by_*.py`). Random seed is `42` for
all subsampling and random-init. Save plots with `dpi=150` and
`bbox_inches="tight"`.

Define each metric primitive once in `step2_tier1_frame_level.py` and
import it elsewhere — do not duplicate `eff_rank`, PR, or MLE-ID across
scripts.

## Testing Guidelines

No formal automated test suite. Validate changes by running the affected
script end to end, confirming the expected CSVs / PNGs land under
`artifacts/comparisons/`, and cross-checking against the numbers in
`RESULTS.md`. For a new analysis, document required inputs and include a
reproducible smoke-test command in the PR.

## Commit & Pull Request Guidelines

Match the existing commit style: short, imperative subjects (e.g.
`Add Veitch orthogonality test for Class / Order`). Keep commits focused
on one analysis or pipeline change. PRs should explain the research
question, list required local assets (manifest path, models extracted),
and attach representative plots when visual outputs change. Do not
commit large binaries — model weights, virtualenvs, or activation shards.
