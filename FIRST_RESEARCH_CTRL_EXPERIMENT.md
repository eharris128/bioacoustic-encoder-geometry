# First Researchctl Experiment

## Scope

This note captures the current recommendation for the first remote experiment and the data that would need to be available on a Lambda instance. No Lambda instance was booted while preparing this note.

## Recommended First Experiment

The best next experiment is a sparse autoencoder (SAE) on AVES layer-11 embeddings.

Why this is the right next step:
- It is already marked high priority in `TODO.md`.
- It builds directly on the current cluster and acoustic-probe results.
- It does not require acquiring a new external dataset.
- It addresses the main open question in the repo: late-layer representations are acoustically meaningful, but not linearly decodable from simple hand-engineered features.

## What the SAE Run Actually Needs

For a first SAE pass, the relevant local inputs are:
- `audio/bullfinch/`
- `models/aves-base-all.torchaudio.pt`
- `aves/config/default_cfg_aves-base-all.json`
- the repo code itself

Practical sizes on the current machine:
- `models/`: about `361M`
- `audio/bullfinch/`: about `89M`
- `aves/config/`: negligible

## Minimal Remote Payload

If the Lambda instance can clone the repo and download the AVES checkpoint/config during bootstrap, the only local dataset that really needs to be copied from this machine is:

- `audio/bullfinch/`

That is the minimal payload for the recommended SAE experiment.

## Zero-Code-Change Payload

If we want to reuse the existing `researchctl` wrappers and current path assumptions without changing code, the Lambda instance should have these repo-root directories present:

- `models/`
- `audio/`
- `aves/`

Reason: the wrapper staging code currently expects those inputs to exist locally and symlinks them into the run workspace.

## Not Needed For The First SAE Pass

These are not required for the initial SAE experiment:
- `audio/hawfinch/`
- `aves/example_audios/`
- existing `*.png` outputs
- `cluster_output/`
- `artifacts/runs/`
- `venv/`

`audio/hawfinch/` only becomes relevant if we add a cross-species validation step later.

## Repo State Note

Tracked git content is aligned with `origin/main`, but the main experiment inputs are intentionally not pushed to GitHub because they are gitignored:
- `audio/`
- `aves/`
- `models/`

So a remote machine cannot rely on `git clone` alone to reproduce the current experiments.

## Decision For The Next Agent

Choose between two paths:

1. Minimal-data SAE path:
   bootstrap AVES config/checkpoint remotely and copy only `audio/bullfinch/`.

2. Lowest-friction wrapper path:
   copy `models/`, `audio/`, and `aves/` unchanged so current wrappers work immediately.
