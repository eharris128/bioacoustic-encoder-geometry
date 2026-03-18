# Claude Handoff: First `researchctl` Remote Run

Date: 2026-03-17

## Purpose

This note captures the state of the first real `researchctl` run against this repo, routed through the existing DigitalOcean control node and executed on the existing Lambda instance.

Use this as the starting point instead of re-discovering the environment.

## What Already Worked

The first remote `researchctl` run succeeded.

- Job: `probe-species`
- Run ID: `RUN-000001`
- Attempt ID: `attempt-0001`
- Exit code: `0`
- Local landed bundle:
  - `artifacts/runs/probe-species/RUN-000001`
- Result manifest:
  - `artifacts/runs/probe-species/RUN-000001/outputs/result.json`

Key result summary from `result.json`:

- `best_layer = 1`
- `best_accuracy_pct = 93.6`
- `worst_layer = 6`
- `worst_accuracy_pct = 83.7`

The local landed bundle includes logs, provenance, sync markers, heartbeat/progress state, and the generated `probe_species.png`.

## Why This Matters

The control-node -> Lambda path is now proven for the extracted `researchctl`.

That proof included:

- staging `researchctl` onto the control node
- staging `sentient-futures` onto the control node
- syncing both to Lambda
- running `researchctl` on Lambda through a control-node SSH hop
- pulling the resulting bundle back to the local repo

This means the next agent should not start by rebuilding the whole path unless there is a specific breakage.

## Remote Topology

Control node:

- `root@134.199.202.234`

Lambda worker:

- `ubuntu@132.145.140.49`

Current staged paths on the control node:

- `/home/researchctl/staging/researchctl`
- `/home/researchctl/staging/sentient-futures`

Current live paths on Lambda:

- `/home/ubuntu/researchctl`
- `/home/ubuntu/sentient-futures`
- `/home/ubuntu/venvs/sentient-futures-researchctl`

Current Lambda state root:

- `/home/ubuntu/.local/state/researchctl`

Current Lambda DB used for this first run:

- `/home/ubuntu/.local/state/researchctl/sentient-futures-first.db`

Remote bundle copy on Lambda:

- `/home/ubuntu/sentient-futures/artifacts/runs/probe-species/RUN-000001`

Control-node pulled copy:

- `/home/researchctl/runs/sentient-futures/probe-species/RUN-000001`

Local pulled copy:

- `artifacts/runs/probe-species/RUN-000001`

## SSH State

The control node originally could not reach the Lambda machine. That is now fixed.

What changed:

- the Lambda `authorized_keys` now includes the control-node worker key
- the control-node `researchctl` user SSH config now points at:
  - `/home/researchctl/.ssh/researchctl_worker`

Do not undo that unless you are intentionally rotating keys.

## Packaging / Python Reality

This is the main operational gap right now.

The extracted `researchctl` repo at `/home/evan/projects/researchctl` declares Python `>=3.11`, but the active Lambda machine has Python `3.10.12`.

Because of that:

- `pip install -e ~/researchctl` failed on Lambda
- the successful run used:
  - a Python 3.10 venv
  - manual dependency installation
  - `PYTHONPATH=~/researchctl/src python -m researchctl ...`

That workaround is real and working, but it should not be the long-term packaging story.

## Transfer / Bandwidth Note

Be mindful of DO egress.

For the first zero-code-change wrapper path, I synced the repo plus these local input directories:

- `models/`
- `audio/`
- `aves/`

The transfer from local -> control node for `sentient-futures` was about `532 MB`.

That is acceptable for one run, but Claude should avoid retransferring those directories unless they actually changed.

At this point the remote machines already have what they need for the existing wrapper-backed jobs.

## What The First Agent Chose

The note in [FIRST_RESEARCH_CTRL_EXPERIMENT.md](/home/evan/projects/sentient-futures/FIRST_RESEARCH_CTRL_EXPERIMENT.md) recommends an SAE on AVES layer-11 embeddings as the best next scientific experiment.

That recommendation still stands.

However, the current `researchctl` adapter in this repo only exposes two runnable jobs:

- `probe-species`
- `explore-clusters`

So for the first real `researchctl` proof run, I chose `probe-species` because it was the smallest valid end-to-end infrastructure test.

## Immediate Next Step For Claude

If the goal is to continue the research stream rather than just validate infra, Claude should add an SAE-backed `researchctl` job in this repo and run that on the already-prepared Lambda machine.

Recommended order:

1. Confirm the remote state is still intact.
2. Reuse the existing staged repos and venv.
3. Add an SAE wrapper + job manifest to `.researchctl/jobs/` and `ops/researchctl_jobs/`.
4. Run that SAE job through the same control-node -> Lambda path.
5. Pull the landed bundle back locally.

If Claude wants a second infra-only validation before touching SAE, `explore-clusters` is the next existing job to run.

## Suggested Verification Command

From the local machine, this is a good status check through the control node:

```bash
ssh root@134.199.202.234 'su - researchctl -c "ssh ubuntu@132.145.140.49 '\''bash -lc \"source ~/venvs/sentient-futures-researchctl/bin/activate; cd ~/sentient-futures; PYTHONPATH=~/researchctl/src python -m researchctl --db ~/.local/state/researchctl/sentient-futures-first.db status\"'\''"'
```

## Known Gaps

1. `researchctl` packaging and the current Lambda Python version disagree.
2. The control-node path is operationally real, but still implemented as an SSH hop, not a first-class remote transport inside `researchctl`.
3. This repo does not yet expose the recommended SAE experiment through `researchctl`.

## `play` Repo: Immediate Cleanup Next Steps

These are the cleanup items I would do next in `/home/evan/projects/spar/play`.

1. Resolve the Python-version story between `researchctl` and the actual runner fleet.
   - Either support Python 3.10 in the package metadata, or move the control-node/Lambda path to Python 3.11+.
2. Finish the naming cleanup around the adapter seam.
   - The package is now `researchctl`, but `play` still uses names like `researchctl_core_adapter` and `export_researchctl_core_jobs.py`.
3. Decide the install/update story for the control node.
   - Right now the old play-bound checkout still exists at `/opt/researchctl/repo`, while the new standalone repo lives separately at `/home/evan/projects/researchctl`.
4. Decide whether `play` will consume `researchctl` from a sibling checkout, a pinned git revision, or an installed package.
5. Make the control-node -> Lambda path explicit.
   - Either keep the SSH-hop model and document it cleanly, or promote it into a proper `researchctl` transport/runner profile.
6. Review and commit or discard the current bridge-related local changes in `play`.
   - Current `git status` there includes modified tracked files plus untracked adapter/bridge files.
7. Keep scientific queue/review semantics out of `researchctl`.
   - The adapter seam should stay focused on discovery and launch translation.

## Relevant Repos And Commits

Standalone `researchctl` repo:

- `/home/evan/projects/researchctl`
- most recent commits:
  - `92327d2 Rename: publish researchctl package surface`
  - `751e005 Init: researchctl extraction scaffold`

`play` adapter seam:

- `/home/evan/projects/spar/play/researchctl_core_adapter`
- `/home/evan/projects/spar/play/scripts/run_play_researchctl_job.py`
- `/home/evan/projects/spar/play/scripts/export_researchctl_core_jobs.py`

## Bottom Line

The first remote `researchctl` run is done and worked.

Claude should not restart from provisioning. The real next move is to use the already-working path to run the next scientifically meaningful job, which is probably the SAE experiment described in [FIRST_RESEARCH_CTRL_EXPERIMENT.md](/home/evan/projects/sentient-futures/FIRST_RESEARCH_CTRL_EXPERIMENT.md).
