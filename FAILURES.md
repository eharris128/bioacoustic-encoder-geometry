# Failures and recovery log

Captured incidents from the autonomous orchestration runs. Each entry
documents a real failure mode and the fix so we don't re-hit it.

## 2026-04-27T22:30Z — HF download hang in extraction

**Symptom.** `collect_esp_aves2_activations.py --device cuda` on the A10
hung for 44+ min with 0 shards written and only ~20s CPU time. The
process held a TCP socket in `CLOSE-WAIT` to a HuggingFace IP — the
remote had closed its end of the connection but our Python process
never read EOF.

**Root cause.** `huggingface_hub` downloads default to no read timeout.
A throttled connection or partial response from HF's CDN leaves the
connection half-open indefinitely. The unauthenticated-rate-limit
warning at the top of the run was a hint.

**Fix.** Set `HF_HUB_DOWNLOAD_TIMEOUT=120` in the environment before
calling extractor. Added to the venv activate script and exported
explicitly in the orchestrate.sh launch command. Also documented in
remote_setup.md.

**Recovery.** Killed the hung process. Once the env var was set, the
restart resumed correctly — the partial download had already cached
the model files.

## 2026-04-27T23:17Z — Missing torchcodec broke every extraction

**Symptom.** Every model's extraction failed in seconds with
`ModuleNotFoundError: No module named 'torchcodec'`. Then every Phase 2
analysis cascaded with `ValueError: need at least one array to
concatenate` because there were no shards to load.

**Root cause.** Newer `torchaudio` (>=2.4 ish) delegates audio decoding
to a separate `torchcodec` package, and `torchaudio.load()` raises an
ImportError when it's missing. We installed `torch torchaudio` but
didn't install `torchcodec`. The local-machine venv has it via a
different installation path so we didn't notice in development.

**Fix.** `pip install torchcodec` on the A10 (installs 0.11.1+cpu —
CPU-only is fine because torchcodec only handles audio decoding before
the GPU forward pass). Added to remote_setup.md.

**Recovery.** Restarted orchestrate.sh after the install. Idempotent
phase-skip logic correctly resumed at eat_all (no shards yet), then
proceeded through the rest of Phase 1.

## Open watch-points

- **wait_for_other_extractions in orchestrate.sh** has a buggy
  comparison (`$$` is the shell PID, not the python child) — but in
  practice it doesn't loop forever because once a stuck extraction is
  killed, pgrep returns empty and the outer while exits. Don't rely on
  it for finer coordination than "wait for any other collect_esp_aves2
  process to finish."
- The `orchestrate.done` sentinel file gets written even if every phase
  failed — `set -u` only catches unbound vars, not subshell exits. The
  babysitter loop should treat orchestrate.done as "ran to completion,"
  not "succeeded." Check actual artifacts on disk before declaring
  Phase X done.
