from __future__ import annotations

import re
import shutil
from pathlib import Path

from common import (
    common_parser,
    current_git_sha,
    now_iso,
    prepare_workspace,
    resolve_paths,
    run_script,
    write_result,
)

FINAL_RECON_RE = re.compile(r"Final reconstruction loss:\s*([\d.eE+\-]+)")
FINAL_SPARSITY_RE = re.compile(r"Final sparsity \(active fraction\):\s*([\d.]+)")
DEAD_FEATURES_RE = re.compile(r"Dead features:\s*([\d.]+)%")
HIDDEN_DIM_RE = re.compile(r"Hidden dim:\s*(\d+)")
TARGET_LAYER_RE = re.compile(r"Target layer:\s*(\d+)")


def main() -> int:
    parser = common_parser("sae-layer11", "Run sae_layer11.py inside a researchctl wrapper")
    args = parser.parse_args()
    run_id, output_root = resolve_paths(args)
    workspace = prepare_workspace(args.job_id, run_id, ["aves", "audio", "models"])

    if args.dry_run:
        print(f"job_id={args.job_id}")
        print(f"run_id={run_id}")
        print(f"workspace={workspace}")
        print(f"output_root={output_root}")
        return 0

    started_at = now_iso()
    proc = run_script("sae_layer11.py", workspace)
    finished_at = now_iso()

    output = proc.stdout or ""

    summary: dict[str, object] = {}
    m = FINAL_RECON_RE.search(output)
    if m:
        summary["final_recon_loss"] = float(m.group(1))
    m = FINAL_SPARSITY_RE.search(output)
    if m:
        summary["final_sparsity"] = float(m.group(1))
    m = DEAD_FEATURES_RE.search(output)
    if m:
        summary["dead_feature_pct"] = float(m.group(1))
    m = HIDDEN_DIM_RE.search(output)
    if m:
        summary["hidden_dim"] = int(m.group(1))
    m = TARGET_LAYER_RE.search(output)
    if m:
        summary["target_layer"] = int(m.group(1))

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        for fname in [
            "sae_layer11_training.png",
            "sae_layer11_features.png",
            "sae_layer11_weights.npz",
        ]:
            src = workspace / fname
            if src.exists():
                dest = output_root / fname
                shutil.copy2(src, dest)
                artifacts.append(fname)

    write_result(
        output_root,
        {
            "job_id": args.job_id,
            "run_id": run_id,
            "git_sha": current_git_sha(),
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": proc.returncode,
            "artifacts": artifacts + ["result.json"],
            "summary": summary,
        },
    )

    if not args.keep_workspace:
        shutil.rmtree(workspace)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
