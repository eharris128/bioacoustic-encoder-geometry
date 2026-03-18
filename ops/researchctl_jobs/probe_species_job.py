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


BEST_LAYER_RE = re.compile(r"Best layer:\s*(\d+)\s*\(([\d.]+)%\)")
WORST_LAYER_RE = re.compile(r"Worst layer:\s*(\d+)\s*\(([\d.]+)%\)")


def main() -> int:
    parser = common_parser("probe-species", "Run probe_species.py inside a researchctl wrapper")
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
    proc = run_script("probe_species.py", workspace)
    finished_at = now_iso()

    summary: dict[str, object] = {}
    best_match = BEST_LAYER_RE.search(proc.stdout or "")
    if best_match:
        summary["best_layer"] = int(best_match.group(1))
        summary["best_accuracy_pct"] = float(best_match.group(2))
    worst_match = WORST_LAYER_RE.search(proc.stdout or "")
    if worst_match:
        summary["worst_layer"] = int(worst_match.group(1))
        summary["worst_accuracy_pct"] = float(worst_match.group(2))

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        src = workspace / "probe_species.png"
        dest = output_root / "probe_species.png"
        shutil.copy2(src, dest)
        artifacts.append("probe_species.png")

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
