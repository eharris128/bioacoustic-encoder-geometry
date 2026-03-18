from __future__ import annotations

import re
import shutil

from common import (
    common_parser,
    current_git_sha,
    now_iso,
    prepare_workspace,
    resolve_paths,
    run_script,
    write_result,
)

BEST_LAYER_RE = re.compile(r"Best causal layer:\s*(\d+)")
BEST_ALPHA50_RE = re.compile(r"alpha_50 = ([\d.]+|None)")


def main() -> int:
    parser = common_parser(
        "contrastive-patch-species",
        "Run contrastive_patch_species.py inside a researchctl wrapper",
    )
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
    proc = run_script("contrastive_patch_species.py", workspace)
    finished_at = now_iso()

    output = proc.stdout or ""

    summary: dict[str, object] = {}
    m = BEST_LAYER_RE.search(output)
    if m:
        summary["best_causal_layer"] = int(m.group(1))
    m = BEST_ALPHA50_RE.search(output)
    if m and m.group(1) != "None":
        summary["best_alpha50"] = float(m.group(1))

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        for fname in [
            "contrastive_patch_alpha_sweep.png",
            "contrastive_patch_summary.png",
        ]:
            src = workspace / fname
            if src.exists():
                shutil.copy2(src, output_root / fname)
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
