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
    copy_path,
)

BEST_K_RE      = re.compile(r"Best k:\s*(\d+)")
SIL_RE         = re.compile(r"Silhouette score:\s*([\d.]+)")
AC_ACC_RE      = re.compile(r"Acoustic accuracy:\s*([\d.]+)")
AC_LIFT_RE     = re.compile(r"lift = ([\d.]+)\)")
TOTAL_RE       = re.compile(r"Total frames:\s*(\d+)")


def main() -> int:
    parser = common_parser(
        "bullfinch-layer11-structure",
        "Run bullfinch_layer11_structure.py inside a researchctl wrapper",
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
    proc = run_script("bullfinch_layer11_structure.py", workspace)
    finished_at = now_iso()

    output = proc.stdout or ""

    summary: dict[str, object] = {}
    for pattern, key, cast in [
        (BEST_K_RE,   "best_k",              int),
        (SIL_RE,      "silhouette",           float),
        (AC_ACC_RE,   "acoustic_accuracy",    float),
        (AC_LIFT_RE,  "acoustic_lift",        float),
        (TOTAL_RE,    "total_frames",         int),
    ]:
        m = pattern.search(output)
        if m:
            summary[key] = cast(m.group(1))

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        for fname in [
            "layer11_cluster_pca.png",
            "layer11_cluster_acoustics.png",
            "layer11_silhouette.png",
        ]:
            src = workspace / fname
            if src.exists():
                shutil.copy2(src, output_root / fname)
                artifacts.append(fname)

        # Copy audio snippets directory
        snippets_src = workspace / "audio_snippets"
        if snippets_src.exists():
            snippets_dest = output_root / "audio_snippets"
            try:
                copied = copy_path(snippets_src, snippets_dest)
                artifacts.extend(copied)
            except Exception as e:
                print(f"Warning: could not copy audio_snippets: {e}")

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
