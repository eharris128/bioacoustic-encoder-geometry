from __future__ import annotations

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


def main() -> int:
    parser = common_parser("explore-clusters", "Run explore_clusters.py inside a researchctl wrapper")
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
    proc = run_script("explore_clusters.py", workspace)
    finished_at = now_iso()

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        src = workspace / "cluster_output"
        dest = output_root / "cluster_output"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        artifacts = sorted(
            str(path.relative_to(output_root))
            for path in dest.rglob("*")
            if path.is_file()
        )

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
            "summary": {"artifact_count": len(artifacts)},
        },
    )

    if not args.keep_workspace:
        shutil.rmtree(workspace)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
