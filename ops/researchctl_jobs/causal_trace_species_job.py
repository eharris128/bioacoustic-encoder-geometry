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

BEST_PATCH_LAYER_RE  = re.compile(r"Best patch layer:\s*(\d+)")
BEST_TRANSFER_RE     = re.compile(r"transfer acc = ([\d.]+)\)")
LAYER1_TRANSFER_RE   = re.compile(r"Layer 1 transfer acc:\s*([\d.]+)")
LAYER11_TRANSFER_RE  = re.compile(r"Layer 11 transfer acc:\s*([\d.]+)")
LAYER1_PROBE_RE      = re.compile(r"Layer 1 transfer acc:.*probe acc = ([\d.]+)\)")
LAYER11_PROBE_RE     = re.compile(r"Layer 11 transfer acc:.*probe acc = ([\d.]+)\)")


def main() -> int:
    parser = common_parser(
        "causal-trace-species",
        "Run causal_trace_species.py inside a researchctl wrapper",
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
    proc = run_script("causal_trace_species.py", workspace)
    finished_at = now_iso()

    output = proc.stdout or ""

    summary: dict[str, object] = {}
    for pattern, key, cast in [
        (BEST_PATCH_LAYER_RE,  "best_patch_layer",      int),
        (BEST_TRANSFER_RE,     "best_transfer_acc",      float),
        (LAYER1_TRANSFER_RE,   "layer1_transfer_acc",    float),
        (LAYER11_TRANSFER_RE,  "layer11_transfer_acc",   float),
        (LAYER1_PROBE_RE,      "layer1_probe_acc",       float),
        (LAYER11_PROBE_RE,     "layer11_probe_acc",      float),
    ]:
        m = pattern.search(output)
        if m:
            summary[key] = cast(m.group(1))

    artifacts: list[str] = []
    if proc.returncode == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        for fname in [
            "causal_trace_species.png",
            "species_separation.png",
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
