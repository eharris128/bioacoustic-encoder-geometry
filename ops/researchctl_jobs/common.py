from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_git_sha() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return value or None


def default_run_id(job_id: str) -> str:
    env_run_id = os.environ.get("RESEARCHCTL_RUN_ID")
    if env_run_id:
        return env_run_id
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{job_id}-{stamp}"


def default_output_root(job_id: str, run_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "runs" / job_id / run_id


def common_parser(job_id: str, description: str) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument("--run-id", default="", help="Explicit run id")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("RESEARCHCTL_OUTPUT_ROOT", ""),
        help="Directory where wrapped artifacts should be written",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Do not delete the temporary workspace after completion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved paths and exit without executing the wrapped script",
    )
    parser.set_defaults(job_id=job_id)
    return parser


def resolve_paths(args: Namespace) -> tuple[str, Path]:
    run_id = args.run_id or default_run_id(args.job_id)
    output_root = Path(args.output_root) if args.output_root else default_output_root(args.job_id, run_id)
    return run_id, output_root.resolve()


def prepare_workspace(job_id: str, run_id: str, input_names: Iterable[str]) -> Path:
    workspace = REPO_ROOT / ".researchctl" / "workspaces" / job_id / run_id
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for name in input_names:
        src = REPO_ROOT / name
        dest = workspace / name
        if not src.exists():
            raise FileNotFoundError(f"Required input path missing: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dest, target_is_directory=src.is_dir())
    return workspace


def run_script(script_name: str, workspace: Path) -> subprocess.CompletedProcess[str]:
    script_path = REPO_ROOT / script_name
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc


def copy_path(src: Path, dest: Path) -> list[str]:
    if not src.exists():
        raise FileNotFoundError(f"Expected artifact missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return sorted(
            str(path.relative_to(dest.parent))
            for path in dest.rglob("*")
            if path.is_file()
        )
    shutil.copy2(src, dest)
    return [dest.name]


def write_result(output_root: Path, payload: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result_path
