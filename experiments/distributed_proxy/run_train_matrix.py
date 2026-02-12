#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunItem:
    group: str
    model: str
    dataset: str
    mode: str
    seed: int
    gpu: int
    entry: str
    args: List[str]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def _build_runs(cfg: Dict, only: Optional[str]) -> List[RunItem]:
    mode_flags = cfg.get("default_modes", {})
    runs: List[RunItem] = []
    for group in cfg.get("groups", []):
        group_name = group["name"]
        if only and only not in group_name:
            continue

        model = group["model"]
        dataset = group["dataset"]
        gpu = int(group.get("gpu", 0))
        entry = group["entry"]
        base_args = list(group.get("args", []))

        seeds = list(group.get("seeds", [0]))
        modes = list(group.get("modes", ["full"]))

        for seed in seeds:
            for mode in modes:
                if mode not in mode_flags:
                    raise ValueError(f"mode '{mode}' missing from default_modes")
                args = ["-d", dataset, "--seed", str(seed), "--gpu", str(gpu)]
                args.extend(base_args)
                args.extend(mode_flags[mode])
                runs.append(
                    RunItem(
                        group=group_name,
                        model=model,
                        dataset=dataset,
                        mode=mode,
                        seed=int(seed),
                        gpu=gpu,
                        entry=entry,
                        args=args,
                    )
                )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rebuttal training matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-python-dir",
        type=Path,
        default=Path("python"),
        help="repository-local python package directory (must contain atlas/)",
    )
    parser.add_argument("--only", type=str, default="", help="substring filter for group name")
    parser.add_argument("--max-runs", type=int, default=0, help="truncate total runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    examples_dir = Path(cfg.get("examples_dir", "examples")).resolve()
    repo_python_dir = args.repo_python_dir.resolve()
    python_exec = cfg.get("python", sys.executable)

    if not examples_dir.exists():
        raise FileNotFoundError(f"examples_dir not found: {examples_dir}")
    if not repo_python_dir.exists():
        raise FileNotFoundError(f"repo_python_dir not found: {repo_python_dir}")

    runs = _build_runs(cfg, args.only or None)
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    if not runs:
        print("No runs selected.")
        return 0

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = args.results_dir.resolve()
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_root / f"run_plan_{timestamp}.json"

    plan_payload = {
        "created_at": timestamp,
        "examples_dir": str(examples_dir),
        "repo_python_dir": str(repo_python_dir),
        "python": python_exec,
        "num_runs": len(runs),
        "runs": [
            {
                "group": item.group,
                "model": item.model,
                "dataset": item.dataset,
                "mode": item.mode,
                "seed": item.seed,
                "gpu": item.gpu,
                "entry": item.entry,
                "args": item.args,
            }
            for item in runs
        ],
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    print(f"[plan] wrote {plan_path}")
    print(f"[plan] total runs: {len(runs)}")

    failures = 0
    for index, item in enumerate(runs, start=1):
        log_name = _safe_name(f"{index:03d}_{item.group}_{item.model}_{item.dataset}_{item.mode}_s{item.seed}.log")
        log_path = logs_dir / log_name
        cmd = [python_exec, item.entry, *item.args]
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[{index}/{len(runs)}] {item.group} | {item.mode} | seed={item.seed}")
        print(f"[cmd] {cmd_str}")
        print(f"[log] {log_path}")

        if args.dry_run:
            continue

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        existing_pp = env.get("PYTHONPATH", "")
        path_parts = [str(repo_python_dir), str(examples_dir)]
        if existing_pp:
            path_parts.append(existing_pp)
        env["PYTHONPATH"] = os.pathsep.join(path_parts)

        start = time.time()
        with log_path.open("w", encoding="utf-8") as out:
            out.write(f"# COMMAND: {cmd_str}\n")
            out.write(f"# START_TS: {start}\n")
            out.flush()
            proc = subprocess.run(
                cmd,
                cwd=examples_dir,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            end = time.time()
            out.write(f"\n# END_TS: {end}\n")
            out.write(f"# DURATION_SEC: {end - start:.3f}\n")
            out.write(f"# RETURN_CODE: {proc.returncode}\n")

        if proc.returncode != 0:
            failures += 1
            print(f"[fail] return code={proc.returncode}")
            if not args.continue_on_error:
                print("Stopping on first failure. Use --continue-on-error to keep going.")
                return proc.returncode
        else:
            print(f"[ok] {time.time() - start:.2f}s")

    if args.dry_run:
        print("\nDry-run completed.")
        return 0

    print(f"\nCompleted with {failures} failure(s).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
