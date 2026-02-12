#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LOG_RE = re.compile(
    r"^(?P<idx>\d+)_"
    r"(?P<group>.+?)_"
    r"(?P<model>tgn|tgat|jodie|apan)_"
    r"(?P<dataset>.+?)_"
    r"(?P<mode>sync|async_only|shape_cuda_only|full)_"
    r"s(?P<seed>\d+)\.log$"
)

EPOCH_RE = re.compile(r"(?:train\s+epoch\s+time|epoch\s+time):\s*([0-9]+(?:\.[0-9]+)?)s")
TEST_RE = re.compile(r"test:\s*AP:([0-9]*\.?[0-9]+)\s+AUC:([0-9]*\.?[0-9]+)")
RC_RE = re.compile(r"# RETURN_CODE:\s*(-?\d+)")


def _parse_log_name(path: Path) -> Optional[Dict[str, str]]:
    match = LOG_RE.match(path.name)
    if not match:
        return None
    return match.groupdict()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _last_float(pattern: re.Pattern, text: str) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    last = matches[-1]
    if isinstance(last, tuple):
        return float(last[0])
    return float(last)


def _last_test(text: str) -> Tuple[Optional[float], Optional[float]]:
    matches = TEST_RE.findall(text)
    if not matches:
        return None, None
    ap, auc = matches[-1]
    return float(ap), float(auc)


def _parse_return_code(text: str) -> Optional[int]:
    matches = RC_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize rebuttal run logs")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, default=Path("examples"))
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    logs_dir = results_dir / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"logs dir not found: {logs_dir}")

    per_seed_rows: List[Dict] = []
    for log_path in sorted(logs_dir.glob("*.log")):
        parsed = _parse_log_name(log_path)
        if parsed is None:
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        epoch_sec = _last_float(EPOCH_RE, text)
        ap, auc = _last_test(text)
        rc = _parse_return_code(text)

        per_seed_rows.append(
            {
                "group": parsed["group"],
                "model": parsed["model"],
                "dataset": parsed["dataset"],
                "mode": parsed["mode"],
                "seed": int(parsed["seed"]),
                "epoch_sec": "" if epoch_sec is None else f"{epoch_sec:.6f}",
                "epoch_ms": "" if epoch_sec is None else f"{epoch_sec * 1000.0:.3f}",
                "test_ap": "" if ap is None else f"{ap:.6f}",
                "test_auc": "" if auc is None else f"{auc:.6f}",
                "return_code": "" if rc is None else str(rc),
                "log": str(log_path.relative_to(results_dir)),
            }
        )

    if not per_seed_rows:
        raise RuntimeError(f"No parseable logs found in {logs_dir}")

    by_key: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for row in per_seed_rows:
        key = (row["group"], row["model"], row["dataset"], row["mode"])
        by_key[key].append(row)

    summary_rows: List[Dict] = []
    for (group, model, dataset, mode), rows in sorted(by_key.items()):
        epoch_vals = [float(r["epoch_ms"]) for r in rows if r["epoch_ms"]]
        ap_vals = [float(r["test_ap"]) for r in rows if r["test_ap"]]
        auc_vals = [float(r["test_auc"]) for r in rows if r["test_auc"]]
        summary_rows.append(
            {
                "group": group,
                "model": model,
                "dataset": dataset,
                "mode": mode,
                "n": len(rows),
                "epoch_ms_mean": f"{_mean(epoch_vals):.3f}" if epoch_vals else "",
                "ap_mean": f"{_mean(ap_vals):.6f}" if ap_vals else "",
                "auc_mean": f"{_mean(auc_vals):.6f}" if auc_vals else "",
            }
        )

    mode_map: Dict[Tuple[str, str, str], Dict[str, Dict]] = defaultdict(dict)
    for row in summary_rows:
        key = (row["group"], row["model"], row["dataset"])
        mode_map[key][row["mode"]] = row

    model_dataset_rows: List[Dict] = []
    for (group, model, dataset), mode_rows in sorted(mode_map.items()):
        sync = mode_rows.get("sync")
        async_only = mode_rows.get("async_only")
        shape_only = mode_rows.get("shape_cuda_only")
        full = mode_rows.get("full")

        def _em(mode_row: Optional[Dict]) -> Optional[float]:
            if not mode_row:
                return None
            raw = mode_row.get("epoch_ms_mean", "")
            return float(raw) if raw else None

        def _speedup(base: Optional[float], target: Optional[float]) -> str:
            if not base or not target or target <= 0:
                return ""
            return f"{base / target:.6f}"

        sync_ms = _em(sync)
        async_ms = _em(async_only)
        shape_ms = _em(shape_only)
        full_ms = _em(full)

        model_dataset_rows.append(
            {
                "group": group,
                "model": model,
                "dataset": dataset,
                "sync_ms": "" if sync_ms is None else f"{sync_ms:.3f}",
                "async_only_ms": "" if async_ms is None else f"{async_ms:.3f}",
                "shape_cuda_only_ms": "" if shape_ms is None else f"{shape_ms:.3f}",
                "full_ms": "" if full_ms is None else f"{full_ms:.3f}",
                "async_vs_sync_x": _speedup(sync_ms, async_ms),
                "shape_vs_sync_x": _speedup(sync_ms, shape_ms),
                "full_vs_sync_x": _speedup(sync_ms, full_ms),
                "ap_sync": "" if not sync else sync.get("ap_mean", ""),
                "ap_full": "" if not full else full.get("ap_mean", ""),
                "auc_sync": "" if not sync else sync.get("auc_mean", ""),
                "auc_full": "" if not full else full.get("auc_mean", ""),
                "ap_delta_full_minus_sync": ""
                if not sync or not full or not sync.get("ap_mean") or not full.get("ap_mean")
                else f"{float(full['ap_mean']) - float(sync['ap_mean']):.6f}",
                "auc_delta_full_minus_sync": ""
                if not sync or not full or not sync.get("auc_mean") or not full.get("auc_mean")
                else f"{float(full['auc_mean']) - float(sync['auc_mean']):.6f}",
            }
        )

    runs_by_seed_path = results_dir / "runs_by_seed.csv"
    runs_summary_path = results_dir / "runs_summary.csv"
    model_dataset_path = results_dir / "model_dataset_mode_summary.csv"

    _write_csv(
        runs_by_seed_path,
        [
            "group",
            "model",
            "dataset",
            "mode",
            "seed",
            "epoch_sec",
            "epoch_ms",
            "test_ap",
            "test_auc",
            "return_code",
            "log",
        ],
        per_seed_rows,
    )
    _write_csv(
        runs_summary_path,
        ["group", "model", "dataset", "mode", "n", "epoch_ms_mean", "ap_mean", "auc_mean"],
        summary_rows,
    )
    _write_csv(
        model_dataset_path,
        [
            "group",
            "model",
            "dataset",
            "sync_ms",
            "async_only_ms",
            "shape_cuda_only_ms",
            "full_ms",
            "async_vs_sync_x",
            "shape_vs_sync_x",
            "full_vs_sync_x",
            "ap_sync",
            "ap_full",
            "auc_sync",
            "auc_full",
            "ap_delta_full_minus_sync",
            "auc_delta_full_minus_sync",
        ],
        model_dataset_rows,
    )

    print(f"wrote {runs_by_seed_path}")
    print(f"wrote {runs_summary_path}")
    print(f"wrote {model_dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
