"""Task-scale-aware comparison of corrected evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def task_returns(summary: dict[str, Any]) -> dict[str, float]:
    entries = summary.get("evaluation")
    if not isinstance(entries, list):
        raise ValueError("summary must contain an 'evaluation' list")
    result: dict[str, float] = {}
    for entry in entries:
        task = entry.get("task")
        value = entry.get("mean_return")
        if not isinstance(task, str) or not isinstance(value, (int, float)):
            raise ValueError("each evaluation entry needs task and numeric mean_return")
        result[task] = float(value)
    return result


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare matched task returns using baseline absolute return as its scale."""
    reference = task_returns(baseline)
    observed = task_returns(candidate)
    if reference.keys() != observed.keys():
        raise ValueError("baseline and candidate must cover the same tasks")
    rows = []
    for task in sorted(reference):
        delta = observed[task] - reference[task]
        scale = max(abs(reference[task]), 1.0)
        rows.append(
            {
                "task": task,
                "baseline_return": reference[task],
                "candidate_return": observed[task],
                "raw_delta": delta,
                "normalised_delta": delta / scale,
            }
        )
    return {
        "raw_delta_mean": float(np.mean([row["raw_delta"] for row in rows])),
        "normalised_delta_mean": float(np.mean([row["normalised_delta"] for row in rows])),
        "per_task": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare(baseline, candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
