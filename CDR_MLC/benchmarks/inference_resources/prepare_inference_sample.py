"""Prepare a fixed Medium inference sample with causal warm-up context.

Only the final ``--sample-size`` rows marked ``benchmark_scored=True`` are
evaluated.  Earlier rows in each selected sequence are retained solely to
initialize causal moving-window features.  The sample is deterministic and is
shared by every method.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import DEFAULT_CANDIDATES, load_dataset  # noqa: E402
from common import frame_sha256, record_id, write_json  # noqa: E402


def _allocate(capacities: list[int], total: int) -> list[int]:
    if total > sum(capacities):
        raise ValueError(
            f"requested {total} scored rows but only {sum(capacities)} are available"
        )
    allocation = [0] * len(capacities)
    remaining = total
    while remaining:
        progressed = False
        for index, capacity in enumerate(capacities):
            if allocation[index] < capacity and remaining:
                allocation[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("unable to allocate inference sample")
    return allocation


def prepare(frame: pd.DataFrame, sample_size: int, context: int):
    if sample_size < 1 or context < 1:
        raise ValueError("sample size and context must be positive")
    groups = []
    for sequence_id, group in frame.groupby("sequence_id", sort=True):
        ordered = group.sort_values(["timestamp", "source_row"], kind="stable")
        capacity = max(0, len(ordered) - (context - 1))
        if capacity:
            groups.append((str(sequence_id), ordered, capacity))
    if not groups:
        raise ValueError("Medium has no sequence with sufficient causal context")
    allocation = _allocate([item[2] for item in groups], sample_size)

    selected, audit = [], []
    for (sequence_id, group, _), count in zip(groups, allocation):
        if count == 0:
            continue
        scored_start = len(group) - count
        input_start = max(0, scored_start - (context - 1))
        block = group.iloc[input_start:].copy()
        block["benchmark_scored"] = False
        block.iloc[-count:, block.columns.get_loc("benchmark_scored")] = True
        block["benchmark_sequence"] = sequence_id
        selected.append(block)
        audit.append({
            "sequence_id": sequence_id,
            "available_rows": int(len(group)),
            "warmup_rows": int(len(block) - count),
            "scored_rows": int(count),
            "first_input_record": record_id(block.iloc[[0]])[0],
            "first_scored_record": record_id(block.iloc[[-count]])[0],
            "last_scored_record": record_id(block.iloc[[-1]])[0],
        })
    result = pd.concat(selected, ignore_index=True)
    if int(result.benchmark_scored.sum()) != sample_size:
        raise RuntimeError("prepared sample size differs from request")
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument(
        "--max-context-window", type=int, default=50,
        help="largest causal window used by any compared method",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "artifacts/medium_inference_sample.pkl",
    )
    args = parser.parse_args()

    data, input_audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    medium = data[data.congestion_level.eq("Medium")].copy()
    sample, sequences = prepare(
        medium, args.sample_size, args.max_context_window
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_pickle(args.output)
    input_audit.to_csv(args.output.with_name("input_audit.csv"), index=False)
    scored = sample[sample.benchmark_scored]
    write_json(args.output.with_suffix(".json"), {
        "data_dir": str(args.data_dir),
        "level": "Medium",
        "sample_file": str(args.output),
        "input_rows_including_warmup": int(len(sample)),
        "scored_rows": int(len(scored)),
        "max_context_window": args.max_context_window,
        "sample_identity_sha256": frame_sha256(sample),
        "scored_identity_sha256": frame_sha256(scored),
        "class_counts": scored.traffic_label.value_counts().sort_index().to_dict(),
        "sequences": sequences,
    })
    print(
        f"prepared {len(scored)} scored Medium rows and "
        f"{len(sample) - len(scored)} causal warm-up rows -> {args.output}"
    )


if __name__ == "__main__":
    main()
