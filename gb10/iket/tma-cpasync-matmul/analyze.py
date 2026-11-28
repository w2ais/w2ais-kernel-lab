"""Reduce IKET traces, Nsight counters and paired timing samples to one report."""

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

KERNELS = ("5A", "6A+", "5M", "6M")


def trace_stats(path, kernel, coarse):
    trace = json.loads(path.read_bytes())
    (launch,) = trace["launches"]
    locations = trace["locationTable"]
    streams = defaultdict(list)
    for marker in launch["markers"]:
        name = trace["stringTable"][marker["markerNameIdx"]]
        streams[marker["locIdx"]].append((marker["timestamp"], name))
    steps_per_tile = 128 if kernel.startswith("5") else 32
    summaries = defaultdict(list)
    for events in streams.values():
        markers = defaultdict(list)
        for stamp, name in sorted(events, key=lambda event: event[0]):
            markers[name].append(stamp)
            if name != "end_output":
                continue
            (begin,) = markers["begin_output"]
            (epilogue,) = markers["epilogue"]
            summaries["pre_epi_us"].append((epilogue - begin) / 1000)
            summaries["epi_us"].append((stamp - epilogue) / 1000)
            if not coarse:
                steps = markers["step"]
                assert len(steps) == steps_per_tile - 1
                intervals = [end - start for start, end in itertools.pairwise(steps)]
                summaries["step_us"].append(median(intervals) / 1000)
                wait = [
                    b - a
                    for a, b in zip(
                        markers["wait_begin"], markers["wait_end"], strict=True
                    )
                ]
                assert len(wait) == len(steps)
                summaries["wait_per_step_ns"].append(mean(wait))
            markers.clear()

        if not markers:
            continue
        # The producer has a separate warp stream and runs one acquisition per K-step.
        if set(markers) != {"acquire_begin", "tma_issue", "tma_issued"}:
            raise ValueError(f"Incomplete warp trace in {path}")
        producer = list(
            zip(
                markers["acquire_begin"],
                markers["tma_issue"],
                markers["tma_issued"],
                strict=True,
            )
        )
        assert len(producer) % steps_per_tile == 0
        for start in range(0, len(producer), steps_per_tile):
            steady = producer[start + 4 : start + steps_per_tile]
            summaries["producer_acquire_us"].append(
                mean(b - a for a, b, _ in steady) / 1000
            )
            summaries["producer_issue_us"].append(
                mean(c - b for _, b, c in steady) / 1000
            )
    return {
        "ctas": len({tuple(locations[i]["ctaId"]) for i in streams}),
        "compute_warp_tiles": len(summaries["epi_us"]),
        **{name: round(median(values), 4) for name, values in summaries.items()},
    }


def read_counters(path):
    with path.open() as file:
        records = csv.DictReader(line for line in file if line.startswith('"'))
        return {
            row["Metric Name"]: float(row["Metric Value"])
            for row in records
            if row["Metric Value"].lower() != "n/a"
        }


def read_instruction_counts(path):
    with path.open() as file:
        records = csv.reader(file)
        next(records)  # Kernel name
        _, _, *metrics = next(records)
        by_opcode = {metric: Counter() for metric in metrics}
        for address, instruction, *values in records:
            if not address.startswith("0x"):
                continue
            tokens = instruction.split()
            opcode = tokens[1] if tokens[0].startswith("@") else tokens[0]
            for name, value in zip(metrics, values, strict=True):
                if value.isdigit() and int(value):
                    by_opcode[name][opcode] += int(value)
    return by_opcode


def summarize(out):
    cohorts = {}
    for cohort in ("primary", "replicate", "coarse"):
        cohorts[cohort] = {}
        for kernel in KERNELS:
            (trace_file,) = (out / cohort / kernel).glob("*.trace.json")
            cohorts[cohort][kernel] = trace_stats(
                trace_file, kernel, cohort == "coarse"
            )
    rounds = json.loads((out / "control-rounds.json").read_text())
    return {
        "traces": cohorts,
        "counters": {k: read_counters(out / f"ncu-{k}.csv") for k in KERNELS},
        "instruction_counts": {
            k: {
                kind: read_instruction_counts(out / f"{kind}-{k}.csv")
                for kind in ("pcs", "memory")
            }
            for k in ("6A+", "6M", "scale_lanes")
        },
        "median_tflops": {
            version: {
                k: json.loads((out / f"time-{version}-{k}.json").read_text())[
                    "median_tflops"
                ]
                for k in KERNELS
            }
            for version in ("4.5.2", "4.6.0")
        },
        "controls_tflops_by_round": rounds,
        "controls_mean_tflops": {k: mean(values) for k, values in rounds.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    summary = out / "summary.json"
    summary.write_text(json.dumps(summarize(out), indent=2) + "\n")
    print(f"Wrote {summary}")
