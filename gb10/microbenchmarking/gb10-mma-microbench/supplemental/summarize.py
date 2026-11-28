"""Summarize the three standalone GB10 probes from one capture directory."""

import argparse
import csv
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def warp_pairs(path):
    cycles = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[k]) for k in ("round", "first", "gap", "a", "b", "trips"))
            if key in cycles:
                raise ValueError(f"Duplicate warp-pair trial {key}")
            cycles[key] = int(row["cycles"])

    def cycles_per_unit(r, warp, gap, a, b):
        long_run = cycles[r, warp, gap, a, b, 1024]
        short_run = cycles[r, warp, gap, a, b, 256]
        return (long_run - short_run) / ((1024 - 256) * 8)

    names = {1: "FFMA", 2: "EX2", 3: "MMA"}
    print("Warp pair, gap, longer solo, sum of solos, pair (SM cycles per unit)")
    for a in names:
        for b in range(a, 4):
            for gap in range(1, 9):
                samples = []
                for r in range(5):
                    for warp in range(16):
                        first = cycles_per_unit(r, warp, 0, a, 0)
                        second = cycles_per_unit(r, (warp + gap) % 16, 0, b, 0)
                        pair = cycles_per_unit(r, warp, gap, a, b)
                        samples.append((max(first, second), first + second, pair))
                longer, serial, paired = map(median, zip(*samples))
                print(f"{names[a]}+{names[b]}, {gap}, {longer:.3f}, {serial:.3f}, {paired:.3f}")


def warp_progress(path):
    timestamps = defaultdict(dict)
    with path.open() as stream:
        for row in csv.reader(stream):
            run, stride, warp, checkpoint, timestamp = map(int, row)
            trace = timestamps[run, stride, warp]
            if checkpoint in trace:
                raise ValueError(f"Duplicate checkpoint {row}")
            trace[checkpoint] = timestamp

    for stride in (1, 4):
        slowest = {32: Counter(), 64: Counter()}
        for run in range(20):
            traces = []
            for warp in range(4):
                trace = [timestamps[run, stride, warp][i] for i in range(65)]
                if trace != sorted(trace):
                    raise ValueError(f"Out-of-order timestamps in run {run}, warp {warp}")
                traces.append(trace)
            for milestone, counts in slowest.items():
                first_time = min(trace[milestone] for trace in traces)
                last_checkpoint = min(bisect_right(trace, first_time) - 1 for trace in traces)
                counts[last_checkpoint] += 1
        for milestone, counts in slowest.items():
            print(f"Warp stride {stride}, leader at {milestone}, slowest checkpoint (runs): {dict(sorted(counts.items()))}")


def shared_read(path):
    cycles = {}
    for line in path.read_text().splitlines():
        run, iterations, ticks = map(int, line.split())
        if (run, iterations) in cycles:
            raise ValueError(f"Duplicate shared-read trial {line}")
        cycles[run, iterations] = ticks
    additional_cycles = median(cycles[r, 1024] - cycles[r, 256] for r in range(7))
    additional_bytes = (1024 - 256) * 512 * 4 * 16
    print(f"Shared reads: {additional_bytes:,} bytes / {additional_cycles:,} median SM cycles"
          f" = {additional_bytes / additional_cycles:.2f} bytes/SM/cycle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    directory = parser.parse_args().capture
    warp_pairs(directory / "warp_pairs.csv")
    warp_progress(directory / "warp_progress.csv")
    shared_read(directory / "shared_read.txt")
