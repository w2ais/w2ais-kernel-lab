"""Aggregate raw captures without picking the fastest configuration or sample."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
from evidence import verify_sources

KEYS = ("op", "warps", "chains", "unroll", "iterations", "variant")


def fit(xs: list[float], ys: list[float]) -> dict:
    xm, ym = stats.mean(xs), stats.mean(ys)
    slope = sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / sum((x-xm)**2 for x in xs)
    intercept = ym - slope*xm
    residual = sum((y-(intercept+slope*x))**2 for x, y in zip(xs, ys))
    total = sum((y-ym)**2 for y in ys)
    return {"slope": slope, "intercept": intercept, "r_squared": 1-residual/total}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    root = args.capture
    manifest = json.loads((root / "manifest.json").read_text())
    verify_sources(root, manifest)
    assert "finished_utc" in manifest, "Capture is incomplete"
    points = defaultdict(list)
    total_samples = 0
    for round_id in range(1, manifest["rounds"] + 1):
        rows = list(csv.DictReader((root / f"round-{round_id}.csv").open()))
        plan = list(csv.DictReader((root / f"round-{round_id}-plan.csv").open()))
        assert len(rows) == sum(int(p["samples"]) for p in plan)
        by_config = defaultdict(list)
        for row in rows:
            assert int(row["local_bytes"]) == 0, "Spilled variant must not be used as a register-only benchmark"
            count = int(row["warps"])*int(row["chains"])*int(row["unroll"])*int(row["iterations"])
            assert count == int(row["mmas"])
            assert abs(float(row["mma_per_sm_cycle"])-count/int(row["cycles"])) < 1e-8
            by_config[tuple(row[k] for k in KEYS)].append(row)
        assert set(by_config) == {tuple(row[k] for k in KEYS) for row in plan}
        for key, samples in by_config.items():
            assert sorted(int(s["sample"]) for s in samples) == list(range(manifest["samples_per_point_per_round"]))
            assert len(samples) == manifest["samples_per_point_per_round"]
            cycles = stats.median(int(r["cycles"]) for r in samples)
            points[key].append({"round": round_id, "cycles": cycles,
                                "min_cycles": min(int(r["cycles"]) for r in samples),
                                "max_cycles": max(int(r["cycles"]) for r in samples),
                                "mmas": int(samples[0]["mmas"]),
                                "flops_per_mma": int(samples[0]["flops_per_mma"]),
                                "registers_per_thread": int(samples[0]["registers_per_thread"]),
                                "shared_bytes": int(samples[0]["shared_bytes"]),
                                "sm_ids": sorted({int(r["sm_id"]) for r in samples})})
        total_samples += len(rows)
    aggregate = []
    for key, rounds in sorted(points.items()):
        assert len(rounds) == manifest["rounds"]
        cycles = stats.mean(r["cycles"] for r in rounds)
        mmas = rounds[0]["mmas"]
        aggregate.append({**dict(zip(KEYS, key)), "mean_round_median_cycles": cycles,
                          "sm_cycles_per_mma": cycles/mmas,
                          "mma_per_sm_cycle": mmas/cycles,
                          "flops_per_sm_cycle": mmas/cycles*rounds[0]["flops_per_mma"],
                          "rounds": rounds})
    summary = []
    for op in ["fp16_k8", "fp16", "bf16", "tf32", "fp4_k32", "mxfp4", "nvfp4"]:
        fixed = next(r for r in aggregate if tuple(r[k] for k in KEYS) == (op,"16","4","16","2048","0"))
        regression = []
        for rid in range(manifest["rounds"]):
            chain = sorted((int(key[4])*16, rounds[rid]["cycles"]) for key, rounds in points.items()
                           if key[0]==op and key[1:4]==("1","1","16") and key[5]=="0")
            regression.append(fit([x for x,y in chain], [y for x,y in chain]))
        summary.append({"op": op, "flops_per_mma": fixed["rounds"][0]["flops_per_mma"],
                        "dependency_chain_cycles_per_mma": stats.mean(r["slope"] for r in regression),
                        "recurrence_fits": regression,
                        "sm_cycles_per_mma": fixed["sm_cycles_per_mma"],
                        "mma_per_sm_cycle": fixed["mma_per_sm_cycle"],
                        "flops_per_sm_cycle": fixed["flops_per_sm_cycle"],
                        "round_min_sm_cycles_per_mma": min(r["cycles"]/r["mmas"] for r in fixed["rounds"]),
                        "round_max_sm_cycles_per_mma": max(r["cycles"]/r["mmas"] for r in fixed["rounds"]),
                        "registers_per_thread": fixed["rounds"][0]["registers_per_thread"],
                        "shared_bytes": fixed["rounds"][0]["shared_bytes"]})
    report = {"method": "Median cycles within each round, arithmetic mean of those medians; "
                        "divide fixed work by mean cycles. Dependency slope: OLS separately "
                        "in each round, then mean slopes. No best-sample selection.",
              "total_samples": total_samples, "configurations": len(points),
              "summary": summary, "points": aggregate}
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [key for key in summary[0] if key != "recurrence_fits"]
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"{total_samples} checked samples across {len(points)} configurations")
    for row in summary:
        print(f'{row["op"]:9s} recurrence={row["dependency_chain_cycles_per_mma"]:.4f} '
              f'SM cycles/MMA={row["sm_cycles_per_mma"]:.6f} '
              f'MMA/SM/cycle={row["mma_per_sm_cycle"]:.6f} '
              f'FLOP/SM/cycle={row["flops_per_sm_cycle"]:.3f}')


if __name__ == "__main__":
    main()
