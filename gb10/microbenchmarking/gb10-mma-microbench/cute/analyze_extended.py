"""Check and aggregate the extended capture without selecting fastest samples."""
import argparse
import csv
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from evidence import verify_sources

from analyze import fit

KEYS = ("family", "op", "warps", "chains", "size", "depth", "iterations", "variant")
ROOT = Path(__file__).resolve().parent


def analyze(capture):
    manifest = json.loads((capture / "manifest.json").read_text())
    assert "finished_utc" in manifest, "Capture is incomplete"
    verify_sources(capture, manifest)
    points = defaultdict(list)
    count = 0
    for rid in range(1, manifest["rounds"] + 1):
        rows = list(csv.DictReader((capture / f"round-{rid}.csv").open()))
        groups = defaultdict(list)
        plan = list(csv.DictReader((capture / f"round-{rid}-plan.csv").open()))
        assert len(rows) == sum(int(p["samples"]) for p in plan)
        for row in rows:
            assert int(row["local_bytes"]) == 0, "Local-memory traffic invalidates these probes"
            assert int(row["cycles"]) > 0
            assert math.isfinite(float(row["max_abs_error"]))
            assert float(row["max_abs_error"]) < .001
            family = row["family"]
            n, w, it = [int(row[k]) for k in ("chains", "warps", "iterations")]
            work = {"alu": w*n*16*it, "mixed": 16*4*16*it,
                    "memory": 32*it, "shared": 32*it, "sync": 16*it,
                    "copy": int(row["size"])*int(row["depth"])*it}[family]
            assert work == int(row["work"])
            groups[tuple(row[k] for k in KEYS)].append(row)
        assert set(groups) == {tuple(row[k] for k in KEYS) for row in plan}
        for key, samples in groups.items():
            assert len(samples) == manifest["samples"]
            assert sorted(int(s["sample"]) for s in samples) == list(range(manifest["samples"]))
            points[key].append({"round": rid,
                "cycles": stats.median(int(s["cycles"]) for s in samples),
                "nanoseconds": stats.median(int(s["nanoseconds"]) for s in samples),
                "min_cycles": min(int(s["cycles"]) for s in samples),
                "max_cycles": max(int(s["cycles"]) for s in samples),
                "work": int(samples[0]["work"]), "unit": samples[0]["unit"],
                "registers_per_thread": int(samples[0]["registers_per_thread"]),
                "shared_bytes": int(samples[0]["shared_bytes"]),
                "max_abs_error": max(float(s["max_abs_error"]) for s in samples)})
        count += len(rows)
    aggregate = []
    for key, rounds in sorted(points.items()):
        assert len(rounds) == manifest["rounds"]
        cycles = stats.mean(r["cycles"] for r in rounds)
        work = rounds[0]["work"]
        aggregate.append({**{k: (v if k in ("family", "op") else int(v)) for k, v in zip(KEYS, key)},
                          "mean_cycles": cycles, "cycles_per_work": cycles/work,
                          "work_per_cycle": work/cycles,
                          "ns_per_work": stats.mean(r["nanoseconds"] for r in rounds)/work,
                          "round_min_cycles_per_work": min(r["cycles"]/work for r in rounds),
                          "round_max_cycles_per_work": max(r["cycles"]/work for r in rounds),
                          "unit": rounds[0]["unit"], "rounds": rounds})

    def select(**criteria):
        return [p for p in aggregate if all(p[k] == v for k, v in criteria.items())]

    def recurrence(**criteria):
        data = select(**criteria)
        assert len(data) == 3, criteria
        regressions = [fit([p["rounds"][rid]["work"] for p in data],
                           [p["rounds"][rid]["cycles"] for p in data])
                       for rid in range(manifest["rounds"])]
        return {"cycles_per_step": stats.mean(r["slope"] for r in regressions),
                "fits": regressions}

    scalar = []
    for op in ["ffma", "hfma2", "dfma", "exp2", "rcp"]:
        p, = select(family="alu", op=op, warps=16, chains=4, iterations=1024)
        lanes = 64 if op == "hfma2" else 32
        scalar.append({"op": op, "recurrence": recurrence(family="alu", op=op, warps=1, chains=1),
                       "sm_cycles_per_step": p["cycles_per_work"],
                       "warp_steps_per_sm_cycle": p["work_per_cycle"],
                       "results_per_sm_cycle": lanes*p["work_per_cycle"],
                       "flops_per_sm_cycle": 2*lanes*p["work_per_cycle"] if op in ["ffma","hfma2","dfma"] else None,
                       "registers_per_thread": p["rounds"][0]["registers_per_thread"],
                       "shared_bytes": p["rounds"][0]["shared_bytes"]})
    shared = [{"stride": s, **recurrence(family="shared", depth=s)} for s in [0,1,2,4,8,16,32]]
    sync = [{"op": op, "warps": w, **recurrence(family="sync", op=op, warps=w)}
            for op, w in [("cta",w) for w in [1,4,8,16,32]]+[("mbarrier",1),("fence",1)]]
    mixed = []
    for op in ["ffma", "exp2"]:
        for ratio in [1,4,8]:
            cases = {p["variant"]: p for p in select(family="mixed", op=op, depth=ratio)}
            mma, scalar_time, both = [cases[m]["cycles_per_work"] for m in [0,1,2]]
            mixed.append({"op": op, "scalar_per_mma": ratio, "mma_only": mma,
                          "scalar_only": scalar_time, "mixed": both,
                          "serial_sum": mma+scalar_time, "ideal_overlap": max(mma, scalar_time),
                          "mixed_vs_mma": both/mma, "mixed_vs_serial": both/(mma+scalar_time)})
    report = {"method": "Median cycles per round, then mean of round medians. Rates divide fixed work by mean cycles. "
                        "Recurrence: three-length OLS within each round, then mean slopes. "
                        "No fastest-sample or fastest-configuration selection.",
              "total_samples": count, "configurations": len(points), "scalar": scalar,
              "shared": shared, "sync": sync, "mixed": mixed, "points": aggregate}
    (capture / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    fields = [k for k in aggregate[0] if k != "rounds"]
    with (capture / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(aggregate)
    print(f"{count} checked samples across {len(points)} configurations")
    for row in scalar:
        print(f'{row["op"]:6}: recurrence {row["recurrence"]["cycles_per_step"]:.3f}; '
              f'{row["results_per_sm_cycle"]:.3f} scalar results/SM/cycle')
    for family in ["shared", "sync", "mixed"]:
        print(family, json.dumps([{k:v for k,v in p.items() if k != "fits"} for p in report[family]]))
    for p in select(family="memory"):
        print(f'pointer {p["op"]} {p["size"]}: {p["cycles_per_work"]:.3f} cycles, {p["ns_per_work"]:.3f} ns')
    for p in select(family="copy", size=4096):
        print(f'copy {p["op"]:8} D={p["depth"]}: {p["work_per_cycle"]:.3f} bytes/SM/cycle')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    analyze(parser.parse_args().capture)
