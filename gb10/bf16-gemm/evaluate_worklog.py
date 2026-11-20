"""Independent square-GEMM evaluation rounds; arithmetic mean of TFLOP/s.

Each sample is one CUDA-event trial averaging 50 launches after 20 warmups.
Each subprocess checks correctness. Shuffle order reproducibly between rounds.
Resume only with unchanged sources/configuration; completed samples are retained.
"""

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from benchmark import KERNELS, command, source_hashes

ROOT = Path(__file__).resolve().parent


def variants(include_tuned_tma=False, size=4096):
    result = {
        key: command(key, size=size, comparison=True)
        for key in KERNELS
        if key != "5M" or include_tuned_tma
    }
    result["ampere"] = command("ampere", size=size)
    shape = ["--m", str(size), "--n", str(size), "--k", str(size)]
    result["cublaslt"] = [
        str(ROOT / "results/cublaslt_bf16_bench"),
        *shape,
        "--a-major",
        "K",
        "--b-major",
        "K",
        "--c-major",
        "N",
        "--warmup",
        "20",
        "--iterations",
        "50",
        "--trials",
        "1",
        "--heuristics",
        "64",
        "--workspace-mib",
        "256",
    ]
    return result


def parse(name, log, output):
    if name == "cublaslt":
        assert "CORRECTNESS mismatches=0" in log
        return float(re.search(r"RESULT .*? tflops=([\d.]+)", log)[1]), {}
    if name in ("5M",):
        d = json.loads(output.read_text())
        assert d["status"] == "passed"
        return d["mean_tflops"]["candidate"], {
            k: v for k, v in d["mean_tflops"].items() if k != "candidate"
        }
    d = json.loads(output.read_text())
    assert d["correctness"]["checked"] and d["correctness"]["passed"]
    timing = d["timing"]
    return timing["median_tflops"], {}


def summarize(report):
    summary = {}
    for name in report["commands"]:
        rows = [r for r in report["samples"] if r["variant"] == name]
        values = [r["tflops"] for r in rows]
        if not values:
            continue
        summary[name] = {
            "count": len(values),
            "mean_tflops": statistics.mean(values),
            "stdev_tflops": statistics.stdev(values) if len(values) > 1 else 0,
            "min_tflops": min(values),
            "max_tflops": max(values),
            "samples_tflops": values,
        }
        keys = set.intersection(*(set(r["controls"]) for r in rows))
        summary[name]["mean_controls_tflops"] = {
            k: statistics.mean(r["controls"][k] for r in rows) for k in keys
        }
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-process wall-clock limit; use 1800 for the scalar 8192-cubed benchmark",
    )
    p.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Square dimension; the complete suite currently requires a multiple of 2048",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--include-tuned-tma",
        action="store_true",
        help="Compatibility flag; Modern checkpoints 4M/5M are included by default",
    )
    p.add_argument(
        "--skip-scalar",
        action="store_true",
        help="Exclude Kernel 0 from the evaluation",
    )
    p.add_argument(
        "--kernels",
        nargs="+",
        default=[
            "0",
            "1",
            "2",
            "3A",
            "4A",
            "5A",
            "3M",
            "4M",
            "5M",
            "ampere",
            "cublaslt",
        ],
        help="Evaluate only these variant IDs, for example --kernels 0",
    )
    args = p.parse_args()
    if args.rounds < 1 or args.timeout_seconds <= 0:
        p.error("Rounds and timeout must be positive")
    if args.size <= 0 or args.size % 2048:
        p.error(
            "The complete suite requires a positive multiple of 2048; edge shapes need separate handling"
        )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commands = variants(True, args.size)
    if args.skip_scalar:
        commands.pop("0")
        args.kernels = [key for key in args.kernels if key != "0"]
    if args.kernels:
        unknown = set(args.kernels) - set(commands)
        if unknown:
            p.error(f"Unknown or excluded variants: {sorted(unknown)}")
        commands = {name: commands[name] for name in args.kernels}
    hashes = source_hashes()
    hashes["benchmarks/cublaslt_bf16_bench.cu"] = hashlib.sha256(
        (ROOT / "benchmarks/cublaslt_bf16_bench.cu").read_bytes()
    ).hexdigest()
    if "cublaslt" in commands:
        hashes["cublaslt_binary"] = hashlib.sha256(
            (ROOT / "results/cublaslt_bf16_bench").read_bytes()
        ).hexdigest()
    manifest = out / "manifest.json"
    if manifest.exists():
        report = json.loads(manifest.read_text())
        assert (
            report["sources"] == hashes
            and report["commands"] == commands
            and report["rounds"] == args.rounds
        ), "Cannot resume changed evaluation"
    else:
        report = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "rounds": args.rounds,
            "shape": [args.size] * 3,
            "precision": {
                "input": "bfloat16",
                "accumulator": "float32",
                "output": "bfloat16",
            },
            "layouts": {"A": "K-major", "B": "K-major", "C": "N-major"},
            "skipped_variants": ["0"] if args.skip_scalar else [],
            "warmups": 20,
            "iterations_per_sample": 50,
            "trials_per_process": 1,
            "timeout_seconds": args.timeout_seconds,
            "aggregation": f"Arithmetic mean of {args.rounds} independent per-process TFLOP/s samples, not throughput calculated from mean latency",
            "commands": commands,
            "sources": hashes,
            "samples": [],
            "complete": False,
        }
    for round_index in range(args.rounds):
        names = list(commands)
        random.Random(20261029 + round_index).shuffle(names)
        for name in names:
            if any(
                r["round"] == round_index + 1 and r["variant"] == name
                for r in report["samples"]
            ):
                continue
            stem = f"round_{round_index + 1:02d}_{name}"
            output = out / (stem + ".json")
            logpath = out / (stem + ".log")
            cmd = commands[name].copy()
            if name != "cublaslt":
                cmd += ["--output", str(output)]
            status = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            print(f"Round {round_index + 1}/{args.rounds} {name}", flush=True)
            with logpath.open("w") as handle:
                run = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                )
            if run.returncode:
                raise RuntimeError(f"{name} failed; see {logpath}")
            value, controls = parse(name, logpath.read_text(), output)
            assert math.isfinite(value) and value > 0
            report["samples"].append(
                {
                    "round": round_index + 1,
                    "variant": name,
                    "tflops": value,
                    "controls": controls,
                    "command": cmd,
                    "log": logpath.name,
                    "gpu_before": status.stdout.strip(),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            report["summary"] = summarize(report)
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"  {value:.3f} TFLOP/s ({len(report['samples'])}/{args.rounds * len(commands)})",
                flush=True,
            )
    report["complete"] = True
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(report["summary"], indent=2) + "\n")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
