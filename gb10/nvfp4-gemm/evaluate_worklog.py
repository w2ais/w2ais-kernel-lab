"""Run serial, shuffled 4096³ rounds and use one common cuBLAS mean."""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from benchmark import KERNELS

ROOT = Path(__file__).resolve().parent


def run(command, log):
    with log.open("w") as f:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--output", type=Path, default=ROOT / "results/evaluation")
    args = p.parse_args()
    if args.rounds < 1:
        p.error("rounds must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        p.error("choose a new output directory; existing results are immutable")
    binary = out / "cublas_nvfp4"
    build = [
        "/usr/local/cuda/bin/nvcc",
        "-O3",
        "-std=c++17",
        "references/cublaslt_nvfp4_bench.cu",
        "-lcublasLt",
        "-o",
        str(binary),
    ]
    run(build, out / "cublas-build.log")
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "shape": [4096] * 3,
        "precision": {"input": "NVFP4 E2M1", "scales": "UE4M3 per 16 K values", "accumulator": "float32", "output": "bfloat16"},
        "rounds": args.rounds,
        "warmups": 20,
        "iterations": 50,
        "order_seed": 20261029,
        "method": "Arithmetic mean of independent process throughputs; common cuBLAS mean denominator; serial CUDA-event measurements; clocks unlocked.",
        "timed_input": "All A/B/scales = 1 for both implementations; custom kernels also pass full-output random E2M1/scales validation.",
        "nvcc": subprocess.check_output(
            ["/usr/local/cuda/bin/nvcc", "--version"], text=True
        ),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        "build_command": build,
        "runs": [],
        "means": {},
    }
    samples = {k: [] for k in [*KERNELS, "cuBLAS"]}
    rng = random.Random(report["order_seed"])
    for round_index in range(args.rounds):
        order = list(samples)
        rng.shuffle(order)
        for key in order:
            target = out / f"round-{round_index + 1}-{key}.json"
            if key == "cuBLAS":
                command = [
                    str(binary),
                    "4096",
                    "4096",
                    "4096",
                    "--warmups",
                    "20",
                    "--iterations",
                    "50",
                    "--trials",
                    "1",
                    "--device-warmups",
                    "100",
                    "--search-iterations",
                    "20",
                    "--search-sweeps",
                    "1",
                    "--json",
                    str(target),
                ]
            else:
                command = [
                    sys.executable,
                    "benchmark.py",
                    key,
                    "--seed",
                    str(2026 + round_index),
                    "--output",
                    str(target),
                ]
            print(f"Round {round_index + 1}/{args.rounds}: {key}", flush=True)
            run(command, target.with_suffix(".log"))
            result = json.loads(target.read_text())
            if key == "cuBLAS":
                assert result["validation"]["correct"]
                value = result["timing"][
                    "median_tflops"
                ]  # Exactly one trial per process.
            else:
                assert result["correctness"]["passed"]
                value = result["mean_tflops"]
            samples[key].append(value)
            report["runs"].append(
                {
                    "round": round_index + 1,
                    "kernel": key,
                    "tflops": value,
                    "file": target.name,
                    "command": command,
                }
            )
            (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"  {value:.3f} TFLOP/s", flush=True)
    baseline = statistics.mean(samples["cuBLAS"])
    report["means"] = {
        k: {
            "tflops": statistics.mean(v),
            "percent_cublas": 100 * statistics.mean(v) / baseline,
            "samples": v,
        }
        for k, v in samples.items()
    }
    report["sources"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *ROOT.glob("*.py"),
            *ROOT.glob("references/*.py"),
            ROOT / "references/cublaslt_nvfp4_bench.cu",
            ROOT / "uv.lock",
        ]
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["means"], indent=2), flush=True)


if __name__ == "__main__":
    main()
