"""Build, validate, and capture serial MMA experiments. No third-party packages."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import platform
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OPS = ["fp16_k8", "fp16", "bf16", "tf32", "fp4_k32", "mxfp4", "nvfp4"]
FIELDS = ["op", "warps", "chains", "unroll", "iterations", "samples", "variant"]


def capture(args: list[str]) -> str:
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout


def write_plan(path: Path, rows: list[tuple]) -> None:
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.samples < 1:
        parser.error("rounds and samples must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite a capture.
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    exe = build / "bench"
    compile_command = ["nvcc", "-std=c++17", "-O3", "-gencode",
                       "arch=compute_121a,code=sm_121a", "-lineinfo", "-Xptxas=-v",
                       "bench.cu", "-o", str(exe)]
    if not args.skip_build:
        with (output / "build.log").open("w") as f:
            subprocess.run(compile_command, cwd=ROOT, stdout=f, stderr=f, check=True)
    if not exe.is_file():
        raise FileNotFoundError(exe)
    sass = capture(["cuobjdump", "--dump-sass", str(exe)])
    with gzip.open(output / "sass.txt.gz", "wt") as f:
        f.write(sass)
    # Independent evidence of what was compiled; do not infer instruction count
    # just from the PTX source. The full disassembly is retained for review.
    sections = sass.split("Function : ")[1:]
    audit = []
    for section in sections:
        lines = section.splitlines()
        math = [line.strip() for line in lines if re.search(r"\b[HOQ]MMA\.", line)]
        template = re.search(r"L2Op(\d+)ELi(\d+)ELi(\d+)E", lines[0])
        if not template:
            raise RuntimeError(f"Unrecognized kernel symbol: {lines[0]}")
        _, chains, unroll = map(int, template.groups())
        if len(math) != chains * unroll:
            raise RuntimeError(f"Unexpected MMA count in {lines[0]}: {len(math)}")
        audit.append({"symbol": lines[0].strip(), "static_mma_instructions": len(math),
                      "first_mma": math[0] if math else None,
                      "local_load_store": [line.strip() for line in lines
                                           if " LDL" in line or " STL" in line]})
    (output / "sass-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    clock_cmd = ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,"
                 "clocks.current.sm,clocks.max.sm,clocks.default_applications.graphics,"
                 "utilization.gpu,power.draw", "--format=csv"]
    manifest = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "platform": platform.platform(), "python": platform.python_version(),
                "device": json.loads(capture([str(exe), "--info"])),
                "nvcc": capture(["nvcc", "--version"]),
                "compile_command": compile_command, "clock_control": "unchanged, unlocked",
                "rounds": args.rounds, "samples_per_point_per_round": args.samples,
                "seed": args.seed, "source_sha256": {}, "round_clock_samples": [],
                "protocol": "single CTA; 3 warmups per point; serial randomized points; "
                            "fresh process each round; all lanes checked each timed launch",
                "primary_throughput_configuration": {"warps": 16, "chains": 4,
                                                       "unroll": 16, "iterations": 2048}}
    for name in ["bench.cu", "ops.cuh", "run.py", "pyproject.toml"]:
        manifest["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    manifest["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    manifest["processes_before"] = capture(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if manifest["processes_before"].strip():
        raise RuntimeError("GPU has a compute process; rerun when idle. See nvidia-smi.")
    # Check all instruction variants, two input values, dependent and independent
    # fragments before collecting the full sweep.
    validation = [(op, w, n, 16, 128, 1, value) for op in OPS
                  for w, n in [(1, 1), (16, 4)] for value in [0, 1]]
    write_plan(output / "validation-plan.csv", validation)
    with (output / "validation.csv").open("w") as f:
        subprocess.run([str(exe), str(output / "validation-plan.csv")], stdout=f, cwd=ROOT, check=True)
    base = {(op, w, n, 16, 2048, args.samples, 0) for op in OPS
            for w in [1, 2, 4, 8, 16, 32] for n in [1, 2, 4, 8]}
    for op in OPS:
        # Length sweep separates a per-instruction slope from fixed timing costs.
        for iters in [256, 1024, 4096]:
            base.add((op, 1, 1, 16, iters, args.samples, 0))
            base.add((op, 16, 4, 16, iters, args.samples, 0))
        # Same work, much less unrolling; sensitivity is reported, not hidden.
        for w, n in [(1, 1), (16, 4)]:
            base.add((op, w, n, 1, 32768, args.samples, 0))
            base.add((op, w, n, 16, 2048, args.samples, 1))
    for round_id in range(args.rounds):
        plan = sorted(base)
        random.Random(args.seed + round_id).shuffle(plan)
        path = output / f"round-{round_id + 1}-plan.csv"
        write_plan(path, plan)
        before = capture(clock_cmd)
        print(f"Round {round_id + 1}/{args.rounds}: {len(plan)} configurations", flush=True)
        with (output / f"round-{round_id + 1}.csv").open("w") as f:
            subprocess.run([str(exe), str(path)], stdout=f, cwd=ROOT, check=True)
        manifest["round_clock_samples"].append({"round": round_id + 1,
                                               "before": before, "after": capture(clock_cmd)})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {output}", flush=True)


if __name__ == "__main__":
    main()
