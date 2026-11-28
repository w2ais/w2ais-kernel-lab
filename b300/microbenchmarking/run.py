"""Run the B300 probes locally on SM103a hardware or inside the Modal image."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
# Modal mounts the reference sources beside this folder. In a checkout they live under gb10.
REFERENCE = HERE.parent if (HERE.parent / "supplemental").is_dir() else (
    HERE.parents[1] / "gb10/microbenchmarking/gb10-mma-microbench"
)
TARGET = "sm_103a"
PORTABLE = ("extended", "scheduling")
SUITES = (*PORTABLE, "tcgen05")


def build(source, binary):
    subprocess.run(["nvcc", "-O3", "-std=c++17", "-gencode", "arch=compute_103a,code=" + TARGET, "-lineinfo",
                    source, "-o", binary], check=True)


def capture_stdout(command, path):
    with path.open("w") as stream:
        subprocess.run(command, stdout=stream, check=True)


def run_suite(suite, work, out, rounds, samples):
    out.mkdir()
    source = work / "source"
    if suite in PORTABLE and not source.exists():
        for name in ("cuda", "supplemental"):
            shutil.copytree(REFERENCE / name, source / name,
                            ignore=shutil.ignore_patterns("build", "__pycache__", "results"))
        with (HERE / "scalar_and_memory.patch").open() as patch:
            subprocess.run(["patch", "--batch", "--fuzz=0", "-p1"], stdin=patch, cwd=source, check=True)
    if suite == "extended":
        data = out / "data"
        subprocess.run([sys.executable, source / "cuda/run_extended.py", "--output", data,
                        "--rounds", str(rounds), "--samples", str(samples)], check=True)
        subprocess.run([sys.executable, source / "cuda/analyze_extended.py", data], check=True)
    elif suite in ("smoke", "scheduling"):
        supplemental = (REFERENCE if suite == "smoke" else source) / "supplemental"
        names = ["shared_read"] if suite == "smoke" else ["shared_read", "warp_pairs", "warp_progress"]
        for name in names:
            exe = work / name
            build(supplemental / f"{name}.cu", exe)
            capture_stdout([exe], out / (name + (".txt" if name == "shared_read" else ".csv")))
        if suite == "scheduling":
            capture_stdout([sys.executable, supplemental / "summarize.py", out], out / "summary.txt")
    elif suite == "tcgen05":
        run_tcgen05(work, out, rounds, samples)


def run_tcgen05(work, out, rounds, samples):
    exe = work / "tcgen05"
    build(HERE / "tcgen05.cu", exe)
    capture_stdout(["cuobjdump", "--dump-sass", exe], out / "native.sass")
    sass = (out / "native.sass").read_text()
    formats = {"fp16": (16, "UTCHMMA"), "bf16": (16, "UTCHMMA"), "tf32": (8, "UTCHMMA"),
               "fp4_k32": (32, "UTCQMMA"), "mxfp4": (64, "UTCOMMA"), "nvfp4": (64, "UTCOMMA.BLOCK16")}
    functions = {int(index): body for index, body in re.findall(
        r"Function : _Z7measureIL6Format(\d+)EE[^\n]*\n(.*?)(?=Function :|\Z)", sass, re.S)}
    for index, (op, (_, opcode)) in enumerate(formats.items()):
        # One static instruction initializes D; the second is the timed loop body.
        if re.findall(r"\bUTC[HQO]MMA(?:\.BLOCK16)?\b", functions[index]) != [opcode, opcode]:
            raise RuntimeError(f"Unexpected native instructions for {op}")
    subprocess.run([exe, str(rounds), str(samples), out / "samples.csv"], check=True)
    with (out / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    summary = []
    if len(rows) != len(formats) * 4 * rounds * samples:
        raise RuntimeError("Unexpected number of native MMA samples")
    for op, (k, opcode) in formats.items():
        for batch in (1, 4, 16, 64):
            medians = []
            for rid in range(rounds):
                group = [row for row in rows if row["op"] == op and int(row["batch"]) == batch and int(row["round"]) == rid]
                if sorted(int(row["sample"]) for row in group) != list(range(samples)) or any(
                        (int(row["m"]), int(row["n"]), int(row["k"]), int(row["instructions"])) != (128, 128, k, 256)
                        or int(row["cycles"]) <= 0 for row in group):
                    raise RuntimeError(f"Incomplete native MMA samples for {op}, batch={batch}, round={rid}")
                medians.append(statistics.median(int(row["cycles"]) / int(row["instructions"]) for row in group))
            interval = statistics.mean(medians)
            summary.append(dict(op=op, batch=batch, m=128, n=128, k=k, opcode=opcode, cycles_per_mma=interval,
                                round_min=min(medians), round_max=max(medians),
                                flops_per_sm_cycle=2 * 128 * 128 * k / interval))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["smoke", "all", "portable", *SUITES])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if min(args.rounds, args.samples) < 1:
        parser.error("rounds and samples must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fields = ["name", "compute_cap", "driver_version", "memory.total"]
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                      "--format=csv,noheader,nounits"], text=True).strip()
    gpu = dict(zip(fields, map(str.strip, device.split(",")), strict=True))
    gpu["memory_MiB"] = int(gpu.pop("memory.total"))
    gpu["nvcc"] = subprocess.check_output(["nvcc", "--version"], text=True).strip()
    (out / "device.json").write_text(json.dumps(gpu, indent=2) + "\n")
    if "B300" not in gpu["name"] or gpu["compute_cap"] != "10.3":
        raise RuntimeError(f"Expected one B300 with compute capability 10.3, got {device}")
    suites = SUITES if args.suite == "all" else PORTABLE if args.suite == "portable" else [args.suite]
    paths = [HERE / name for name in ("run.py", "tcgen05.cu", "scalar_and_memory.patch")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out / "protocol.json").write_text(json.dumps({
        "target": TARGET, "nvcc_flags": ["-O3", "-std=c++17", "-gencode", "arch=compute_103a,code=sm_103a", "-lineinfo"],
        "suites": list(suites), "rounds": args.rounds, "samples_per_round": args.samples,
        "source_sha256": hashes,
    }, indent=2) + "\n")
    errors = {}
    with TemporaryDirectory() as tmp:
        for name in suites:
            print(f"Starting {name}", flush=True)
            try:
                run_suite(name, Path(tmp), out / name, args.rounds, args.samples)
            except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
                errors[name] = str(exc)
                print(f"{name}: {exc}", file=sys.stderr, flush=True)
    (out / "status.json").write_text(json.dumps({"suites": list(suites), "errors": errors}, indent=2) + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
