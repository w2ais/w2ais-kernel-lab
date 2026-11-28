"""Capture arithmetic, memory, synchronization, mixed-loop and copy probes."""
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
FIELDS = ["family", "op", "warps", "chains", "size", "depth", "iterations", "samples", "variant"]
ALU = ["ffma", "dfma", "hfma2", "rcp", "exp2"]
COPY = ["vector", "cp_async", "bulk", "tma2d", "store"]


def text(command):
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True).stdout


def plan(path, rows):
    with path.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def configurations(samples):
    rows = set()
    for op in ALU:
        for warps in [1, 2, 4, 8, 16, 32]:
            for chains in [1, 2, 4, 8]:
                rows.add(("alu", op, warps, chains, 0, 0, 1024, samples, 0))
        for length in [256, 4096]:
            rows.add(("alu", op, 1, 1, 0, 0, length, samples, 0))
    for op in ["ffma", "exp2"]:
        for ratio in [1, 4, 8]:
            for mode in [0, 1, 2]:
                rows.add(("mixed", op, 16, 4, 0, ratio, 1024, samples, mode))
    for size in [4096, 65536, 1048576, 8388608, 67108864, 268435456]:
        for policy in ["ca", "cg"]:
            # Every timed interval traverses the whole random cycle, plus 224
            # nodes. Repeats cannot accidentally time only a cache-sized prefix.
            iterations = max(size // 128, 65536) // 32 + 7
            rows.add(("memory", policy, 1, 1, size, 0, iterations, samples, 0))
    for stride in [0, 1, 2, 4, 8, 16, 32]:
        for length in [128, 512, 2048]:
            rows.add(("shared", "lds", 1, 1, 4096, stride, length, samples, 0))
    for length in [256, 1024, 4096]:
        for warps in [1, 4, 8, 16, 32]:
            rows.add(("sync", "cta", warps, 1, 0, 0, length, samples, 0))
        for op in ["mbarrier", "fence"]:
            rows.add(("sync", op, 1, 1, 0, 0, length, samples, 0))
    for op in COPY:
        for size in [1024, 4096, 8192]:
            for depth in [1, 2, 4, 8]:
                rows.add(("copy", op, 4, 1, size, depth, 128, samples, 0))
    return sorted(rows)


def validation_cases():
    rows = [("alu", op, w, n, 0, 0, 32, 1, v) for op in ALU
            for w, n in [(1, 1), (16, 4)] for v in [0, 1]]
    rows += [("mixed", op, 16, 4, 0, 4, 32, 1, mode)
             for op in ["ffma", "exp2"] for mode in [0, 1, 2]]
    rows += [("memory", op, 1, 1, 1048576, 0, 519, 1, 0) for op in ["ca", "cg"]]
    rows += [("shared", "lds", 1, 1, 4096, stride, 128, 1, 0) for stride in [0, 1, 2, 4, 8, 16, 32]]
    rows += [("sync", op, w, 1, 0, 0, 32, 1, 0) for op, w in [("cta", 1), ("cta", 16), ("mbarrier", 1), ("fence", 1)]]
    rows += [("copy", op, 4, 1, size, depth, 8, 1, 0) for op in COPY for size, depth in [(1024, 1), (8192, 8)]]
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.rounds < 1 or args.samples < 1:
        parser.error("rounds and samples must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (ROOT / "build").mkdir(exist_ok=True)
    command = ["nvcc", "-std=c++17", "-O3", "-gencode", "arch=compute_121a,code=sm_121a",
               "-lineinfo", "-Xptxas=-v", "probes/extended.cu", "-lcuda", "-o", "build/extended"]
    with (out / "build.log").open("w") as f:
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=f, check=True)
    sass = text(["cuobjdump", "--dump-sass", "build/extended"])
    with gzip.open(out / "sass.txt.gz", "wt") as f:
        f.write(sass)
    audit = []
    for section in sass.split("Function : ")[1:]:
        lines = section.splitlines()
        symbol = lines[0].strip()
        instructions = [line.strip() for line in lines if re.search(r"/\*[0-9a-f]+\*/\s+", line)]
        entry = {"symbol": symbol, "static_instructions": len(instructions),
                 "local_load_store": [line for line in instructions if re.search(r"\b(LDL|STL)\b", line)]}
        match = re.search(r"arithmeticIL3Alu(\d+)ELi(\d+)E", symbol)
        if match:
            op, chains = map(int, match.groups())
            mnemonic = [r"FFMA", r"DFMA", r"HFMA2", r"MUFU.RCP", r"MUFU.EX2"][op]
            # Exclude initialization and final checksum arithmetic. Identify
            # the actual backward loop edge in SASS, not just total opcodes.
            addressed = [(int(re.search(r"/\*([0-9a-f]+)\*/", line).group(1),16), line)
                         for line in instructions]
            loops = []
            for address, line in addressed:
                target = re.search(r"\bBRA(?:\.U)?\s+(?:[^;]*?,\s*)?(0x[0-9a-f]+)\s*;", line)
                if target and int(target.group(1),16) < address:
                    start = int(target.group(1),16)
                    loops.append([line for addr,line in addressed if start<=addr<=address])
            if len(loops)!=1:
                raise RuntimeError(f"Expected one arithmetic loop: {symbol}, found {len(loops)}")
            hits = [line for line in loops[0] if re.search(r"\b"+mnemonic+r"\b", line)]
            entry.update({"operation": ALU[op], "chains": chains,
                          "main_opcode_count": len(hits), "example": hits[0] if hits else None})
            if len(hits) != 16*chains:
                raise RuntimeError(f"Unexpected {mnemonic} count: {entry}")
            if op == 3:
                additions = sum(bool(re.search(r"\bFADD\b", line)) for line in loops[0])
                if additions != 16*chains:
                    raise RuntimeError(f"Missing reciprocal recurrence additions: {symbol}")
                entry["recurrence_additions"] = additions
        if entry["local_load_store"]:
            raise RuntimeError(f"Spilled probe: {symbol}")
        audit.append(entry)
    (out / "sass-audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    clock = ["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw", "--format=csv"]
    manifest = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "device": json.loads(text(["build/extended", "--info"])),
                "platform": platform.platform(), "python": platform.python_version(),
                "nvcc": text(["nvcc", "--version"]), "compile_command": command,
                "rounds": args.rounds, "samples": args.samples, "seed": args.seed,
                "warmups": 2, "clock_control": "unchanged, unlocked", "round_clock_samples": [],
                "primary_alu_configuration": {"warps": 16, "chains": 4, "iterations": 1024},
                "source_sha256": {},
                "protocol": "fresh process per round; serial randomized configurations; one CTA; "
                            "validate every timed launch; separate profiling and sanitizers"}
    for path in [ROOT / "run_extended.py", ROOT / "ops.cuh", *sorted((ROOT / "probes").glob("*"))]:
        manifest["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["executable_sha256"] = hashlib.sha256((ROOT / "build/extended").read_bytes()).hexdigest()
    manifest["processes_before"] = text(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if manifest["processes_before"].strip():
        raise RuntimeError("GPU has a compute process; wait for an idle device.")
    plan(out / "validation-plan.csv", validation_cases())
    with (out / "validation.csv").open("w") as f:
        subprocess.run(["build/extended", str(out / "validation-plan.csv")], cwd=ROOT, stdout=f, check=True)
    manifest["validation_cases"] = len(validation_cases())
    for rid in range(1, args.rounds+1):
        rows = configurations(args.samples)
        random.Random(args.seed+rid).shuffle(rows)
        filename = out / f"round-{rid}-plan.csv"
        plan(filename, rows)
        before = text(clock)
        print(f"Round {rid}/{args.rounds}: {len(rows)} configurations", flush=True)
        with (out / f"round-{rid}.csv").open("w") as f:
            subprocess.run(["build/extended", str(filename)], cwd=ROOT, stdout=f, check=True)
        manifest["round_clock_samples"].append({"round": rid, "before": before, "after": text(clock)})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Captured {out}", flush=True)


if __name__ == "__main__":
    main()
