"""Run numbered checkpoints serially and preserve commands, output and provenance."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from benchmark import KERNELS, command, source_hashes

ALL_KERNELS = list(KERNELS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "benchmark"], default="smoke")
    p.add_argument("--kernels", nargs="+", choices=ALL_KERNELS, default=ALL_KERNELS)
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    args = p.parse_args()
    size = args.size
    if size <= 0 or size % 2048:
        p.error("The complete suite requires a positive multiple of 2048")
    warmup, iterations, trials = (1, 1, 1) if args.mode == "smoke" else (20, 50, 5)
    out = args.output.resolve() / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    out.mkdir(parents=True)
    report = {
        "mode": args.mode,
        "size": size,
        "python": sys.version,
        "platform": platform.platform(),
        "sources": source_hashes(),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "runs": [],
    }
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (out / "nvidia-smi.txt").write_text(gpu.stdout + gpu.stderr)
    for number in args.kernels:
        cmd = command(
            number,
            size=size,
            warmups=warmup,
            iterations=iterations,
            trials=trials,
            comparison=args.mode == "benchmark",
        )
        cmd += ["--output", str(out / f"kernel_{number}.json")]
        print(f"Kernel {number}: {' '.join(cmd)}", flush=True)
        log = out / f"kernel_{number}.log"
        with log.open("w") as handle:
            result = subprocess.run(
                cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=300
            )
        report["runs"].append(
            {
                "kernel": number,
                "command": cmd,
                "returncode": result.returncode,
                "log": log.name,
            }
        )
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"  exit={result.returncode}; {log}", flush=True)
        if result.returncode:
            print(log.read_text()[-6000:])
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
