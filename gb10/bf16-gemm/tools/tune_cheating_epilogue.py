"""Screen stmatrix output tiles while keeping NVIDIA's mainloop unchanged.

Candidate scores are tuning data, not the published comparison. Validate a
selected configuration in fresh shuffled rounds with the cuBLAS control.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TILES = [(64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)]


def sample(tile, output):
    import torch
    import cutlass

    output.parent.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "kernel_bf16_cheating.py").read_text()
    source, replacements = re.subn(
        r"EPI_TILE = \(\d+, \d+\)", f"EPI_TILE = {tile!r}", source
    )
    assert replacements == 1
    source = source.replace(
        "HERE = Path(__file__).resolve().parent", f"HERE = Path({str(ROOT)!r})"
    )
    generated = output.with_suffix(".generated.py")
    source = source.replace(
        'Path("/tmp/gb10_nvidia_ampere_stmatrix_generated.py")',
        f"Path({str(generated)!r})",
    )
    transform = output.with_suffix(".transform.py")
    transform.write_text(source)
    spec = importlib.util.spec_from_file_location("candidate", transform)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(2026)
    us = module.run(
        "k",
        "k",
        "n",
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.Float32,
        (4096, 4096, 4096, 1),
        (2, 2, 1),
        20,
        100,
        False,
        False,
    )
    record = {
        "tile": tile,
        "tflops": 2 * 4096**3 / (us * 1e6),
        "correctness": "passed",
        "mean_us": us,
        "mainloop_unchanged": module.ORIGINAL_MAINLOOP_SHA256
        == module.TRANSFORMED_MAINLOOP_SHA256,
        "generated_source_sha256": module.GENERATED_SOURCE_SHA256,
        "transform_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    output.write_text(json.dumps(record, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tile", help="Internal single-sample mode, M,N")
    args = p.parse_args()
    if args.tile:
        sample(tuple(map(int, args.tile.split(","))), args.output.resolve())
        return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "epilogue tuning only; separate validation required",
        "rounds": args.rounds,
        "tiles": TILES,
        "samples": [],
        "baseline_source_sha256": hashlib.sha256(
            (ROOT / "kernel_bf16_cheating.py").read_bytes()
        ).hexdigest(),
    }
    for round_index in range(args.rounds):
        tiles = TILES.copy()
        random.Random(20261109 + round_index).shuffle(tiles)
        for tile in tiles:
            stem = f"round-{round_index + 1}-{tile[0]}x{tile[1]}"
            target = out / f"{stem}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--tile",
                ",".join(map(str, tile)),
                "--output",
                str(target),
            ]
            print(stem, flush=True)
            with (out / f"{stem}.log").open("w") as log:
                run = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=180
                )
            result = (
                json.loads(target.read_text())
                if run.returncode == 0
                else {"tile": list(tile), "failed": True}
            )
            report["samples"].append(
                {"round": round_index + 1, "command": command, **result}
            )
            print(result.get("tflops", "FAILED"), flush=True)
            (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    report["means"] = {}
    for m, n in TILES:
        samples = [v for v in report["samples"] if list(v["tile"]) == [m, n]]
        if len(samples) == args.rounds and all("tflops" in v for v in samples):
            report["means"][f"{m}x{n}"] = statistics.mean(v["tflops"] for v in samples)
    report["complete"] = True
    (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["means"], indent=2))


if __name__ == "__main__":
    main()
