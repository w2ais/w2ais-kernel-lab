#!/usr/bin/env python3
"""Benchmark the exact vendored NVIDIA Ampere CuTe DSL TensorOp GEMM.

The kernel implementation lives in ``nvidia_ampere_tensorop_gemm.py`` and is
byte-for-byte identical to NVIDIA CUTLASS's current BSD-3-Clause example at
the time it was vendored.  Keeping this harness separate is intentional:
seemingly harmless simplifications of the kernel changed CuTe DSL lowering,
occupancy, shared-memory conflicts, and ultimately GB10 performance.

The benchmark contract is BF16 A/B/C, FP32 accumulation, K-major
A/B and N-major C, and ``C = A @ B.T``.  At 4096^3, the upstream automatic rasterizer
selects factor eight.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

import cutlass
from references import nvidia_ampere_tensorop_gemm as upstream


CTA_TILER = (128, 128, 32)
MMA_INSTRUCTION_MNK = (16, 8, 16)
UPSTREAM_SOURCE = (
    "https://github.com/NVIDIA/cutlass/blob/main/"
    "examples/python/CuTeDSL/ampere/tensorop_gemm.py"
)
EXPECTED_VENDOR_SHA256 = (
    "6ae4a6bb8214f865a1d8d9bef6c6da5abc621f0b6a1fa6ea9deba3bac7fb215b"
)


def parse_atom_layout(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "atom layout must contain three comma-separated integers"
        ) from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError(
            "atom layout must have MNK form, for example 2,2,1"
        )
    return result


def parse_args(argv=None, *, implementation="upstream") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--implementation",
        choices=("upstream", "stmatrix"),
        default=implementation,
        help="Exact NVIDIA kernel, or the exact-mainloop StMatrix experiment",
    )
    parser.add_argument("--atom-layout-mnk", type=parse_atom_layout, default=(2, 2, 1))
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Use upstream's circular workspaces to exceed L2 capacity",
    )
    parser.add_argument("--json", "--output", dest="json", type=Path)
    args = parser.parse_args(argv)
    if args.size is not None:
        args.m = args.n = args.k = args.size
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.m <= 0 or args.m % 8:
        raise ValueError("M must be positive and a multiple of eight")
    if args.n <= 0 or args.n % 8:
        raise ValueError("N must be positive and a multiple of eight")
    if args.k <= 0:
        raise ValueError("K must be positive")
    atom_m, atom_n, atom_k = args.atom_layout_mnk
    if atom_m <= 0 or atom_n <= 0 or atom_k != 1:
        raise ValueError("atom layout M/N must be positive and K must equal one")
    if CTA_TILER[0] % (atom_m * MMA_INSTRUCTION_MNK[0]):
        raise ValueError("128 must be divisible by atom_m * 16")
    if CTA_TILER[1] % (atom_n * MMA_INSTRUCTION_MNK[1]):
        raise ValueError("128 must be divisible by atom_n * 8")
    if atom_m * atom_n * 32 > 1024:
        raise ValueError("atom layout produces more than 1024 threads")
    if args.warmups < 0 or args.iterations <= 0 or args.trials <= 0:
        raise ValueError("warmups must be nonnegative; iterations/trials positive")
    if args.implementation == "stmatrix":
        if args.atom_layout_mnk != (2, 2, 1):
            raise ValueError("StMatrix requires --atom-layout-mnk 2,2,1")
        if args.m % CTA_TILER[0] or args.n % CTA_TILER[1]:
            raise ValueError("StMatrix requires full 128x128 output tiles")


def upstream_raster_factor(n: int) -> int:
    grid_n = (n + CTA_TILER[1] - 1) // CTA_TILER[1]
    if grid_n > 5:
        return 8
    if grid_n > 2:
        return 4
    if grid_n > 1:
        return 2
    return 1


def git_value(*git_args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *git_args],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def main(argv=None, *, implementation="upstream") -> None:
    args = parse_args(argv, implementation=implementation)
    torch.backends.cuda.matmul.allow_tf32 = False
    validate_args(args)

    vendor_path = Path(upstream.__file__).resolve()
    vendor_sha256 = hashlib.sha256(vendor_path.read_bytes()).hexdigest()
    if vendor_sha256 != EXPECTED_VENDOR_SHA256:
        raise RuntimeError(
            "vendored NVIDIA source changed unexpectedly: "
            f"expected {EXPECTED_VENDOR_SHA256}, got {vendor_sha256}"
        )

    if args.implementation == "upstream":
        implementation = upstream
        implementation_path = vendor_path
        variant = "exact_nvidia_ampere_tensorop_gemm"
        epilogue = "upstream shared-staged vector store"
        transform_metadata: dict[str, object] = {}
    else:
        implementation = importlib.import_module("kernel_bf16_cheating")
        implementation_path = Path(implementation.__file__).resolve()
        variant = "exact_nvidia_mainloop_4warp_stmatrix_epilogue"
        epilogue = "128x32 subtiled StMatrix + 128-bit vector store"
        transform_metadata = {
            "transform_source_file": str(implementation_path),
            "transform_source_sha256": hashlib.sha256(
                implementation_path.read_bytes()
            ).hexdigest(),
            "generated_source_sha256": implementation.GENERATED_SOURCE_SHA256,
            "upstream_mainloop_sha256": implementation.ORIGINAL_MAINLOOP_SHA256,
            "transformed_mainloop_sha256": (implementation.TRANSFORMED_MAINLOOP_SHA256),
            "mainloop_text_is_unchanged": (
                implementation.ORIGINAL_MAINLOOP_SHA256
                == implementation.TRANSFORMED_MAINLOOP_SHA256
            ),
        }

    trial_microseconds: list[float] = []
    for trial in range(args.trials):
        # Seed each independent upstream run identically.  Correctness is
        # checked once; subsequent trials still execute the exact same kernel
        # benchmark but avoid repeating the large host reference transfer.
        torch.manual_seed(args.seed)
        average_us = implementation.run(
            "k",
            "k",
            "n",
            cutlass.BFloat16,
            cutlass.BFloat16,
            cutlass.Float32,
            (args.m, args.n, args.k, 1),
            args.atom_layout_mnk,
            args.warmups,
            args.iterations,
            args.skip_correctness or trial > 0,
            args.cold_l2,
        )
        trial_microseconds.append(float(average_us))

    trial_milliseconds = [value / 1000.0 for value in trial_microseconds]
    median_ms = statistics.median(trial_milliseconds)
    trial_tflops = [
        2.0 * args.m * args.n * args.k / (milliseconds * 1.0e9)
        for milliseconds in trial_milliseconds
    ]

    source_path = Path(__file__).resolve()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    raster_factor = upstream_raster_factor(args.n)
    grid_m = (args.m + CTA_TILER[0] - 1) // CTA_TILER[0]
    grid_n = (args.n + CTA_TILER[1] - 1) // CTA_TILER[1]
    from benchmark import source_hashes

    result = {
        "sources": source_hashes(),
        "metadata": {
            "variant": variant,
            "implementation": args.implementation,
            "upstream_source": UPSTREAM_SOURCE,
            "upstream_license": "BSD-3-Clause",
            "vendor_source_file": str(vendor_path),
            "vendor_source_sha256": vendor_sha256,
            "vendor_is_expected_byte_exact_copy": True,
            "implementation_source_file": str(implementation_path),
            **transform_metadata,
            "benchmark_source_file": str(source_path),
            "benchmark_source_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "contract": "A[M,K] @ B[N,K].T -> C[M,N]",
            "layouts": "K-major A/B, N-major C",
            "input_dtype": "bfloat16",
            "accumulator_dtype": "float32",
            "output_dtype": "bfloat16",
            "cta_tile_mnk": CTA_TILER,
            "mma_instruction_mnk": MMA_INSTRUCTION_MNK,
            "atom_layout_mnk": args.atom_layout_mnk,
            "threads": 32
            * args.atom_layout_mnk[0]
            * args.atom_layout_mnk[1]
            * args.atom_layout_mnk[2],
            "stages": 3,
            "gmem_to_smem": "128-bit cp.async",
            "smem_to_register": "ldmatrix.x4 rolling register pipeline",
            "epilogue": epilogue,
            "gpu": props.name,
            "compute_capability": f"sm_{props.major}{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cutlass_module": str(Path(cutlass.__file__).resolve()),
            "git_commit": git_value("rev-parse", "HEAD"),
        },
        "problem": {"m": args.m, "n": args.n, "k": args.k},
        "raster": {
            "factor": raster_factor,
            "mapping": "upstream automatic factor-N remap",
        },
        "launch_grid": [
            grid_m * raster_factor,
            (grid_n + raster_factor - 1) // raster_factor,
            1,
        ],
        "correctness": {
            "checked": not args.skip_correctness,
            "passed": None if args.skip_correctness else True,
            "implementation": (
                None if args.skip_correctness else "upstream torch.testing.assert_close"
            ),
        },
        "cache_policy": "cold L2 workspaces" if args.cold_l2 else "warm L2",
        "timing": {
            "warmups_per_trial": args.warmups,
            "iterations_per_trial": args.iterations,
            "trials": args.trials,
            "trial_microseconds": trial_microseconds,
            "trial_milliseconds": trial_milliseconds,
            "median_milliseconds": median_ms,
            "minimum_milliseconds": min(trial_milliseconds),
            "maximum_milliseconds": max(trial_milliseconds),
            "trial_tflops": trial_tflops,
            "median_tflops": 2.0 * args.m * args.n * args.k / (median_ms * 1.0e9),
            "best_tflops": max(trial_tflops),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
