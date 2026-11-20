"""Compile, validate and time a BF16 checkpoint, independently of Nsight.

    uv run --locked python benchmark.py 3A --output results/3A.json
    uv run --locked python benchmark.py 5M --profile-only --output results/5M.json

The optional 5M --compare-controls mode preserves the published comparison
runner's allocations, checks and rotating per-trial warmup/timing order.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
KERNELS = {
    "0": "kernel_bf16_0_scalar_fma.py",
    "1": "kernel_bf16_1_warp_mma.py",
    "2": "kernel_bf16_2_l2_scheduler.py",
    "3A": "kernel_bf16_3A_cpasync.py",
    "4A": "kernel_bf16_4A_register_pipeline.py",
    "5A": "kernel_bf16_5A_readable_stmatrix.py",
    "3M": "kernel_bf16_3M_tma_persistent.py",
    "4M": "kernel_bf16_4M_register_pipeline.py",
    "5M": "kernel_bf16_5M_tma_tuned.py",
    "cheating": "kernel_bf16_cheating.py",
}
NUMBERED_KERNELS = tuple(key for key in KERNELS if key != "cheating")


def command(
    kernel, *, size=4096, warmups=20, iterations=50, trials=1, comparison=False
):
    result = [
        sys.executable,
        str(ROOT / "benchmark.py"),
        kernel,
        "--size",
        str(size),
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
        "--trials",
        str(trials),
    ]
    if kernel == "5M" and comparison:
        result += ["--compare-controls", "--include-readable"]
    return result


def source_hashes():
    paths = [
        *ROOT.glob("*.py"),
        *ROOT.glob("benchmarks/*.py"),
        *ROOT.glob("helpers/*.py"),
        *ROOT.glob("references/*.py"),
        ROOT / "uv.lock",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def parse_args(argv, *, legacy=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", choices=NUMBERED_KERNELS)
    parser.add_argument("--size", type=int)
    for axis in ("m", "n", "k"):
        parser.add_argument(f"--{axis}", type=int)
    parser.add_argument("--warmups", "--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input", choices=("normal", "integer"))
    parser.add_argument("--correctness-repeats", type=int)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.004)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--profile-only", action="store_true", help="One launch; no reference or timing"
    )
    parser.add_argument("--output", "--json", dest="output", type=Path)
    args = parser.parse_args(argv)
    scalar_legacy = legacy and args.kernel == "0"
    default_size = 256 if scalar_legacy else 4096
    if args.size is not None and any(
        getattr(args, a) is not None for a in ("m", "n", "k")
    ):
        parser.error("Use --size or --m/--n/--k, not both")
    for axis in ("m", "n", "k"):
        if getattr(args, axis) is None:
            setattr(args, axis, args.size if args.size is not None else default_size)
    if args.warmups is None:
        args.warmups = 3 if scalar_legacy else 20
    if args.iterations is None:
        args.iterations = 20 if scalar_legacy else 50
    if args.input is None:
        args.input = "integer" if args.kernel == "5M" else "normal"
    if args.correctness_repeats is None:
        args.correctness_repeats = 2 if args.kernel == "5M" else 1
    if (
        min(
            args.m,
            args.n,
            args.k,
            args.iterations,
            args.trials,
            args.correctness_repeats,
        )
        <= 0
        or args.warmups < 0
    ):
        parser.error(
            "Shapes and check/timing counts must be positive; warmups may be zero"
        )
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    # Numbered checkpoints have fixed schedules; tuning lives in benchmarks/.
    args.persistent = args.kernel.endswith("M")
    args.mma_warps = 8 if args.persistent else None
    args.raster = (
        "group-m8"
        if args.persistent
        else "natural"
        if args.kernel in ("0", "1")
        else "group-n4"
    )
    args.epilogue = (
        "stmatrix"
        if args.kernel == "5A"
        else "stmatrix-tma"
        if args.kernel == "5M"
        else "direct"
    )
    return args


def compile_kernel(args, problem):
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cuda.bindings import driver as cuda
    import torch

    module = importlib.import_module(Path(KERNELS[args.kernel]).stem)
    views = problem.cute_tensors()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    metadata = dict(getattr(module, "SOURCE_METADATA", {}))
    tile = (
        (1, 256, 1)
        if args.kernel == "0"
        else getattr(module, "CTA_TILER", (128, 128, 32))
    )
    if args.kernel != "0" and any(
        size % block for size, block in zip((args.m, args.n, args.k), tile)
    ):
        raise ValueError(f"Kernel {args.kernel} requires complete {tile} tiles")
    gm, gn = (args.m + tile[0] - 1) // tile[0], (args.n + tile[1] - 1) // tile[1]
    grid = [gm, gn, 1]
    if args.kernel == "0":
        op, constants = module.naive_bf16_gemm, ()
    elif args.kernel.endswith("A") or args.kernel in ("1", "2"):
        constants = ()
        if args.kernel in ("1", "2"):
            op = module.tensor_core_sync_gemm
            grid = [gm * gn, 1, 1]
        else:
            if gn % 4:
                raise ValueError(
                    "Group-N4 ordering requires a multiple of four N tiles"
                )
            if gm * gn < 128 or gm * gn % 128:
                raise ValueError("Ampere checkpoints require a multiple of 128 CTAs")
            op = module.gb10_cpasync_gemm
            grid = [128, gm * gn // 128, 1]
        metadata.update(raster=args.raster, epilogue=args.epilogue)
    elif args.kernel in ("3M", "4M"):
        cap = utils.HardwareInfo().get_max_active_clusters(1)
        op, constants = module.bf16_tma_baseline, (cap,)
        grid = [min(gm * gn, cap), 1, 1]
        metadata.update(
            mma_warps=args.mma_warps,
            threads=32 * (args.mma_warps + 1),
            worker_grid_cap=cap,
            raster="group-m8",
        )
    else:
        import cutlass

        op = module.TunedSm120GemmKernel(cutlass.Float32, tile)
        cap = torch.cuda.get_device_properties(0).multi_processor_count
        constants = (cap,)
        grid = [min(gm * gn, cap), 1, 1]
    start = time.perf_counter()
    compiled = cute.compile(op, *views, *constants, stream)
    compile_seconds = time.perf_counter() - start
    if args.kernel == "5M":
        metadata.update(
            worker_grid_cap=cap,
            mma_warps=8,
            producer_warps=1,
            threads=288,
            ab_stages=op.ab_stage,
            epi_stages=op.epi_stage,
            epi_tile=list(op.epi_tile),
            raster="group-m8",
            ab_payload_bytes=(tile[0] + tile[1]) * tile[2] * 2 * op.ab_stage,
            c_payload_bytes=op.epi_tile[0] * op.epi_tile[1] * 2 * op.epi_stage,
        )
    metadata.update(cta_tile_mnk=list(tile), persistent=args.kernel.endswith("M"))
    # Hold the CuTe views and compiled callable for the lifetime of the launch.
    return lambda: compiled(*views, stream), metadata, grid, compile_seconds


def run(args):
    import torch
    from problem import make_problem
    from benchmarks.timing import time_launch, timing_record

    allocation = "tma" if args.kernel == "5M" else "checkpoint"
    record = {
        "kernel": args.kernel,
        "source": KERNELS[args.kernel],
        "sources": source_hashes(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "problem": {"m": args.m, "n": args.n, "k": args.k},
        "precision": {
            "input": "bfloat16",
            "accumulator": "float32",
            "output": "bfloat16",
        },
        "layouts": "K-major A/B, N-major C",
        "input": {
            "distribution": args.input,
            "seed": args.seed,
            "allocation": allocation,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        problem = make_problem(
            args.m,
            args.n,
            args.k,
            seed=args.seed,
            input=args.input,
            allocation=allocation,
        )
        launch, metadata, grid, seconds = compile_kernel(args, problem)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        metadata.update(
            gpu=props.name,
            compute_capability=f"sm_{props.major}{props.minor}",
            multiprocessor_count=props.multi_processor_count,
            torch_version=torch.__version__,
            cuda_runtime_version=torch.version.cuda,
        )
        record.update(
            metadata=metadata,
            resources=metadata,
            launch_grid=grid,
            compile_seconds=seconds,
            tensor_strides={
                name: list(t.stride())
                for name, t in zip("ABC", (problem.a, problem.b, problem.c))
            },
        )
        if args.profile_only:
            launch()
            torch.cuda.synchronize()
            record.update(
                status="profile_only_no_correctness", correctness={"checked": False}
            )
        else:
            record["correctness"] = (
                {"checked": False}
                if args.skip_correctness
                else problem.check(
                    launch,
                    repeats=args.correctness_repeats,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            milliseconds = time_launch(
                launch,
                warmups=args.warmups,
                iterations=args.iterations,
                trials=args.trials,
            )
            record["timing"] = timing_record(
                milliseconds,
                args.m,
                args.n,
                args.k,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            record["status"] = (
                "timed_without_correctness" if args.skip_correctness else "passed"
            )
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


def main(kernel=None, *, legacy=False):
    argv = sys.argv[1:]
    if kernel is not None:
        argv = [kernel, *argv]
    if not argv:
        parse_args(argv)
    if argv[0] in ("ampere", "cheating"):
        from benchmarks.ampere import main as ampere_main

        return ampere_main(
            argv[1:], implementation="stmatrix" if argv[0] == "cheating" else "upstream"
        )
    if argv[0] == "5M" and (legacy or "--compare-controls" in argv):
        from benchmarks.tma import main as tma_main

        return tma_main([arg for arg in argv[1:] if arg != "--compare-controls"])
    return run(parse_args(argv, legacy=legacy))


if __name__ == "__main__":
    main()
