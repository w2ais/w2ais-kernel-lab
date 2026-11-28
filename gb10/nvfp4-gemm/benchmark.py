"""Check every output, then time one numbered kernel with CUDA events."""

import argparse, hashlib, importlib, json, statistics, time, platform
from importlib.metadata import version
from pathlib import Path
import torch
import cutlass.cute as cute
from problem import make_problem, launch

ROOT = Path(__file__).resolve().parent
KERNELS = {
    "6A": "kernel_nvfp4_6A_scale_pipeline",
    "6A+": "kernel_nvfp4_6A_plus_delayed_b",
    "7A": "kernel_nvfp4_7A_tile_traversal",
    "6M": "kernel_nvfp4_6M_port",
    "8M": "kernel_nvfp4_8M_local_tma",
    "7M": "kernel_nvfp4_7M_tile_traversal",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kernel", choices=KERNELS)
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--compile-options", default="")
    p.add_argument("--tile", default=None)
    p.add_argument("--stages", type=int)
    swizzle = p.add_mutually_exclusive_group()
    swizzle.add_argument("--group", type=int, help="L2 swizzle band width in output tile columns")
    swizzle.add_argument("--group-m", type=int, help="L2 swizzle band height in output tile rows")
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.size % 256 or args.size < 512:
        p.error("size must be a multiple of 256, at least 512")
    if args.warmups < 0 or args.iterations < 1 or args.trials < 1:
        p.error("warmups must be nonnegative; iterations and trials must be positive")
    ampere_style = args.kernel.rstrip("+").endswith("A")
    if any(value is not None and value < 1 for value in (args.group, args.group_m)):
        p.error("swizzle band size must be at least 1")
    torch.backends.cuda.matmul.allow_tf32 = False
    problem = make_problem(args.size, args.size, args.size, args.seed)
    module = importlib.import_module(KERNELS[args.kernel])
    tensors = [
        problem[k]
        for k in ["a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "c_tensor"]
    ]
    extra = {}
    if ampere_style:
        if args.tile:
            module.TILE = tuple(map(int, args.tile.split(",")))
        if args.stages:
            module.STAGES = args.stages
        if args.group is not None:
            module.GROUP_N = args.group
            module.GROUP_M = None
        elif args.group_m is not None:
            module.GROUP_M = args.group_m
        op = module.gemm
        compile_args = [*tensors, problem["stream"]]
        extra = {
            "tile": module.TILE,
            "stages": module.STAGES,
        }
    else:
        if args.group is not None:
            module.L2_GROUP = args.group
        if args.group_m is not None:
            # Width one in CuTe's persistent scheduler always walks M first.
            # A full-width column band walks N first, exactly one row at a time.
            module.RASTER_ALONG_M = args.group_m == 1
            module.L2_GROUP = args.size // module.TILE[1] if args.group_m == 1 else args.group_m
        op = module.make_kernel()
        compile_args = [
            *tensors,
            torch.cuda.get_device_properties(0).multi_processor_count,
            problem["stream"],
        ]
    t = time.perf_counter()
    compiled = cute.compile(op, *compile_args, options=args.compile_options)
    seconds = time.perf_counter() - t
    launch(compiled, problem)
    torch.cuda.synchronize()
    if not ampere_style:
        extra = {
            "tile": op.tile_shape_mnk,
            "stages": op.ab_stage,
            "epilogue_stages": op.epi_stage,
            "epilogue_tile": op.epi_tile,
            "warps": op.threads_per_cta // 32,
        }
    row_band = module.GROUP_M if ampere_style else args.group_m
    if row_band is None:
        extra["group_n"] = module.GROUP_N if ampere_style else module.L2_GROUP
    else:
        extra["group_m"] = row_band
    record = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cutlass_dsl": version("nvidia-cutlass-dsl"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "kernel": args.kernel,
        "precision": {"input": "NVFP4", "accumulator": "float32", "output": "bfloat16"},
        "configuration": vars(args) | {"output": str(args.output)},
        "resources": extra,
        "compile_seconds": seconds,
        "sources": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                *ROOT.glob("kernel_*.py"),
                *(ROOT / "references").glob("*.py"),
                ROOT / "benchmark.py",
                ROOT / "problem.py",
                ROOT / "tile_order.py",
                ROOT / "uv.lock",
            ]
        },
    }
    if not args.profile_only:
        sa = problem["sfa_reference"][:, :, 0].repeat_interleave(16, dim=1)
        sb = problem["sfb_reference"][:, :, 0].repeat_interleave(16, dim=1)
        expected = (
            (problem["a_reference"][:, :, 0] * sa)
            @ (problem["b_reference"][:, :, 0] * sb).T
        ).to(torch.bfloat16)
        actual = problem["c_storage"][:, :, 0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        record["correctness"] = {
            "passed": True,
            "comparison": "all output elements exactly equal to rounded FP32 reference",
            "max_error": float((actual - expected).abs().max()),
        }
        # Match cuBLAS's timed values after the stronger random-data check.
        # E2M1 1.0 is nibble 0b0010; packed pair = 0x22.
        for storage in problem["live_storage"][:2]:
            storage.view(torch.uint8).fill_(0x22)
        for storage in problem["live_storage"][2:4]:
            storage.view(torch.uint8).fill_(0x38)  # E4M3 1.0
        launch(compiled, problem)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual, torch.full_like(actual, args.size), atol=0, rtol=0
        )
        record["timing_input"] = (
            "A=B=1, SFA=SFB=1, same values as cuBLAS; random all-output check before replacement"
        )
        for _ in range(100):
            launch(compiled, problem)
        record["conditioning_launches"] = 100
        for _ in range(args.warmups):
            launch(compiled, problem)
        samples = []
        for _ in range(args.trials):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(args.iterations):
                launch(compiled, problem)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / args.iterations)
        record["milliseconds"] = samples
        record["tflops"] = [2 * args.size**3 / (x * 1e9) for x in samples]
        record["mean_tflops"] = statistics.mean(record["tflops"])
    else:
        record["profile_only"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in ["kernel", "resources", "mean_tflops", "correctness"]
                if k in record
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
