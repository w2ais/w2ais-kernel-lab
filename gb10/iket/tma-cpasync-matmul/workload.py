"""Launch one warmed 4096³ GEMM for IKET, Nsight Compute, or CUDA-event timing."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from statistics import median

import torch
from cutlass import cute

SIZE = 4096
MODULES = {
    "5A": "kernel_bf16_5A_readable_stmatrix",
    "5M": "kernel_bf16_5M_tma_tuned",
    "6A+": "kernel_nvfp4_6A_plus_delayed_b",
    "6M": "kernel_nvfp4_6M_port",
}


def compile_nvfp4(problem, module_name, ampere):
    module = importlib.import_module(module_name)
    op = module.gemm if ampere else module.make_kernel()
    tensors = [problem[key + "_tensor"] for key in ("a", "b", "sfa", "sfb", "c")]
    worker_count = (
        [] if ampere else [torch.cuda.get_device_properties(0).multi_processor_count]
    )
    stream = problem["stream"]
    compiled = cute.compile(op, *tensors, *worker_count, stream)
    return lambda: compiled(*tensors, stream)


def fill_nvfp4_ones(problem):
    for storage in problem["live_storage"][:2]:
        storage.view(torch.uint8).fill_(0x22)
    for storage in problem["live_storage"][2:4]:
        storage.view(torch.uint8).fill_(0x38)


def milliseconds(launch, iterations):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("kernel", choices=MODULES)
    parser.add_argument("mode", choices=("trace", "ncu", "time"))
    parser.add_argument("--cohort", default="primary")
    parser.add_argument("--scale-lanes", action="store_true")
    args = parser.parse_args()
    kernel = args.kernel
    out = args.output.resolve()
    bf16 = kernel.startswith("5")
    source = out / "gb10" / ("bf16-gemm" if bf16 else "nvfp4-gemm")
    sys.path[:0] = [
        str(out / args.cohort / "modules"),
        str(out / "controls"),
        str(source),
    ]
    from problem import make_problem

    torch.backends.cuda.matmul.allow_tf32 = False
    if bf16:
        import benchmark

        options = benchmark.parse_args([kernel])
        allocation = "tma" if kernel == "5M" else "checkpoint"
        problem = make_problem(
            SIZE, SIZE, SIZE, seed=2026, input=options.input, allocation=allocation
        )
        problem.a.fill_(1)
        problem.b.fill_(1)
        result = problem.c
        launch = benchmark.compile_kernel(options, problem)[0]
        if args.mode == "trace":
            benchmark.KERNELS[kernel] = "traced_" + benchmark.KERNELS[kernel]
            traced = benchmark.compile_kernel(options, problem)[0]
    else:
        problem = make_problem(SIZE, SIZE, SIZE, 2026)
        fill_nvfp4_ones(problem)
        result = problem["c_storage"]
        name = "scale_lanes" if args.scale_lanes else MODULES[kernel]
        launch = compile_nvfp4(problem, name, kernel == "6A+")
        if args.mode == "trace":
            traced = compile_nvfp4(problem, "traced_" + name, kernel == "6A+")

    for _ in range(40):
        launch()
    # A stale warmup result would otherwise pass the output check.
    result.fill_(float("nan"))
    torch.cuda.synchronize()
    if args.mode == "ncu":
        torch.cuda.cudart().cudaProfilerStart()
    (traced if args.mode == "trace" else launch)()
    torch.cuda.synchronize()
    if args.mode == "ncu":
        torch.cuda.cudart().cudaProfilerStop()
    if not torch.all(result == SIZE).item():
        raise ValueError(f"Incorrect output for {kernel}")
    if args.mode == "time":
        samples = [milliseconds(launch, 50) for _ in range(7)]
        version = importlib.metadata.version("nvidia-cutlass-dsl")
        timing = {
            "compiler": version,
            "milliseconds": samples,
            "median_tflops": 2 * SIZE**3 / (1e9 * median(samples)),
        }
        (out / f"time-{version}-{kernel}.json").write_text(
            json.dumps(timing, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
