"""Compare the four NVFP4 raster schedules on one square problem per run."""

import argparse
import importlib
from statistics import mean

import cutlass.cute as cute
import torch

from benchmark import KERNELS
from problem import launch, make_problem


def measure(size):
    torch.backends.cuda.matmul.allow_tf32 = False
    problem = make_problem(size, size, size, seed=2026)
    tensors = [problem[key] for key in ("a_tensor", "b_tensor", "sfa_tensor", "sfb_tensor", "c_tensor")]
    stream = problem["stream"]
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    kernels = {}
    for name in ("6A+", "7A", "6M", "7M"):
        module = importlib.import_module(KERNELS[name])
        if name.rstrip("+").endswith("A"):
            kernels[name] = cute.compile(module.gemm, *tensors, stream)
        else:
            kernels[name] = cute.compile(module.make_kernel(), *tensors, sm_count, stream)

    a = problem["a_reference"][:, :, 0] * problem["sfa_reference"][:, :, 0].repeat_interleave(16, dim=1)
    b = problem["b_reference"][:, :, 0] * problem["sfb_reference"][:, :, 0].repeat_interleave(16, dim=1)
    expected = (a @ b.T).bfloat16()
    output = problem["c_storage"][:, :, 0]
    for kernel in kernels.values():
        output.fill_(float("nan"))
        launch(kernel, problem)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    for storage in problem["live_storage"][:2]:
        storage.view(torch.uint8).fill_(0x22)
    for storage in problem["live_storage"][2:4]:
        storage.view(torch.uint8).fill_(0x38)
    for kernel in kernels.values():
        for _ in range(20):
            launch(kernel, problem)
    torch.cuda.synchronize()

    names = list(kernels)
    samples = {name: [] for name in names}
    for round_index in range(8):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index >= len(names):
            order.reverse()
        for name in order:
            kernel = kernels[name]
            for _ in range(5):
                launch(kernel, problem)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(12):
                launch(kernel, problem)
            end.record()
            end.synchronize()
            samples[name].append(2 * size**3 * 12 / (start.elapsed_time(end) * 1e9))
    for name in names:
        print(f"{size}³ {name}: {mean(samples[name]):.1f} TFLOP/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", type=int, choices=(4096, 6144, 8192))
    measure(parser.parse_args().size)
