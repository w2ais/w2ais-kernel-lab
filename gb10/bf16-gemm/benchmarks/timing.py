"""CUDA-event timing, kept separate from kernel code and Nsight profiling."""

import statistics

import torch


def event_trial(launch, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def time_launch(launch, *, warmups, iterations, trials):
    for _ in range(warmups):
        launch()
    torch.cuda.synchronize()
    return [event_trial(launch, iterations) for _ in range(trials)]


def timing_record(milliseconds, m, n, k, *, warmups, iterations):
    median_ms = statistics.median(milliseconds)
    tflops = [2.0 * m * n * k / (ms * 1e9) for ms in milliseconds]
    return {
        "method": "CUDA events; compilation, allocation and validation excluded",
        "warmups": warmups,
        "iterations_per_trial": iterations,
        "trials": len(milliseconds),
        "trial_milliseconds": milliseconds,
        "median_milliseconds": median_ms,
        "minimum_milliseconds": min(milliseconds),
        "maximum_milliseconds": max(milliseconds),
        "trial_tflops": tflops,
        "median_tflops": 2.0 * m * n * k / (median_ms * 1e9),
        "mean_tflops": statistics.mean(tflops),
        "best_tflops": max(tflops),
    }
