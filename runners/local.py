"""Run from the repository root with python -m runners.local."""
import argparse
import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware", choices=["gb10", "b300"])
    parser.add_argument("mode", choices=["check", "bench", "profile", "capture"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--ncu-set", choices=["basic", "full"], default="basic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.heads, args.dim, args.page_size) <= 0 or args.context < 0:
        parser.error("batch, heads, dim, and page-size must be positive; context must be nonnegative")

    if args.mode == "profile":
        ncu = shutil.which("ncu")
        if not ncu:
            parser.error("ncu is not on PATH; install Nsight Compute or add its bin directory")
        if args.context == 0:
            parser.error("profiling requires a nonempty context")
        directory = output_dir(args)
        command = [ncu, "--clock-control", "none", "--set", args.ncu_set,
                   "--profile-from-start", "off", "--target-processes", "all",
                   "--export", str(directory / "profile"),
                   sys.executable, "-m", "runners.local", args.hardware, "capture"]
        for name in ("batch", "heads", "dim", "context", "page_size", "dtype", "ncu_set"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        command += ["--output", str(directory)]
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / "ncu.log").write_text(result.stdout + result.stderr)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()
        for suffix, flags in (("txt", []), ("csv", ["--csv"])):
            with (directory / f"profile.{suffix}").open("w") as stream:
                subprocess.run([ncu, "--import", str(directory / "profile.ncu-rep"),
                                "--page", "details", *flags], stdout=stream, check=True)
        print(f"Reports saved to {directory}")
        return

    import torch
    import triton
    import triton.testing
    from shared.attention import launch, make_inputs, reference

    solve = importlib.import_module(f"{args.hardware}.flash_attention_decode.kernel").solve
    inputs = make_inputs(args.batch, args.heads, args.dim, args.context,
                         args.page_size, getattr(torch, args.dtype))
    torch.testing.assert_close(reference(*inputs), run(solve, inputs), atol=2e-3, rtol=2e-2)
    # Compile and stabilize before either timing or enabling the profiler.
    for _ in range(10):
        launch(solve, inputs)
    torch.cuda.synchronize()
    result = {"hardware_target": args.hardware, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "triton": triton.__version__,
              "cuda": torch.version.cuda, "arguments": vars(args) | {"output": str(args.output)},
              "clock_control": "none" if args.mode == "capture" else None}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    result["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    result["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    result["nvidia_smi"] = smi.stdout
    ncu = shutil.which("ncu")
    result["ncu_version"] = subprocess.run([ncu, "--version"], capture_output=True, text=True).stdout if ncu else None
    if args.mode == "capture":
        torch.cuda.cudart().cudaProfilerStart()
        launch(solve, inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    elif args.mode == "bench":
        result["latency_ms_p50_p20_p80"] = triton.testing.do_bench(
            lambda: launch(solve, inputs), warmup=100, rep=500, quantiles=[0.5, 0.2, 0.8])
    else:
        result["correct"] = True
    directory = output_dir(args)
    (directory / "metadata.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def output_dir(args):
    directory = args.output or Path("artifacts") / args.hardware / "flash_attention_decode" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def run(solve, inputs):
    from shared.attention import launch
    launch(solve, inputs)
    return inputs[-2:]


if __name__ == "__main__":
    main()
