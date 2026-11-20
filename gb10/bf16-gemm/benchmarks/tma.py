"""Correctness and timing for Modern Kernel 5M and tuning candidates.

Controls use unchanged NVIDIA Ampere and a fixed vendor TMA configuration on
the same allocations. The published comparison uses the candidate's own
throughput and a separate common cuBLAS baseline. Optional source transforms
support tuning while preserving the checked-in kernel and vendor files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path

import torch
from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from helpers import tma_config as scheduler
from helpers.tma_config import make_op
from problem import make_problem
from benchmarks.timing import event_trial
from references import nvidia_sm120_dense_gemm as vendor
from references import nvidia_ampere_tensorop_gemm as ampere
import kernel_bf16_5A_readable_stmatrix as readable

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VENDOR = "613052799aff35d5564d49c8bbb4bbac2e22bc58cb3e27499c4c9c3ee95c6e03"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", default="128,128,32")
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--epilogue-stages", type=int, default=2)
    parser.add_argument("--group", type=int, choices=[1, 2, 4, 8, 16], default=8)
    parser.add_argument("--raster", choices=["m", "n"], default="m")
    parser.add_argument("--waves", type=int, choices=[1, 2], default=1)
    parser.add_argument("--mma-warps", type=int, choices=[4, 8], default=8)
    parser.add_argument(
        "--epi-tile", choices=["auto", "32,32", "64,64"], default="auto"
    )
    parser.add_argument("--output-sync", choices=["original", "reuse"], default="reuse")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--input", choices=["integer", "normal"], default="integer")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--correctness-repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--include-readable", action="store_true")
    parser.add_argument("--profile-only", choices=["baseline", "candidate", "ampere"])
    parser.add_argument("--json", "--output", dest="json", type=Path, required=True)
    args = parser.parse_args(argv)
    args.tile = tuple(map(int, args.tile.split(",")))
    if args.tile not in scheduler.SUPPORTED_TILES | {
        (64, 128, 32),
        (128, 128, 32),
        (128, 256, 32),
    }:
        parser.error(
            "Tile must be an upstream shape or an explicitly screened K=32 shape"
        )
    if args.size <= 0 or args.size % 1024 or args.size % args.tile[2]:
        parser.error("Size must be a positive multiple of 1024 and tile K")
    if (
        min(
            args.stages,
            args.epilogue_stages,
            args.correctness_repeats,
            args.iterations,
            args.trials,
        )
        < 1
        or args.warmups < 0
    ):
        parser.error("Invalid stages or timing/check counts")
    return args


def load_candidate(directory, args):
    source = (ROOT / "kernel_bf16_5M_tma_tuned.py").read_text()
    if args.mma_warps == 4:
        source = source.replace(
            "self.atom_layout = (2, 4, 1)", "self.atom_layout = (2, 2, 1)"
        )
        source = source.replace(
            "        self.epi_tile = (64, 64)\n",
            "        self.epi_tile = sm90_utils.compute_tile_shape_or_override(\n"
            "            self.tile_shape_mnk, self.c_dtype, is_cooperative=False\n"
            "        )\n",
        )
    epi_tile = args.epi_tile if args.epi_tile != "auto" else None
    if epi_tile is not None:
        marker = "        # Compute stage before compute smem layout"
        assert source.count(marker) == 1
        source = source.replace(
            marker, f"        self.epi_tile = ({epi_tile})\n\n" + marker
        )
    if args.output_sync == "original":
        marker = "                    # Join all compute warps before any can overwrite the next slot.\n                    self.epilog_sync_barrier.arrive_and_wait()"
        assert source.count(marker) == 1
        source = source.replace(marker, "")
    path = directory / "tuned_sm120.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("tuned_sm120", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main(argv=None):
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.json.parent.mkdir(parents=True, exist_ok=True)
    configuration = vars(args) | {"json": str(args.json)}
    from benchmark import source_hashes

    record = {
        "package_sources": source_hashes(),
        "configuration": configuration,
        "precision": {
            "input": "bfloat16",
            "accumulator": "float32",
            "output": "bfloat16",
        },
        "status": "started",
        "sources": {
            Path(m.__file__).name: hashlib.sha256(
                Path(m.__file__).read_bytes()
            ).hexdigest()
            for m in (scheduler, vendor, ampere, readable)
        },
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save():
        args.json.write_text(json.dumps(record, indent=2) + "\n")

    source_archive = args.json.parent / "sources"
    source_archive.mkdir(exist_ok=True)
    (source_archive / (record["experiment_sha256"] + ".py")).write_bytes(
        Path(__file__).read_bytes()
    )
    save()
    try:
        with tempfile.TemporaryDirectory(prefix="gb10-tma-tuning-") as directory:
            module, source = load_candidate(Path(directory), args)
            record["generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            args.json.with_suffix(".generated.py").write_text(source)
            problem = make_problem(
                args.size,
                args.size,
                args.size,
                seed=args.seed,
                input=args.input,
                allocation="tma",
            )
            a, b, c = problem.a, problem.b, problem.c
            record["layouts"] = "K-major A/B, N-major C"
            record["tensor_strides"] = {
                name: list(t.stride()) for name, t in zip(("A", "B", "C"), (a, b, c))
            }
            av, bv, cv = problem.cute_tensors()
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            baseline = make_op(vendor, "baseline", (64, 128, 64), 3, 4, 16, "n")
            candidate = make_op(
                module,
                "candidate",
                args.tile,
                args.stages,
                args.epilogue_stages,
                args.group,
                args.raster,
            )
            ops = {"baseline": baseline, "candidate": candidate}
            launches = {}
            resources = {}
            for name, op in ops.items():
                cap = sms * (args.waves if name == "candidate" else 1)
                compiled = cute.compile(op, av, bv, cv, cap, stream)
                launches[name] = lambda fn=compiled: fn(av, bv, cv, stream)
                resources[name] = {
                    "worker_grid_cap": cap,
                    "ab_stages": op.ab_stage,
                    "epi_stages": op.epi_stage,
                    "epi_tile": list(op.epi_tile),
                    "ab_payload_bytes": (op.tile_shape_mnk[0] + op.tile_shape_mnk[1])
                    * op.tile_shape_mnk[2]
                    * 2
                    * op.ab_stage,
                    "c_payload_bytes": op.epi_tile[0]
                    * op.epi_tile[1]
                    * 2
                    * op.epi_stage,
                }
            compiled_ampere = cute.compile(
                ampere.TensorOpGemm(
                    cutlass.BFloat16, cutlass.BFloat16, cutlass.Float32, (2, 2, 1)
                ),
                av,
                bv,
                cv,
            )
            launches["ampere"] = lambda: compiled_ampere(av, bv, cv)
            if args.include_readable:
                a2, b2, c2 = [
                    from_dlpack(t[:, :, 0], assumed_align=16) for t in (a, b, c)
                ]
                compiled_readable = cute.compile(
                    readable.gb10_cpasync_gemm, a2, b2, c2, stream
                )
                launches["kernel5A"] = lambda: compiled_readable(a2, b2, c2, stream)
            record["resources"] = resources
            if args.profile_only:
                for _ in range(args.warmups):
                    launches[args.profile_only]()
                launches[args.profile_only]()
                torch.cuda.synchronize()
                record["status"] = "profile_only_no_correctness"
                save()
                return
            reference = problem.reference()
            errors = {}
            for name, launch in launches.items():
                errors[name] = []
                for _ in range(args.correctness_repeats):
                    c.fill_(float("nan"))
                    launch()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        c[:, :, 0].float(), reference, atol=0.01, rtol=0.004
                    )
                    errors[name].append(
                        float((c[:, :, 0].float() - reference.float()).abs().max())
                    )
            record["correctness"] = errors
            samples = {name: [] for name in launches}
            orders = []
            for trial in range(args.trials):
                names = list(launches)
                offset = trial % len(names)
                order = names[offset:] + names[:offset]
                if trial % 2:
                    order = order[::-1]
                orders.append(order)
                for name in order:
                    for _ in range(args.warmups):
                        launches[name]()
                    samples[name].append(event_trial(launches[name], args.iterations))
            throughputs = {
                name: [2 * args.size**3 / (ms * 1e9) for ms in values]
                for name, values in samples.items()
            }
            means = {
                name: statistics.mean(values) for name, values in throughputs.items()
            }
            record.update(
                status="passed",
                milliseconds=samples,
                trial_tflops=throughputs,
                mean_tflops=means,
                stdev_tflops={
                    name: statistics.stdev(values) if len(values) > 1 else 0
                    for name, values in throughputs.items()
                },
                trial_order=orders,
                candidate_vs_baseline_percent=100
                * (means["candidate"] / means["baseline"] - 1),
                candidate_vs_ampere_percent=100
                * (means["candidate"] / means["ampere"] - 1),
            )
            # Validate the output once more after the timing loop.
            for launch in launches.values():
                launch()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    c[:, :, 0].float(), reference, atol=0.01, rtol=0.004
                )
            save()
            print(json.dumps(record, indent=2))
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
