# Tracing TMA and cooperative copies on GB10

This experiment compares four 4096³ GEMM kernels. It traces the BF16 baselines (5A and 5M) and their NVFP4 successors (6A+ and 6M) with NVIDIA In-Kernel Event Tracing (IKET). Separate Nsight Compute runs measure warp stalls and memory accesses. A paired benchmark tests the 6A+ scale-copy lane order and two 6M pipeline changes. [Report](REPORT.md) · [Reference measurements](reference.json)

## Inspect the recorded traces

Each column contains a Perfetto timeline. The primary and replicate captures sample different CTAs. The coarse capture uses the replicate CTAs and records only output-tile boundaries and the start of the epilogue.

| Kernel | Primary | Replicate | Coarse |
|---|---|---|---|
| BF16 5A | [5A](traces/primary/5A.pftrace.gz) | [5A](traces/replicate/5A.pftrace.gz) | [5A](traces/coarse/5A.pftrace.gz) |
| BF16 5M | [5M](traces/primary/5M.pftrace.gz) | [5M](traces/replicate/5M.pftrace.gz) | [5M](traces/coarse/5M.pftrace.gz) |
| NVFP4 6A+ | [6A+](traces/primary/6A+.pftrace.gz) | [6A+](traces/replicate/6A+.pftrace.gz) | [6A+](traces/coarse/6A+.pftrace.gz) |
| NVFP4 6M | [6M](traces/primary/6M.pftrace.gz) | [6M](traces/replicate/6M.pftrace.gz) | [6M](traces/coarse/6M.pftrace.gz) |

On macOS or Linux, run the following from this directory and open the resulting file with **Open trace file** in [Perfetto](https://ui.perfetto.dev/).

```sh
gzip -dc 'traces/primary/6A+.pftrace.gz' > /tmp/6A-plus.pftrace
```

## Reproduce on a GB10

Use a Linux GB10 host with Git, Python 3.12, uv, the CUDA 13 toolkit, GNU patch, and Nsight Compute with permission to collect hardware counters. The original run used driver 580.159.03 and Nsight Compute 2026.2.1. Use a Git clone containing the benchmark revision `ee17c2c`. From this directory, install the two environments and run the experiment.

```sh
uv sync --project ../../nvfp4-gemm --locked
uv venv --python 3.12 .venv/iket
uv pip install --python .venv/iket/bin/python -r requirements-iket.txt
./reproduce.sh
```

The benchmark environment uses CuTe DSL 4.5.2. The second environment uses 4.6.0 for IKET and the compiler timing comparison. To use existing environments, set `BENCH_PYTHON` and `TRACE_PYTHON` to their Python executables. The script uses the `run-iket` installed beside `TRACE_PYTHON`.

The runner extracts the two GEMM packages from the pinned revision into a fresh output directory, reconstructs the measured 6A+ checkpoint, and verifies the four kernel source hashes. It runs all 12 traces, the uninstrumented counters, CUDA-event checks on both compilers, the five-variant control experiment, and three instruction-level profiles. Each captured kernel's output is checked. The controls also check random inputs and scales against a BF16-rounded FP32 reference.

Each invocation creates an ignored `artifacts/run-*` directory. Pass a new directory as the first argument to choose another destination. Open its `summary.json` for the measurements, its `*.ncu-rep` files in Nsight Compute, or its `*.pftrace` files in Perfetto. To regenerate the summary from an existing run, use `python3 analyze.py <output-directory>`. The checked-in reference measurements come from the run described in the report. Fresh timings depend on clocks and system load.
