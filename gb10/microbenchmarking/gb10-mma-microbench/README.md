# GB10 DGX Spark microbenchmarks

The upcoming GB10 microbenchmarking article reports CUDA C++ timings. This directory contains the kernels, input plans, timing and analysis scripts, plot generators, and independent verification tools. Run the examples from this directory on an idle DGX Spark. The original CUDA C++ measurements used CUDA 13.0.88 and driver 580.159.03.

| Article experiment | Implementation |
| --- | --- |
| Tensor Core recurrence and throughput for seven MMA forms | [CUDA MMA kernels](cuda/bench.cu) and [PTX instructions](cuda/ops.cuh) |
| Scalar instructions and mixed MMA/scalar streams | [Arithmetic probes](cuda/probes/arithmetic.cuh) |
| L1/L2 latency, shared bank conflicts, barriers and fences | [Memory probes](cuda/probes/memory.cuh) |
| Vector copies, `cp.async`, bulk copies and 2D TMA | [Copy probes](cuda/probes/copy.cuh) |
| Warp placement and execution pipeline overlap | [Warp-pair probe](supplemental/warp_pairs.cu) |
| Scheduling progress of four competing warps | [Checkpoint probe](supplemental/warp_progress.cu) |
| Sustained shared-memory read bandwidth | [Vector read probe](supplemental/shared_read.cu) |

The forward-attention roofline uses the measured MMA, exponential and shared-read rates as inputs to an analytical model.

## Reproduce the article measurements

Install Python 3.11 or newer and put the CUDA Toolkit's `nvcc`, `cuobjdump` and the driver's `nvidia-smi` on PATH. The two CUDA runners use the Python standard library. Each compiles for `sm_121a`, checks results against host references, audits the disassembly and runs five shuffled rounds with seven samples per configuration. Use fresh capture directory names on subsequent runs.

```bash
python3 cuda/run.py --output results/mma
python3 cuda/analyze.py results/mma
python3 cuda/run_extended.py --output results/extended
python3 cuda/analyze_extended.py results/extended
```

Each capture produces an `analysis.json` and a `summary.csv`. The MMA suite measures 238 configurations (8,330 samples). The extended suite measures 262 (9,170 samples). The reported SM cycles come from timers inside the kernels. The analyzers compute a median within each round, then average the round medians. At 16 warps and four independent accumulator fragments per warp, the article measured 4.0171 SM cycles per BF16 MMA and 4.0207 per packed NVFP4 MMA.

Run hardware counter and correctness checks separately with Nsight Compute (`ncu`) and Compute Sanitizer. These commands use the binaries built in the timing step. The verification scripts check tensor instruction counts, cache behavior, bank conflicts, memory accesses and shared-memory races.

```bash
python3 cuda/verify.py results/mma
python3 cuda/verify_extended.py results/extended
```

To regenerate the six data figures, use `uv` with the supplied plotting lockfiles. SVGs appear under each capture's `figures/` directory.

```bash
uv run --script --locked plots/figures.py results/mma
uv run --script --locked plots/figures_extended.py results/extended
```

## Warp scheduling and shared-read bandwidth

Build the three standalone probes and capture their output. The summary lists paired and solo cycle counts for eight warp gaps, the slowest warp's checkpoint when the leader reaches checkpoints 32 and 64, and sustained shared reads in bytes per SM cycle.

```bash
mkdir -p build results
mkdir results/scheduling
nvcc -std=c++17 -O3 -arch=sm_121 supplemental/warp_pairs.cu -o build/warp_pairs
nvcc -std=c++17 -O3 -arch=sm_121 supplemental/warp_progress.cu -o build/warp_progress
nvcc -std=c++17 -O3 -arch=sm_121 supplemental/shared_read.cu -o build/shared_read
build/warp_pairs > results/scheduling/warp_pairs.csv
build/warp_progress > results/scheduling/warp_progress.csv
build/shared_read > results/scheduling/shared_read.txt
python3 supplemental/summarize.py results/scheduling
```

For the SMSP placement table, profile three launches in order with active warp IDs (0,1), (0,4) and (0,8). Each active warp executes 8,192 BF16 tensor instructions. The sum counts all of them and the maximum gives the count on the busiest subpartition.

```bash
ncu --clock-control none --cache-control none --csv \
    --metrics smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_tensor.max \
    --log-file results/scheduling/placement.csv build/warp_pairs --placement
```

## CuTe DSL port

The [CuTe DSL implementation](cute/probes) repeats the main and extended suites with CuTe DSL 4.8.1 and inline PTX. Use it to compare compiled instruction streams. The two compilers can produce different schedules and cycle counts. The port pins Python 3.12 and its dependencies and saves the source snapshot with each capture.

```bash
cd cute
uv sync --locked
uv run --locked run.py --output ../results/cute-mma
uv run --locked analyze.py ../results/cute-mma
uv run --locked run_extended.py --output ../results/cute-extended
uv run --locked analyze_extended.py ../results/cute-extended
uv run --locked verify.py ../results/cute-mma
uv run --locked verify_extended.py ../results/cute-extended
```

The optional [Nsight Python runner](cute/profile_nsight.py) profiles an individual CuTe kernel. From `cute/`, run `uv run --locked --extra profile profile_nsight.py --op nvfp4 --output ../results/cute-nvfp4-profile`. The ordinary verification commands use the `ncu` CLI.
