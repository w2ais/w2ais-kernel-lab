# B300 microbenchmarks

Run the upcoming microbenchmarking article's B300 experiments on a single NVIDIA B300 SXM6 AC. All B300 Tensor Core measurements use native `tcgen05.mma.cta_group::1` and FP32 accumulation. The [GB10 suite](../../gb10/microbenchmarking/gb10-mma-microbench/) contains the separate warp-MMA experiments.

## Run on Modal

From the kernel-lab repository root with an authenticated Modal profile:

```bash
uv run --locked modal run b300/microbenchmarking/modal_app.py --suite smoke
uv run --locked modal run b300/microbenchmarking/modal_app.py --suite all
```

The [Modal runner](modal_app.py) requests one B300, compiles for `sm_103a` with CUDA 13.1.1 and downloads each capture into `artifacts/b300/microbenchmarking/`. Select `--suite tcgen05`, `extended`, or `scheduling` to run a single suite. On a B300 with Python 3.11+, the CUDA toolkit and `patch`, the same entry point runs locally:

```bash
python3 b300/microbenchmarking/run.py tcgen05 --output artifacts/b300/native-mma
```

## Native Tensor Core timing

The [CUDA kernel](tcgen05.cu) launches one 128-thread CTA and uses one issuing thread. A and B are staged in shared memory. One 128×128 FP32 accumulator tile and both block-scale operands reside in tensor memory. An untimed MMA initializes the output tile. Then the issuer times 256 further MMAs into that same tile with `clock64()`, committing and waiting for completion after each group of 1, 4, 16 or 64 instructions. This includes the loop, commit, barrier wait and fence. Operand setup, allocation, output readback and validation are outside the interval.

| Format | K | Scale type | Native instruction |
| --- | ---: | --- | --- |
| FP16 / BF16 | 16 | — | `UTCHMMA` |
| TF32 | 8 | — | `UTCHMMA` |
| Padded E2M1 | 32 | UE8M0 | `UTCQMMA` |
| Packed MXFP4 | 64 | UE8M0 | `UTCOMMA` |
| Packed NVFP4 | 64 | UE4M3 | `UTCOMMA.BLOCK16` |

Five rounds shuffle format and batch order with a fixed seed. Each configuration has two checked warmups and seven timed launches per round. The first warmup changes the operand values and, for block-scaled instructions, the A scale. Every launch checks all 16,384 output values. The runner verifies native opcodes in the compiled SASS and summarizes each configuration by averaging the five within-round median cycle counts. FLOPs per SM per cycle equals `2 × 128 × 128 × K` divided by cycles per completed MMA. These are single-CTA instruction-stream measurements.

The [reference results](tcgen05_results.csv) contain all 24 format/batch configurations measured on a B300 SXM6 AC with driver 620.70.06 and CUDA compiler 13.1.115. The round minimum and maximum are the extrema of the five round medians in SM cycles per MMA.

## Scalar, memory and scheduling probes

The extended suite runs the GB10 scalar, pointer-chase, bank-conflict, synchronization and asynchronous copy experiments with the B300 target. The scheduling suite measures scalar warp pairs, dependent FMA checkpoint progress and sustained shared reads. The [target patch](scalar_and_memory.patch) applies to a temporary copy of the GB10 sources and removes the mixed warp-MMA dispatch and tensor warp-pair cases. A scalar pair timing is a contention measurement; identifying an SMSP assignment requires additional evidence.

Across 8,540 timed samples in 244 extended configurations, the independent exponential probe sustained 31.97 results/SM/cycle. The separate seven-pair shared-read measurement took 196,608 median additional cycles for 24 MiB, or 128.00 bytes/SM/cycle.

The run folder holds device metadata, source hashes and per-suite results, including the native raw samples and SASS, extended capture manifests and analysis, and scheduling summaries. Detailed captures stay under the repository's ignored artifacts directory.
