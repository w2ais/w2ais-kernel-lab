# GB10 NVFP4 GEMM at 4096³

All 35 processes passed their output checks. Each mean covers five shuffled rounds with 100 conditioning launches, 20 warmups and 50 timed launches per process. Both implementations used all-one inputs for timing and BF16 output.

| Kernel | Mean TFLOP/s | % of cuBLAS |
| --- | ---: | ---: |
| 6A | 236.94 | 72.99% |
| 6A+ | 269.75 | 83.09% |
| 7A | 290.10 | 89.36% |
| 6M | 306.92 | 94.55% |
| 7M | 330.49 | 101.81% |
| 8M | 335.65 | 103.40% |
| cuBLAS | 324.63 | 100.00% |

Delaying the B load (6A → 6A+) raised throughput by 13.8% and won all five rounds. Registers per thread fell from 255 to 241. Nsight Compute recorded zero local-memory loads or stores in 6A+ and 7A.

Switching from four-column to two-row bands on the spill-free schedule (6A+ → 7A) gained 7.5% and won all five rounds. Going from eight-column bands to M-first on the persistent schedule (6M → 7M) gained 7.7% and won all five rounds. In separate warmed launches, L2 read miss rates fell from 7.3% to 3.4% for the first pair and from 6.0% to 2.5% for the second.

### Independent rerun of the published sources

All 35 processes passed a second five-round run. The table pairs the new arithmetic means with a separate Nsight Compute launch-resource capture using the locked environment. The 6A+ and 7A builds in this capture used 240 registers per thread.

| Kernel | Rerun TFLOP/s | Registers per thread | Shared memory per CTA |
| --- | ---: | ---: | ---: |
| 6A | 227.56 | 255 | 81 KiB |
| 6A+ | 265.63 | 240 | 81 KiB |
| 7A | 297.85 | 240 | 81 KiB |
| 6M | 299.22 | 154 | 89 KiB |
| 7M | 327.99 | 168 | 89 KiB |
| 8M | 330.97 | 168 | 89 KiB |
| cuBLAS | 324.37 | — | — |

Nsight Compute counted 249,856 local-memory loads and 16,384 local-memory stores across 4,096 warps for 6A. Both counts were zero for 6A+ and 7A. To repeat the spill capture, run this from `gb10/nvfp4-gemm` after `uv sync --locked`.

```sh
for kernel in 6A 6A+ 7A; do
    ncu --clock-control none --cache-control none --launch-count 1 \
        --kernel-name 'regex:.*nvfp4_cpasync_kernel.*' \
        --metrics smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum \
        .venv/bin/python benchmark.py "$kernel" --profile-only --output "results/spills-$kernel.json"
done
```

## Tile-order size sweep

Each entry is mean TFLOP/s across eight rotated rounds. All four kernels passed full-output checks at every size.

| Size | 6A+ · four columns | 7A · two rows | 6M · eight columns | 7M · M-first |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 265.6 | 298.5 | 302.7 | 336.4 |
| 6144³ | 257.1 | 189.1 | 326.7 | 153.2 |
| 8192³ | 291.5 | 154.8 | 338.7 | 91.1 |

## Other L2 swizzles on the spill-free Ampere-style kernel

A two-row band assigns CTA IDs 0–3 to output tiles `(0,0), (1,0), (0,1), (1,1)`. It visits every column for that pair of rows, then advances to rows 2 and 3. The compute tile, pipeline, register allocation and zero spill traffic are the same as in 6A+ and 7A.

| Schedule | 4096³, 12 fresh-process rounds | 4096³, 24 paired rounds with random inputs | 6144³, 10 paired rounds |
| --- | ---: | ---: | ---: |
| 6A+, four-column band | 270.2 | 275.0 | **257.7** |
| Natural M-first control | 271.4 | 283.1 | 153.6 |
| 7A, two-row band | **294.8** | **294.2** | 188.8 |

All figures are mean TFLOP/s. The fresh-process test used all-one timing inputs, 100 conditioning launches, 20 warmups, and 50 timed launches, with each of the six kernel orders used twice. The two-row band beat both baselines in all 12 rounds. In the random-input run all kernels shared the same seeded inputs, scales and output buffer. The two-row band also won all 24 rounds against both baselines in each of two separately seeded all-one confirmation runs. At 6144³ the four-column schedule won among the five finalists retested.

The initial 4096³ sweep covered column widths 1, 2, 3, 4, 8 and 16, row heights 2, 4 and 8, serpentine M-first traversal, and direct M-first and N-first indexing on both 1-D and 2-D grids. Across 14 position-balanced rounds, two-row and four-row bands averaged 297.4 and 297.1 TFLOP/s, followed by two-column bands at 289.4. All 14 produced exact BF16-rounded reference outputs at 4096³ and on a 5×5 tile grid with partial bands. Compute Sanitizer reported zero memory errors, shared-memory race hazards or synchronization errors for 6A+, the natural-order control and 7A on the partial grid.

In one separately profiled, warmed 4096³ launch per kernel, Nsight Compute measured L2 read miss rates of 7.19% for 6A+, 2.87% for natural M-first and 3.37% for 7A. The fastest kernel had an intermediate miss rate. These aggregate counters cannot identify which operand accounted for the timing difference.

Run two-row 7A with `uv run --locked python benchmark.py 7A --output results/two-row.json`. Pass `--group 1` to reproduce the natural M-first control. Both use the same spill-free mainloop as 6A+.
