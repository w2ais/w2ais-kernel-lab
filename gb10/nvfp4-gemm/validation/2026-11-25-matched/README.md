# Matched 4096³ NVFP4 run

GB10, BF16 output. Results are means across five shuffled rounds. All 30 processes passed their output checks.

| Kernel | Mean TFLOP/s | % cuBLAS |
| --- | ---: | ---: |
| 6A | 237.32 | 73.10% |
| 7A | 253.84 | 78.19% |
| 6M | 303.25 | 93.41% |
| 7M | 325.47 | 100.26% |
| 8M | 330.38 | 101.77% |
| cuBLAS | 324.64 | 100.00% |

## Within-branch changes

- 6A → 7A: +6.96% by ratio of means. The later kernel was faster in 5 of five rounds.
- 6M → 7M: +7.33% by ratio of means. The later kernel was faster in 5 of five rounds.
- 7M → 8M: +1.51% by ratio of means. The later kernel was faster in 4 of five rounds.
