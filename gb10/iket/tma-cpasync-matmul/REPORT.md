# Why the TMA branch scales better on GB10

For the upcoming NVFP4 post’s 4096³ matmul, the cooperative `cp.async` branch improves from 83.6 to 269.7 TFLOP/s (5A → 6A+, **3.23×**). The persistent TMA branch improves from 85.8 to 306.9 TFLOP/s (5M → 6M, **3.58×**). Nsight Compute measured a full memory input/output (MIO) instruction queue stalling warps for **16.54% of active warp cycles in 6A+ and 0.81% in 6M**. On 6A+, they issue the global-to-shared copies for both values and scales and load their own MMA fragments from shared memory. On 6M, a dedicated producer issues the four global-to-shared loads while the compute warps work on an earlier stage.

## Where the time goes

Both branches go from BK=32 to BK=128. Each NVFP4 K-tile covers four times as much K as its BF16 predecessor and issues the same number of warp MMA instructions per output tile per K-step. Across the launch, Nsight Compute counted 33,554,432 tensor-pipe instructions in each BF16 kernel and 8,388,608 in each NVFP4 kernel.

| Measurement | BF16 5A | NVFP4 6A+ | BF16 5M | NVFP4 6M |
|---|---:|---:|---:|---:|
| IKET median steady-state K-step, µs | 1.06 | 1.40–1.42 | 0.58 | 0.48 |
| IKET time per output tile before epilogue, µs, coarse trace | 134.0 | 42.9 | 62.6 | 16.3 |
| IKET epilogue per output tile, µs, coarse trace | 1.34 | 1.63 | 1.06 | 1.25 |
| Nsight Compute MIO queue stall, % of active warp cycles | 0.51 | 16.54 | 0.18 | 0.81 |
| Nsight Compute short-scoreboard stall, % of active warp cycles | 1.30 | 10.40 | 5.75 | 1.83 |
| Tensor pipe active, % of peak sustained elapsed cycles | 75.6 | 55.8 | 78.0 | 60.4 |

Compare times **within each branch**. An A-branch CTA produces a 128×256 output tile. An M-branch CTA produces a 128×128 tile and persists across multiple tiles. The coarse traces put the speedup before the epilogue at **3.12×** for the A branch and **3.84×** for the M branch. Output stores still write BF16 in all four kernels.

**6A+.** BF16 5A stages two inputs. NVFP4 6A+ adds separate four-byte-per-lane copies of both scale tensors and loads their fragments from shared memory into registers. Across the full launch the load/store-pipe instruction count falls from 16,429,056 to 6,533,120, while the tensor-pipe instruction count falls fourfold. The ratio rises from 0.49 to 0.78 load/store instructions per tensor instruction (+59%). This describes the instruction mix across all warps and phases. Queue occupancy also depends on when those instructions issue and how quickly they drain.

The compute warp spends about 42–43 ns per steady-state K-step between entering the copy-group wait and leaving the CTA barrier in the two detailed captures. Much of the pressure appears while issuing instructions elsewhere in the step. Nsight Compute’s MIO stall metric counts cycles when an active warp cannot issue because the memory input/output instruction queue is full. In a separate instruction-level capture, 4,283 of 10,610 such samples landed on shared-to-register matrix loads (`LDSM`), and another 3,409 on global-to-shared copies (`LDGSTS`). Together these account for **72.5% of the MIO-stall samples**. The sampled instruction identifies where issue was blocked, which can reflect traffic from earlier instructions. [NVIDIA’s stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler)

**6M.** Per steady-state K-step, the producer spent about **0.49 µs acquiring an empty slot** and **0.03 µs issuing four TMA loads**. The latter interval ends at issue and excludes transfer completion. On the consumer side, checking and waiting for the next full slot took about **0.07–0.08 µs** per step. These intervals overlap across warps. The producer was frequently waiting for consumers to release a slot, while the consumers still experienced some exposed input latency. The instruction-issue pressure seen in 6A+ largely disappears. As a separate test, reducing 6M from four input stages to three changed mean throughput by −0.9%. Adding 5M’s extra consumer synchronization before each release changed it by +1.0%. Neither change reproduced the gap between the branches.

## A scale-copy change that improves 6A+

Each lane copies four scale bytes. In the original lane order, adjacent lanes copy four-byte pieces 16 bytes apart. A warp’s 128 useful bytes touch 16 sectors, four times the four-sector minimum. Walking rows 0, 32, 64 and 96 with the first four lanes puts those four pieces next to one another in the physical scale layout. Only the thread layout of the scale copy changes.

```diff
- cute.make_layout((128, 2), stride=(1, 128))
+ cute.make_layout(((32, 4), 2), stride=((4, 1), 128))
```

Nsight Compute attributed all **4,718,592 excessive theoretical global sectors** to the original 32-bit scale-copy instructions. With the new lane order the scale copies fell from 6,291,456 to 1,572,864 theoretical sectors, exactly the ideal count. The corresponding excessive shared wavefronts at those scale-copy instructions also fell from 1,179,648 to zero. These metrics describe accesses generated by instructions. Reuse in L2 affects how much traffic reaches LPDDR.

In ten paired rounds on the original compiler, the new layout won **10/10**, improving the arithmetic mean from **277.1 to 285.0 TFLOP/s (+2.8%)**. In the same rounds 6M averaged **308.6 TFLOP/s**. All variants passed two full-output comparisons against the FP32 reference rounded to BF16 on random values and scales. The experiment isolates one correctable cost. The rest of the gap includes warp roles, tile shape, synchronization, persistence, and the output path. A copy-mechanism-only attribution would require matching those choices. [Control results](reference.json)

A second run of the reproduction script measured 274.1 → 281.0 TFLOP/s for the lane-order change (+2.5%, nine wins in ten rounds), with 6M at 305.5 TFLOP/s. The 4,718,592 excessive theoretical sectors again fell to zero. Memory-queue stalls accounted for 17.70% of active warp cycles in 6A+ and 0.68% in 6M.

## Measurement details

All captures used one GB10, 4096³, warmed all-one inputs and scales, and unlocked clocks. Output was checked after each captured launch. The headline throughputs are the article’s reference results. The control experiment uses its own baselines. Its ten rounds rotated and reversed execution order, with 80 warmup and 80 timed launches per variant in each round.

IKET required CuTe DSL 4.6.0. Uninstrumented Nsight Compute 2026.2.1, the lane-order experiment and the stage/barrier experiments used the article’s CuTe DSL 4.5.2. In separate CUDA-event checks using the same harness, the A-branch speedup was 3.09× on 4.5.2 and 3.01× on 4.6.0. The M-branch speedup was 3.46× and 3.51×, respectively. Nsight counters came from separate warmed launches with cache flushing disabled. No LPDDR bandwidth percentage was available in the counter capture. [NVIDIA’s IKET guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)

Each detailed capture sampled eight A-branch CTAs or two persistent M-branch CTAs, recording every warp in the sampled CTAs. A second capture used different CTA IDs. Each M-branch cohort processed 43 output tiles. K-step times are medians within each compute warp/output tile followed by the median across those observations. Wait and producer times average intervals within a warp/output tile before taking the median. Producer averages omit the first four stages. The coarse capture used just three compute-warp markers per output tile to check sensitivity to marker overhead. It provides the pre-epilogue and epilogue timings in the table. The traces resolve time in 32 ns increments, so short regions are approximate. Use the separate CUDA-event runs for throughput.

The original compressed Perfetto timelines and a kernel-by-kernel index are in this folder. The reproduction command generates the timestamp JSONs and Nsight Compute reports for a new run. [Trace index and reproduction instructions](README.md) · [Reference measurements](reference.json)
