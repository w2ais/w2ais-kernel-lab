# GPU kernel lab

Experiments organized by hardware: local NVIDIA GB10 and Modal NVIDIA B300.
Run commands from the repository root.

```text
gb10/flash_attention_decode/kernel.py  Local baseline and future GB10 tuning
b300/flash_attention_decode/kernel.py  Remote baseline and future B300 tuning
shared/attention.py                   Inputs and PyTorch reference
shared/test_attention.py              Correctness checks for both implementations
runners/local.py                      Check, benchmark, and NCU workflows
runners/modal_app.py                  B300 environment and artifact download
artifacts/<hardware>/<kernel>/<run>/  Generated results (ignored by Git)
```

## Local GB10

Install Python dependencies with `uv sync`. The lockfile records resolved versions.
Nsight Compute (`ncu`) must be installed separately and on PATH.

```bash
uv run pytest -q
uv run python -m runners.local gb10 check
uv run python -m runners.local gb10 bench --context 4096
uv run python -m runners.local gb10 profile --context 4096
```

The tests exercise both implementations on the available GPU; this does not
establish that the B300 implementation works on actual B300 hardware.

## Modal B300

```bash
uv sync --extra remote
uv run modal setup
uv run modal run runners/modal_app.py --mode check
uv run modal run runners/modal_app.py --mode bench --context 4096
uv run modal run runners/modal_app.py --mode profile --context 4096
```

The runner uses a CUDA 13 image, explicitly installs Nsight Compute, requests
`gpu="B300"`, and downloads generated artifacts after a successful run.
The remote Torch/Triton versions match the initial local lockfile; keep those
pins aligned when upgrading dependencies. Remote execution incurs Modal charges.

## First experiment: paged KV-cache decode attention

Both hardware folders start with identical copies of the supplied kernel so
future tuning can diverge. This is single-token decode with equal query and KV
head counts, paged caches, and an online softmax. It reads existing cache entries;
it does not append new K/V, implement GQA, or use tensor-core matrix operations.

Inputs must be contiguous CUDA tensors on the same device:

- Q/output: `[batch, heads, head_dim]`
- Optional LSE output: `[batch, heads]`, float32, passed as `solve(..., lse=lse)`.
  Uses base-2 scores and `exp2` internally; stores natural-log LSE as
  `(running_max + log2(normalizer)) * ln(2)`. Empty contexts give `-inf`.
  Existing callers may omit it. Shared tests, benchmarks, and profiles request it.
- K/V caches: `[pages, page_size, heads, head_dim]`
- Block table: `[batch, max_blocks_per_seq]`, integer physical page indices
- Context lengths: `[batch]`, integers in `[0, max_blocks_per_seq * page_size]`

Referenced page IDs must be valid. Empty contexts produce zero output. The caller
owns validation; the supplied launch wrapper is preserved without extra checks.
The independent reference gathers logical tokens from shuffled physical pages.
Tests cover partial tiles/pages, mixed and empty lengths, non-power-of-two head
sizes, three floating-point dtypes, and empty batch/head dimensions.

## Profiling and results

Profiling compiles and warms up first, then captures one kernel launch using
CUDA profiler start/stop. All NCU runs explicitly use `--clock-control none`,
including Modal runs. This avoids NCU attempting to change GPU clock frequencies.
Clock variation can add timing noise: repeat measurements, inspect clocks in the
report, and use the separate benchmark workflow for latency comparisons.

Each successful profile saves `profile.ncu-rep`, text/CSV detail exports,
`ncu.log`, and `metadata.json`. Metadata records the input configuration, GPU,
software versions, NCU version, Git commit, and dirty-worktree status. A commit
plus dirty status does not preserve uncommitted source; commit experiments when
recording results you want to reproduce. Benchmark output includes median,
20th-percentile, and 80th-percentile latency in milliseconds.

Use `--ncu-set full` for more metrics (and greater profiling overhead), or
`--output <directory>` with the local runner to choose the artifact destination.
Do not compare NCU-instrumented timings directly to ordinary benchmark timings.

References: [Modal GPU configuration](https://modal.com/docs/guide/gpu),
[NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/),
[NCU clock control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#clock-control).
