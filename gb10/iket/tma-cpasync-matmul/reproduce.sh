#!/usr/bin/env bash
# Run the four kernels, instrumentation passes, and controls serially on one GB10.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
base=ee17c2caf216beb7c02e25bdaa22c4e05a95d96b
bench_python=${BENCH_PYTHON:-"$here/../../nvfp4-gemm/.venv/bin/python"}
trace_python=${TRACE_PYTHON:-"$here/.venv/iket/bin/python"}
iket="$(dirname "$trace_python")/run-iket"
ncu=${NCU:-ncu}
for tool in "$bench_python" "$trace_python" "$iket" "$ncu" patch; do
    command -v "$tool" >/dev/null || { echo "Missing $tool. See README.md for setup." >&2; exit 1; }
done
if (( $# > 1 )); then echo "Usage: $0 [new-output-directory]" >&2; exit 2; fi
if (( $# )); then mkdir -p "$(dirname "$1")"; mkdir "$1"; out=$(cd "$1" && pwd)
else mkdir -p "$here/artifacts"; out=$(mktemp -d "$here/artifacts/run-XXXXXX"); fi
echo "Writing results to $out"
git -C "$repo" archive "$base" gb10/bf16-gemm gb10/nvfp4-gemm | tar -x -C "$out"
nv="$out/gb10/nvfp4-gemm"
cp "$nv/kernel_nvfp4_6A_scale_pipeline.py" "$nv/kernel_nvfp4_6A_plus_delayed_b.py"
patch --batch --fuzz=0 -s -p0 -d "$nv" < "$here/checkpoint-6a-plus.diff"
(cd "$out" && sha256sum --check "$here/checkpoints.sha256")
nvidia-smi > "$out/gpu.txt"
"$ncu" --version > "$out/ncu-version.txt"

for cohort in primary replicate coarse; do
    python3 "$here/instrument.py" "$out" "$cohort"
    for k in 5A 6A+ 5M 6M; do
        case "$k" in 5A) depth=520;; 6A+) depth=140;; 5M) depth=13000;; 6M) depth=3500;; esac
        if [[ "$cohort" == coarse ]]; then depth=130; fi
        echo "IKET $cohort $k"
        "$iket" --context-buffer-size 268435456 -o "$out/$cohort/$k" profile \
            --max-ts-cnt-per-warp "$depth" --postprocess all -- \
            "$trace_python" "$here/workload.py" "$out" "$k" trace --cohort "$cohort"
    done
done
metrics=(
    gpu__time_duration.sum sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
    smsp__inst_executed.sum smsp__inst_executed_pipe_tensor.sum
    smsp__inst_executed_pipe_lsu.sum smsp__inst_executed_pipe_tma.sum
    smsp__inst_executed_pipe_alu.sum smsp__inst_executed_pipe_uniform.sum
    lts__t_sectors_op_read.sum lts__t_sectors_op_read_lookup_hit.sum launch__registers_per_thread
    smsp__issue_active.avg.pct_of_peak_sustained_elapsed
)
for reason in barrier short_scoreboard long_scoreboard mio_throttle math_pipe_throttle not_selected wait; do
    metrics+=("smsp__warp_issue_stalled_${reason}_per_warp_active.pct")
done
metric_list=$(IFS=,; echo "${metrics[*]}")
profile=(--profile-from-start off --launch-count 1 --clock-control none --cache-control none)
for k in 5M 6A+ 6M 5A; do
    echo "Nsight Compute and CUDA-event timing $k"
    "$ncu" "${profile[@]}" --csv --print-units base --metrics "$metric_list" \
        --log-file "$out/ncu-$k.csv" "$bench_python" "$here/workload.py" "$out" "$k" ncu
    for python in "$bench_python" "$trace_python"; do
        "$python" "$here/workload.py" "$out" "$k" time
    done
done
echo "Running paired controls"
"$bench_python" "$here/controls.py" "$out"
pc_metrics=smsp__pcsamp_warps_issue_stalled_mio_throttle,smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_sample_count
memory_metrics=derived__memory_l1_wavefronts_shared_excessive,derived__memory_l2_theoretical_sectors_global_excessive,memory_l2_theoretical_sectors_global,memory_l2_theoretical_sectors_global_ideal
for variant in 6A+ 6M scale_lanes; do
    kernel_args=("$variant" ncu)
    if [[ "$variant" == scale_lanes ]]; then kernel_args=(6A+ ncu --scale-lanes); fi
    "$ncu" "${profile[@]}" --section SourceCounters --export "$out/pcs-$variant" \
        "$bench_python" "$here/workload.py" "$out" "${kernel_args[@]}"
    report=(-i "$out/pcs-$variant.ncu-rep" --page source --print-source sass --csv)
    "$ncu" "${report[@]}" --metrics "$pc_metrics" > "$out/pcs-$variant.csv"
    "$ncu" "${report[@]}" --metrics "$memory_metrics" > "$out/memory-$variant.csv"
done
python3 "$here/analyze.py" "$out"
