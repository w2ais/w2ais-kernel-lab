"""Generate IKET-instrumented copies of the four pinned kernels in the output directory."""

import argparse
import ast
from pathlib import Path

MODULES = {
    "5A": "kernel_bf16_5A_readable_stmatrix",
    "5M": "kernel_bf16_5M_tma_tuned",
    "6A+": "kernel_nvfp4_6A_plus_delayed_b",
    "6M": "kernel_nvfp4_6M_port",
}


def instrument(src, out, k, name, cohort):
    offset = 4 if cohort == "primary" else 12
    lines = src.read_text().splitlines()
    before = {}

    def put(line, indent, code):
        before.setdefault(line, []).append(" " * indent + code)

    def mark(line, indent, name):
        if cohort == "coarse" and name not in (
            "begin_output",
            "epilogue",
            "end_output",
        ):
            return
        put(line, indent, "if trace_cta:")
        put(line, indent + 4, f'iket.mark("{name}")')

    def find(s, start=0):
        return next(
            i for i in range(start, len(lines)) if lines[i].strip().startswith(s)
        )

    i = find("import cutlass.cute as cute")
    put(i + 1, 0, "from cutlass.cute.experimental import iket")
    if k in ("5A", "6A+"):
        bf = k == "5A"
        i = find("linear_cta = " if bf else "bid, _, _ = cute.arch.block_idx()") + 1
        put(
            i,
            4,
            "trace_cta = (" + ("linear_cta" if bf else "bid") + f" % 64) == {offset}",
        )
        mark(i, 4, "begin_output")
        i = (
            find(
                "for k_tile in range(k_tile_count - 1)"
                if bf
                else "for tile in range(kt - 1)"
            )
            + 1
        )
        mark(i, 8, "step")
        i = find("if has_future_copy:" if bf else "if has_future:", i)
        mark(i, 16, "wait_begin")
        i = find("cute.arch.sync_threads()", i)
        mark(i + 1, 16, "wait_end")
        i = find("cute.arch.cp_async_wait_group(0)", i + 1)
        mark(i, 4, "epilogue")
        i = find("@cute.jit", i)
        mark(i - 1, 4, "end_output")
    else:
        i = find("warp_idx = cute.arch.make_warp_uniform(warp_idx)") + 1
        put(i, 8, f"trace_cta = (cute.arch.block_idx()[2] % 24) == {offset}")
        i = find("while work_tile.is_valid_tile:", i) + 1
        mark(i, 16, "begin_output")
        i = find("for k_tile in range(0, k_tile_cnt - 1", i) + 1
        mark(i, 20, "step")
        i = find("if k_tile_slice == k_tile_slice_count - 1:", i) + 1
        mark(i, 28, "release_begin")
        i = find("peek_ab_full_status = cutlass.Boolean(1)", i)
        mark(i, 28, "wait_begin")
        i = find("mainloop_pipeline.consumer_wait(", i)
        mark(find(")", i) + 1, 28, "wait_end")
        i = find("copy_atom_r2s = ", i)
        mark(i, 16, "epilogue")
        i = find("tma_store_pipeline.producer_tail()", i) + 1
        mark(i, 16, "end_output")
        i = find("for k_tile in range(0, k_tile_cnt,", i) + 1
        mark(i, 20, "acquire_begin")
        i = find("mainloop_pipeline.producer_acquire", i) + 1
        mark(i, 20, "tma_issue")
        i = find("mainloop_pipeline.producer_commit", i) + 1
        mark(i, 20, "tma_issued")
    result = (
        "\n".join(t for i, l in enumerate(lines) for t in [*before.get(i, []), l])
        + "\n"
    )
    if k in ("5A", "6A+"):
        ident = "gb10_cpasync_kernel" if k == "5A" else "nvfp4_cpasync_kernel"
        result = result.replace(ident, "traced_" + ident)
    ast.parse(result)
    (out / ("traced_" + name + ".py")).write_text(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("cohort", choices=("primary", "replicate", "coarse"))
    args = parser.parse_args()
    modules = args.output / args.cohort / "modules"
    modules.mkdir(parents=True, exist_ok=True)
    for kernel, name in MODULES.items():
        precision = "bf16-gemm" if kernel.startswith("5") else "nvfp4-gemm"
        source = args.output / "gb10" / precision / (name + ".py")
        instrument(source, modules, kernel, name, args.cohort)
