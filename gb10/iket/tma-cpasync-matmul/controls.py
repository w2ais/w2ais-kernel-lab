"""Measure lane order, TMA stage depth, and consumer synchronization controls."""

import argparse
import json
import sys
from pathlib import Path

import torch

from workload import MODULES, SIZE, compile_nvfp4, fill_nvfp4_ones, milliseconds


def make_variants(source, destination):
    destination.mkdir(exist_ok=True)
    ampere = (source / (MODULES["6A+"] + ".py")).read_text()
    modern = (source / (MODULES["6M"] + ".py")).read_text()
    old_layout = "cute.make_layout((128, 2), stride=(1, 128))"
    new_layout = "cute.make_layout(((32, 4), 2), stride=((4, 1), 128))"
    assert ampere.count(old_layout) == modern.count("AB_STAGES = 4") == 1
    (destination / "scale_lanes.py").write_text(ampere.replace(old_layout, new_layout))
    (destination / "three_stages.py").write_text(
        modern.replace("AB_STAGES = 4", "AB_STAGES = 3")
    )
    release = "mainloop_pipeline.consumer_release(mainloop_consumer_state)"
    assert modern.count(release) == 2
    lines = []
    for line in modern.splitlines():
        if line.strip() == release:
            indent = line[: line.index(release)]
            lines.append(
                indent
                + "pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_mma_warps * 32).arrive_and_wait()"
            )
        lines.append(line)
    (destination / "plus_barrier.py").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output.resolve()
    source = out / "gb10/nvfp4-gemm"
    make_variants(source, out / "controls")
    sys.path[:0] = [str(out / "controls"), str(source)]
    from problem import make_problem

    torch.backends.cuda.matmul.allow_tf32 = False
    problem = make_problem(SIZE, SIZE, SIZE, 2026)
    a, b = [problem[k + "_reference"][:, :, 0] for k in ("a", "b")]
    sa, sb = [
        problem[k + "_reference"][:, :, 0].repeat_interleave(16, dim=1)
        for k in ("sfa", "sfb")
    ]
    expected = ((a * sa) @ (b * sb).T).to(torch.bfloat16)
    result = problem["c_storage"][:, :, 0]
    launches = {}
    for name in ("6A+", "scale_lanes", "6M", "three_stages", "plus_barrier"):
        launch = compile_nvfp4(
            problem, MODULES.get(name, name), name in ("6A+", "scale_lanes")
        )
        launches[name] = launch
        for _ in range(2):
            result.fill_(float("nan"))
            launch()
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
    fill_nvfp4_ones(problem)
    for launch in launches.values():
        result.fill_(float("nan"))
        launch()
        torch.testing.assert_close(
            result, torch.full_like(result, SIZE), atol=0, rtol=0
        )

    timings = {name: [] for name in launches}
    names = list(launches)
    for round_index in range(2 * len(names)):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index >= len(names):
            order.reverse()
        for name in order:
            launch = launches[name]
            for _ in range(80):
                launch()
            timings[name].append(2 * SIZE**3 / (1e9 * milliseconds(launch, 80)))
    (out / "control-rounds.json").write_text(json.dumps(timings, indent=2) + "\n")


if __name__ == "__main__":
    main()
