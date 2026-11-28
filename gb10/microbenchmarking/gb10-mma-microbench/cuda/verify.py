"""Independent instruction counts and sanitizer checks, outside timing runs."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("capture", type=Path)
parser.add_argument("--check-only", action="store_true", help="Check existing logs without running the GPU")
args = parser.parse_args()
root = Path(__file__).resolve().parent
capture = args.capture.resolve()
commands = []
checks = {}
for op, peak in [("fp16", "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32_sparsity_off.avg.peak_sustained"),
                 ("nvfp4", "sm__ops_path_tensor_src_fp4_dst_fp32_sparsity_off.avg.peak_sustained")]:
    plan = capture / f"profile-{op}-plan.csv"
    if not args.check_only:
        plan.write_text(f"op,warps,chains,unroll,iterations,samples,variant\n{op},16,4,16,2048,1,0\n")
    command = ["ncu", "--clock-control", "none", "--launch-skip", "3", "--launch-count", "1",
               "--metrics", f"sm__inst_executed_pipe_tensor.sum,sm__cycles_active.max,sm__cycles_elapsed.max,{peak}",
               "--csv", "--log-file", str(capture / f"ncu-{op}.csv"), "build/bench", str(plan)]
    commands.append(command)
    if not args.check_only:
        with (capture / f"ncu-{op}-stdout.csv").open("w") as out, (capture / f"ncu-{op}-stderr.txt").open("w") as err:
            subprocess.run(command, cwd=root, stdout=out, stderr=err, check=True)
    lines = [line for line in (capture / f"ncu-{op}.csv").read_text().splitlines() if line.startswith('"')]
    metrics = {r["Metric Name"]: float(r["Metric Value"].replace(",", "")) for r in csv.DictReader(lines)}
    assert metrics["sm__inst_executed_pipe_tensor.sum"] == 16*4*16*2048
    checks[op] = {"measured_tensor_instructions": metrics["sm__inst_executed_pipe_tensor.sum"],
                  "active_sm_cycles": metrics["sm__cycles_active.max"],
                  "model_peak_flops_per_sm_cycle": metrics[peak]}
for tool in ["memcheck", "racecheck"]:
    command = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", "--log-file",
               str(capture / f"{tool}.log"), "build/bench", str(capture / "validation-plan.csv")]
    commands.append(command)
    if not args.check_only:
        with (capture / f"{tool}-stdout.csv").open("w") as out, (capture / f"{tool}-stderr.txt").open("w") as err:
            subprocess.run(command, cwd=root, stdout=out, stderr=err, check=True)
    log = (capture / f"{tool}.log").read_text()
    assert "0 errors" in log, log
    if tool == "racecheck":
        assert "0 warnings" in log, log
    checks[tool] = "passed"
versions = {}
for tool in ["ncu", "compute-sanitizer"]:
    versions[tool] = subprocess.run([tool, "--version"], check=True, text=True, capture_output=True).stdout
(capture / "verification.json").write_text(json.dumps({"checks": checks, "commands": commands,
                                                        "tool_versions": versions}, indent=2) + "\n")
print(json.dumps(checks, indent=2))
