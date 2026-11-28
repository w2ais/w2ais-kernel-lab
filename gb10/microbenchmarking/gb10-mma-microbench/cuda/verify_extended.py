"""Profile cache paths / bank conflicts and sanitize separately from timing."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

from run_extended import plan

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    capture = args.capture.resolve()
    manifest = json.loads((capture / "manifest.json").read_text())
    assert "finished_utc" in manifest
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    if not args.check_only:
        assert hashlib.sha256((ROOT / "build/extended").read_bytes()).hexdigest() == manifest["executable_sha256"]
    commands, checks = [], {}
    sass = gzip.open(capture / "sass.txt.gz", "rt").read()
    mixed = []
    for section in sass.split("Function : ")[1:]:
        symbol = section.splitlines()[0].strip()
        m = re.search(r"mixedIL3Alu(\d+)ELi(\d+)ELi(\d+)E", symbol)
        if not m:
            continue
        op, ratio, mode = map(int, m.groups())
        lines = [(int(a, 16), line) for line in section.splitlines()
                 if (address := re.search(r"/\*([0-9a-f]+)\*/\s+", line))
                 for a in [address.group(1)]]
        loops = []
        for end, line in lines:
            target = re.search(r"\bBRA(?:\.U)?\s+(?:[^;]*?,\s*)?(0x[0-9a-f]+)\s*;", line)
            if target and int(target.group(1), 16) < end:
                loops.append("\n".join(line for a, line in lines if int(target.group(1),16) <= a <= end))
        assert len(loops) == 1, symbol
        tensor = len(re.findall(r"\bHMMA\.", loops[0]))
        scalar = len(re.findall(r"\bFFMA\b" if op == 0 else r"\bMUFU.EX2\b", loops[0]))
        assert tensor == (0 if mode == 1 else 64), symbol
        assert scalar == (0 if mode == 0 else 64*ratio), symbol
        mixed.append({"symbol": symbol, "mma_count": tensor, "scalar_count": scalar})
    assert len(mixed) == 18
    checks["mixed_sass"] = mixed

    profiles = []
    for policy, size in [("ca",4096), ("cg",1048576), ("cg",268435456)]:
        profiles.append((f"cache-{policy}-{size}",
            ("memory",policy,1,1,size,0,max(size//128,65536)//32+7,1,0),
            ["l1tex__t_sector_hit_rate.pct", "lts__t_sector_op_read_hit_rate.pct", "dram__bytes_read.sum",
             "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
             "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"]))
    for stride in [0,1,8,32]:
        profiles.append((f"banks-{stride}", ("shared","lds",1,1,4096,stride,512,1,0),
            ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"]))
    for name, case, metrics in profiles:
        path = capture / f"profile-{name}-plan.csv"
        if not args.check_only:
            plan(path, [case])
        command = ["ncu", "--clock-control", "none", "--cache-control", "none",
                   "--launch-skip", "2", "--launch-count", "1", "--metrics", ",".join(metrics),
                   "--csv", "--log-file", str(capture / f"ncu-{name}.csv"),
                   "build/extended", str(path)]
        commands.append(command)
        if not args.check_only:
            print(f"Profiling {name}", flush=True)
            with (capture / f"ncu-{name}-stdout.csv").open("w") as out, (capture / f"ncu-{name}-stderr.txt").open("w") as err:
                subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
        lines = [line for line in (capture / f"ncu-{name}.csv").read_text().splitlines() if line.startswith('"')]
        # A hit ratio with zero requests, or an unsupported device counter,
        # can be n/a. Retain it as missing, never interpret it as zero.
        measured = {r["Metric Name"]: (None if r["Metric Value"] == "n/a" else
                     float(r["Metric Value"].replace(",", ""))) for r in csv.DictReader(lines)}
        assert set(metrics) <= measured.keys(), name
        if name.startswith("cache-"):
            hits = measured["lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"]
            misses = measured["lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"]
            measured["l1tex_read_l2_hit_pct"] = 100*hits/(hits+misses) if hits+misses else None
        checks[name] = measured

    for tool in ["memcheck", "racecheck"]:
        command = ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", "--log-file",
                   str(capture / f"{tool}.log"), "build/extended", str(capture / "validation-plan.csv")]
        commands.append(command)
        if not args.check_only:
            print(f"Running {tool}", flush=True)
            with (capture / f"{tool}-stdout.csv").open("w") as out, (capture / f"{tool}-stderr.txt").open("w") as err:
                subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
        log = (capture / f"{tool}.log").read_text()
        assert "0 errors" in log, log[-3000:]
        if tool == "racecheck":
            assert "0 warnings" in log, log[-3000:]
        checks[tool] = "passed"
    checks["cache-path-expectations"] = {
        "small_ca_hits_l1": checks["cache-ca-4096"]["l1tex__t_sector_hit_rate.pct"] > 99,
        "cg_hits_l2": checks["cache-cg-1048576"]["l1tex_read_l2_hit_pct"] > 99,
        "large_cg_misses_l2": checks["cache-cg-268435456"]["l1tex_read_l2_hit_pct"] < 10}
    conflict = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    assert checks["banks-0"][conflict] == checks["banks-1"][conflict] == 0
    assert checks["banks-32"][conflict] > checks["banks-8"][conflict] > 0
    versions = {name: subprocess.run([name, "--version"], check=True, text=True, capture_output=True).stdout
                for name in ["ncu", "compute-sanitizer"]}
    (capture / "verification.json").write_text(json.dumps({"checks": checks, "commands": commands,
                                                           "tool_versions": versions}, indent=2)+"\n")
    print(json.dumps({k:v for k,v in checks.items() if k != "mixed_sass"}, indent=2))


if __name__ == "__main__":
    main()
