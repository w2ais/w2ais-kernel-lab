"""Collect launch metadata for the article's 4096³ resource cards.

Run: uv run --locked python tools/profile_kernel_resources.py --output PATH
LaunchStats and Occupancy are collected for one launch per kernel.
Profiler timing is not a new benchmark.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import KERNELS

KEYS = list(KERNELS)


def parse_csv(path):
    raw = path.read_text()
    rows = list(csv.DictReader(raw[raw.index('"ID"'):].splitlines()))
    rows = [r for r in rows if r.get('Metric Name')]
    assert len({r['Kernel Name'] for r in rows}) == 1
    metrics = {r['Metric Name']: r['Metric Value'] for r in rows}
    def value(key): return int(metrics[key].replace(',', ''))
    def decimal(key): return float(metrics[key].replace(',', ''))
    threads = value('Block Size')
    assert threads % 32 == 0
    registers = value('Registers Per Thread')
    shared = value('Dynamic Shared Memory Per Block') + value('Static Shared Memory Per Block')
    active_warps = value('Theoretical Active Warps per SM')
    assert active_warps % (threads // 32) == 0
    return {
        'sm_count': value('# SMs'),
        'resident_ctas_per_sm': active_warps // (threads // 32),
        'theoretical_active_warps_per_sm': active_warps,
        'achieved_active_warps_per_sm': decimal('Achieved Active Warps Per SM'),
        'theoretical_occupancy_percent': decimal('Theoretical Occupancy'),
        'achieved_occupancy_percent': decimal('Achieved Occupancy'),
        'ctas_launched': value('Grid Size'),
        'threads_per_cta': threads,
        'warps_per_cta': threads // 32,
        'registers_per_thread': registers,
        'registers_per_warp': registers * 32,
        'shared_bytes_per_cta': shared,
        'shared_bytes_per_warp_average': shared / (threads // 32),
        'driver_shared_bytes_per_cta': value('Driver Shared Memory Per Block'),
        'grid': rows[0]['Grid Size'],
        'block': rows[0]['Block Size'],
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'collected_utc': datetime.now(timezone.utc).isoformat(),
        'shape_mnk': [4096, 4096, 4096],
        'scope': 'Launch resources and occupancy for current readable sources; not a new performance evaluation.',
        'occupancy_definition': 'Achieved occupancy is the measured average active warps per SM as a percentage of the hardware maximum. Theoretical occupancy and resident CTA capacity are resource limits, including producer warps and driver allocations; neither is a compute-utilization percentage.',
        'waves_definition': 'Nsight Compute Waves Per SM: grid CTAs / (SM count × resident CTA capacity per SM). Fractional values describe the partial final wave. grid_waves is null for persistent kernels, whose CTAs loop over work tiles.',
        'definitions_source': 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/',
        'register_definition': '32-bit registers reported per thread by Nsight Compute, multiplied by 32; not a per-role runtime liveness measurement or allocation-granularity correction.',
        'shared_definition': 'Static plus dynamic application shared memory per CTA. Per-warp value divides this shared pool by ALL launched warps, including the producer. It is not privately allocated to each warp. Driver overhead is recorded separately.',
        'ncu_version': subprocess.check_output(['ncu','--version'],text=True).strip(),
        'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*ROOT.glob('*.py'), *(ROOT/'references').glob('*.py'), Path(__file__).resolve(), ROOT/'uv.lock']},
        'kernels': {},
    }
    previous = json.loads((out/'resources.json').read_text()) if (out/'resources.json').exists() else None
    if previous:
        # Resume completed captures only when all kernel/helper sources and the lock match.
        tool = str(Path(__file__).resolve().relative_to(ROOT))
        assert {k:v for k,v in previous['sources'].items() if k != tool} == {k:v for k,v in report['sources'].items() if k != tool}, 'Source changes require a fresh output directory'
        report['collected_utc'] = previous['collected_utc']
    for key in KEYS:
        cmd = [sys.executable,'benchmark.py',key,'--profile-only','--output',str(out/f'kernel_{key}-run.json')]
        target = out / f'kernel_{key}.csv'
        profile = ['ncu','--kernel-name-base','function','--kernel-name','regex:.*(nvfp4_cpasync_kernel|BF16DerivedNVFP4).*',
                   '--launch-count','1','--section','LaunchStats','--section','Occupancy','--csv','--log-file',str(target),*cmd]
        if previous and key in previous['kernels']:
            saved = previous['kernels'][key]
            assert saved['command'] == profile and target.exists(), 'Capture command changed'
            report['kernels'][key] = saved
            continue
        print(f'Collecting Kernel {key}', flush=True)
        with (out/f'kernel_{key}.log').open('w') as handle:
            subprocess.run(profile,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=180)
        resource = parse_csv(target)
        resource['producer_warps'] = 1 if key.endswith('M') else 0
        resource['persistent'] = key.endswith('M')
        resource['grid_waves'] = None if resource['persistent'] else float(resource['metrics']['Waves Per SM'].replace(',', ''))
        resource['compute_warps'] = 8
        resource['padding_warps'] = 0
        resource['source'] = KERNELS[key] + '.py'
        resource['command'] = profile
        resource['evidence'] = f'kernel_{key}.csv'
        report['kernels'][key] = resource
        (out/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"  {resource['ctas_launched']} CTAs, {resource['warps_per_cta']} warps/CTA, {resource['registers_per_thread']} registers/thread, {resource['shared_bytes_per_cta']} shared bytes/CTA, {resource['achieved_occupancy_percent']}% achieved occupancy",flush=True)


if __name__ == '__main__': main()
