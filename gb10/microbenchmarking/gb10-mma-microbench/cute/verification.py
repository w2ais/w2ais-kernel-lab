"""Independent profiler counters and sanitizer checks for captured probes."""
from pathlib import Path
import argparse
import csv
import json
import importlib.metadata
import subprocess
import sys
from evidence import ROOT,verify_sources
from capture import plan
from plans import MMA_FIELDS,EXT_FIELDS


def metric_values(path):
    lines=[s for s in path.read_text().splitlines() if s.startswith('"')]
    return {r['Metric Name']:None if r['Metric Value']=='n/a' else float(r['Metric Value'].replace(',',''))
            for r in csv.DictReader(lines)}


def main(suite):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    capture=args.capture.resolve()
    manifest=verify_sources(capture)
    assert 'finished_utc' in manifest,'Capture is incomplete'
    modern=manifest.get('implementation')=='cute-dsl'
    if not args.check_only:
        if not modern:
            parser.error('Historical C++ data: use --check-only, or run the verifiers in ../cuda for a new capture.')
        verify_sources(capture,manifest,current=True)
        for package,version in manifest['versions'].items():
            actual=importlib.metadata.version(package)
            if actual!=version:
                raise RuntimeError(f'{package}: capture used {version}, installed {actual}; restore the capture lockfile')
    checks,commands={},[]
    if modern:
        audit=json.loads((capture/'sass-audit.json').read_text())
        assert all(not row['local_load_store'] for row in audit)
        checks['sass_variants']=len(audit)
        if suite=='extended':
            assert len([r for r in audit if r['name'].startswith('mixed-')])==18
    profiles=[]
    if suite=='mma':
        for op,peak in [('fp16','sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32_sparsity_off.avg.peak_sustained'),
                        ('nvfp4','sm__ops_path_tensor_src_fp4_dst_fp32_sparsity_off.avg.peak_sustained')]:
            profiles.append((op,(op,16,4,16,2048,1,0),[
                'sm__inst_executed_pipe_tensor.sum','sm__cycles_active.max','sm__cycles_elapsed.max',peak]))
    else:
        for policy,size in [('ca',4096),('cg',1048576),('cg',268435456)]:
            profiles.append((f'cache-{policy}-{size}',
                ('memory',policy,1,1,size,0,max(size//128,65536)//32+7,1,0),[
                    'l1tex__t_sector_hit_rate.pct','lts__t_sector_op_read_hit_rate.pct','dram__bytes_read.sum',
                    'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum',
                    'lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum']))
        for stride in [0,1,8,32]:
            profiles.append((f'banks-{stride}',('shared','lds',1,1,4096,stride,512,1,0),
                ['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']))
    for name,case,metrics in profiles:
        path=capture/f'profile-{name}-plan.csv'
        command=['ncu','--clock-control','none','--cache-control','none',
                 '--launch-skip','3' if suite=='mma' else '2','--launch-count','1',
                 '--metrics',','.join(metrics),'--csv','--log-file',str(capture/f'ncu-{name}.csv'),
                 sys.executable,'bench.py',str(path),'--suite',suite]
        commands.append(command)
        if not args.check_only:
            plan(path,[case],MMA_FIELDS if suite=='mma' else EXT_FIELDS)
            print(f'Profile {name}',flush=True)
            with (capture/f'ncu-{name}-stdout.csv').open('w') as out,(capture/f'ncu-{name}-stderr.txt').open('w') as err:
                subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=True)
        measured=metric_values(capture/f'ncu-{name}.csv')
        assert set(metrics)<=measured.keys(),name
        if suite=='mma':
            assert measured['sm__inst_executed_pipe_tensor.sum']==2097152
        elif name.startswith('cache-'):
            hits=measured['lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum']
            misses=measured['lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum']
            measured['l1tex_read_l2_hit_pct']=100*hits/(hits+misses) if hits+misses else None
        checks[name]=measured
    for tool in ['memcheck','racecheck']:
        command=['compute-sanitizer','--tool',tool,'--error-exitcode','99','--log-file',str(capture/f'{tool}.log'),
                 sys.executable,'bench.py',str(capture/'validation-plan.csv'),'--suite',suite]
        commands.append(command)
        if not args.check_only:
            print(f'Run {tool}',flush=True)
            with (capture/f'{tool}-stdout.csv').open('w') as out,(capture/f'{tool}-stderr.txt').open('w') as err:
                subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,check=True)
        log=(capture/f'{tool}.log').read_text()
        assert '0 errors' in log,log[-3000:]
        if tool=='racecheck': assert '0 warnings' in log,log[-3000:]
        checks[tool]='passed'
    if suite=='extended':
        conflict='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'
        assert checks['banks-0'][conflict]==checks['banks-1'][conflict]==0
        # Pipeline bookkeeping can add shared requests, but the explicit LDS
        # loop must still produce exactly 7/31 extra conflicts per warp load.
        assert checks['banks-8'][conflict]==7*512*32
        assert checks['banks-32'][conflict]==31*512*32
        assert checks['cache-ca-4096']['l1tex__t_sector_hit_rate.pct']>99
        assert checks['cache-cg-1048576']['l1tex_read_l2_hit_pct']>99
        assert checks['cache-cg-268435456']['l1tex_read_l2_hit_pct']<10
    if not args.check_only:
        versions={tool:subprocess.run([tool,'--version'],check=True,capture_output=True,text=True).stdout
                  for tool in ['ncu','compute-sanitizer']}
        (capture/'verification.json').write_text(json.dumps({'checks':checks,'commands':commands,'tool_versions':versions},indent=2)+'\n')
    print(json.dumps(checks,indent=2))
