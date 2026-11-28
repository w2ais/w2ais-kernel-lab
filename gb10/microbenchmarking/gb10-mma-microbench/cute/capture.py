"""Run independent, shuffled CuTe DSL rounds and retain their evidence."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import zipfile
from evidence import ROOT,snapshot
from audit import collect
from plans import *


def text(command):
    return subprocess.run(command,cwd=ROOT,check=True,text=True,capture_output=True).stdout


def plan(path,rows,fields):
    with path.open('w') as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rows)


def main(suite):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--seed',type=int,default=20261105)
    args=parser.parse_args()
    if args.rounds<1 or args.samples<1: parser.error('rounds and samples must be positive')
    processes=text(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if processes.strip(): raise RuntimeError('GPU has a compute process; rerun when idle.\n'+processes)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    artifacts=ROOT/'build'/out.name
    if artifacts.exists(): raise FileExistsError(f'Use a unique capture name: {artifacts}')
    artifacts.mkdir(parents=True)
    fields=MMA_FIELDS if suite=='mma' else EXT_FIELDS
    validation=mma_validation() if suite=='mma' else extended_validation()
    configs=mma_configurations(args.samples) if suite=='mma' else extended_configurations(args.samples)
    manifest={'implementation':'cute-dsl','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'platform':platform.platform(),'python':platform.python_version(),'architecture':'sm_121a',
        'device':json.loads(text([sys.executable,'bench.py','--info'])),
        'versions':{name:importlib.metadata.version(name) for name in ['nvidia-cutlass-dsl','cuda-bindings','numpy']},
        'rounds':args.rounds,'samples':args.samples,'samples_per_point_per_round':args.samples,
        'seed':args.seed,'suite':suite,'warmups':3 if suite=='mma' else 2,
        'compile_options':'--gpu-arch=sm_121a --generate-line-info --keep-ptx --keep-cubin',
        'source_sha256':snapshot(out),'clock_control':'unchanged, unlocked',
        'processes_before':processes,'round_clock_samples':[],'commands':[],
        'validation_cases':len(validation),'configurations':len(configs),
        'pointer_permutation':'NumPy PCG64; seed=20261105+address_span_bytes',
        'protocol':'one CTA; direct device clock64; fresh process per round; shuffled configurations; validate every launch; profiling separate'}
    clock=['nvidia-smi','--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw','--format=csv']
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    def run(name,rows):
        path=out/f'{name}-plan.csv'; plan(path,rows,fields)
        command=[sys.executable,'bench.py',str(path),'--suite',suite,'--artifacts',str(artifacts)]
        manifest['commands'].append(command); save()
        with (out/f'{name}.csv').open('w') as f,(out/f'{name}.log').open('w') as err:
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=err,check=True)
    print(f'Validate {len(validation)} cases',flush=True)
    run('validation',validation)
    for rid in range(1,args.rounds+1):
        rows=list(configs); random.Random(args.seed+rid).shuffle(rows)
        print(f'Round {rid}/{args.rounds}: {len(rows)} configurations',flush=True)
        before=text(clock); run(f'round-{rid}',rows)
        manifest['round_clock_samples'].append({'round':rid,'before':before,'after':text(clock)})
        save()
    checks=collect(artifacts,out)
    # Keep source PTX and resource metadata without shipping CUDA executables.
    with zipfile.ZipFile(out/'compiled-evidence.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(artifacts.iterdir()):
            if path.suffix in ('.ptx','.json'): z.write(path,path.name)
    manifest['compiled_variants']=len(checks)
    manifest['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat(); save()
    print(f'Captured {out}',flush=True)
