"""Profile one CuTe probe with Nsight Python, separately from timing captures.

Example: uv run --locked --extra profile profile_nsight.py --op nvfp4 --output /tmp/nvfp4
"""
# Import Nsight before anything that may initialize CUDA or use NVTX.
import nsight
import argparse
import json
from pathlib import Path
import importlib.metadata
from bench import Bench,Case
from device import check,cu


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--family',choices=['mma','alu','mixed','memory','shared','sync','copy'],default='mma')
    parser.add_argument('--op',default='fp16')
    parser.add_argument('--warps',type=int,default=16)
    parser.add_argument('--chains',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=2048)
    parser.add_argument('--size',type=int,default=0)
    parser.add_argument('--depth',type=int,default=0)
    parser.add_argument('--variant',type=int,default=0)
    parser.add_argument('--runs',type=int,default=1)
    parser.add_argument('--metrics',nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.runs<1: parser.error('runs must be positive')
    operations={'mma':{'fp16_k8','fp16','bf16','tf32','fp4_k32','mxfp4','nvfp4'},
                'alu':{'ffma','dfma','hfma2','rcp','exp2'}, 'mixed':{'ffma','exp2'},
                'memory':{'ca','cg'},'shared':{'lds'},'sync':{'cta','mbarrier','fence'},
                'copy':{'vector','cp_async','bulk','tma2d','store'}}
    if args.op not in operations[args.family]:
        parser.error(f'{args.family} supports: {", ".join(sorted(operations[args.family]))}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.with_suffix('.csv').exists():
        parser.error('Use a fresh output prefix')
    metrics=args.metrics or ['gpu__time_duration.sum'] + (
        ['sm__inst_executed_pipe_tensor.sum'] if args.family in ('mma','mixed') else [])
    case=Case(family=args.family,op=args.op,warps=args.warps,chains=args.chains,
              iterations=args.iterations,size=args.size,depth=args.depth,variant=args.variant,samples=1)

    @nsight.analyze.kernel(runs=args.runs,metrics=metrics,clock_control='none',cache_control='none',
                          thermal_mode='off',output_prefix=str(args.output),output_csv=True)
    def profile_probe():
        bench=Bench()
        try:
            _,_,launch,validate,_,_=bench.prepare(case)
            for _ in range(3):
                launch(); check(cu.cuCtxSynchronize()); validate()
            with nsight.annotate('probe'):
                launch()
            check(cu.cuCtxSynchronize())
            validate()
        finally:
            bench.close()

    result=profile_probe()
    frame=result.to_dataframe()
    frame.to_csv(args.output.with_suffix('.csv'),index=False)
    args.output.with_suffix('.json').write_text(json.dumps({
        'case':case.__dict__,'metrics':metrics,'runs':args.runs,
        'clock_control':'none','cache_control':'none',
        'nsight_python':importlib.metadata.version('nsight-python'),
        'purpose':'Independent counters; not part of direct-cycle timing captures'},indent=2)+'\n')
    print(frame.to_string(index=False))


if __name__=='__main__':
    main()
