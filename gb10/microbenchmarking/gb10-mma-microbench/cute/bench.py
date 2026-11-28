"""Launch CuTe DSL probes from a CSV plan; validate every measured sample."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import numpy as np
import cutlass
from cutlass import Float32, Float64, Int32, Uint32, Uint64
from device import Buffer, check, compile_probe, cu, device_info, initialize, resources
from probes.mma import SPECS, measure
from probes.arithmetic import arithmetic, mixed
from probes.memory import pointer_chase, shared_chase, synchronization
from probes.copy import copies

MMA_FIELDS = 'op warps chains unroll iterations variant sample cycles sm_id mmas flops_per_mma mma_per_sm_cycle flops_per_sm_cycle registers_per_thread shared_bytes local_bytes resident_cta_capacity max_abs_error'.split()
EXT_FIELDS = 'family op warps chains size depth iterations variant sample cycles sm_id work unit registers_per_thread shared_bytes local_bytes max_abs_error nanoseconds'.split()


@dataclass(frozen=True)
class Case:
    family: str = 'mma'
    op: str = 'fp16'
    warps: int = 16
    chains: int = 4
    unroll: int = 16
    size: int = 0
    depth: int = 0
    iterations: int = 2048
    samples: int = 7
    variant: int = 0

    @classmethod
    def from_row(cls,row):
        return cls(**{k:v if k in ('family','op') else int(v) for k,v in row.items()})


def scalar_reference(op,lane,chain,steps,variant=0):
    value = .25 + .125*((lane+chain)%4)
    if op == 'rcp':
        value = 1.25 + ((lane+chain)%4)
        for _ in range(min(steps,128)):
            value = float(np.float32(1)/np.float32(np.float32(value)+np.float32(1)))
    elif op == 'exp2':
        for _ in range(min(steps,128)):
            value = float(np.float32(math.exp2(-float(np.float32(value)))))
    elif variant == 0:
        value = 1-value if steps%2 else value
    else:
        dtype = {'dfma':float,'ffma':np.float32,'hfma2':np.float16}[op]
        for _ in range(min(steps,128)):
            value = float(dtype(.5*value+1))
    return 2*value if op == 'hfma2' else value


def compare(actual,expected,tolerance=1e-5):
    actual=np.asarray(actual); expected=np.asarray(expected)
    if not np.all(np.isfinite(actual)) or not np.all(np.abs(actual-expected) <= tolerance*np.maximum(1,np.abs(expected))):
        raise AssertionError(f'Incorrect result: actual={actual.ravel()[:8]}, expected={expected.ravel()[:8]}')
    return float(np.max(np.abs(actual-expected)))


class Bench:
    def __init__(self,artifacts=None):
        self.context=initialize()
        self.timing=Buffer(4,np.uint64)
        self.float_output=Buffer(2048,np.float64)
        self.uint_output=Buffer(1024,np.uint64)
        self.source=Buffer(16384,np.uint32)
        self.destination=Buffer(16384,np.uint32)
        self.tensor_map=Buffer(128,np.uint8)
        self.chains={}
        self.compiled={}
        self.attributes={}
        self.artifacts=artifacts

    def close(self):
        for buffer in [self.timing,self.float_output,self.uint_output,self.source,self.destination,self.tensor_map]:
            buffer.close()
        for buffer,_ in self.chains.values():
            buffer.close()
        check(cu.cuDevicePrimaryCtxRelease(0))

    def prepare(self,c):
        """Compile and prepare inputs outside both timing and NVTX ranges."""
        if c.warps not in (1,2,4,8,16,32) or c.chains not in (1,2,4,8) or c.iterations<1 or c.samples<1:
            raise ValueError(f'Invalid case: {c}')
        tp=self.timing.pointer(Uint64)
        fo=self.float_output.pointer(Float64)
        uo=self.uint_output.pointer(Uint64)
        unit=''
        if c.family == 'mma':
            if c.unroll not in (1,16) or c.variant not in (0,1):
                raise ValueError(c)
            k,one,half=SPECS[c.op]
            args=(c.op,c.chains,c.unroll,Int32(c.iterations),Uint32(half if c.variant else one),tp,self.float_output.pointer(Float32))
            kernel=measure
            key=f'mma-{c.op}-w{c.warps}-c{c.chains}-u{c.unroll}'
            work=c.warps*c.chains*c.unroll*c.iterations
            expected=sum(j+i+c.iterations*c.unroll*k*(.25 if c.variant else 1) for j in range(c.chains) for i in range(4))
            def validate():
                values=self.float_output.download().view(np.float32)[:c.warps*32]
                return compare(values,expected,2e-6)
        elif c.family == 'alu':
            kernel=arithmetic
            args=(c.op,c.chains,Int32(c.iterations),Float64(.5 if c.variant else -1),tp,fo)
            key=f'alu-{c.op}-w{c.warps}-c{c.chains}'
            work=c.warps*c.chains*16*c.iterations
            unit='warp_rcp_fadd_pair' if c.op=='rcp' else 'warp_instruction'
            expected=[sum(scalar_reference(c.op,lane,j,c.iterations*16,c.variant) for j in range(c.chains)) for lane in range(c.warps*32)]
            def validate():
                return compare(self.float_output.download()[:c.warps*32],expected,1e-12 if c.op=='dfma' else 1e-5)
        elif c.family == 'mixed':
            if c.warps !=16 or c.chains!=4 or c.depth not in (1,4,8) or c.variant not in (0,1,2):
                raise ValueError(c)
            kernel=mixed
            args=(c.op,c.depth,c.variant,Int32(c.iterations),Float64(-1),tp,fo)
            key=f'mixed-{c.op}-r{c.depth}-mode{c.variant}'
            work=16*4*16*c.iterations; unit='base_iteration'
            expected=np.zeros((512,2))
            if c.variant !=1:
                expected[:,0]=sum(j+k+c.iterations*16*16 for j in range(4) for k in range(4))
            if c.variant !=0:
                expected[:,1]=[sum(scalar_reference(c.op,lane,j+r,c.iterations*16) for j in range(4) for r in range(c.depth)) for lane in range(512)]
            def validate():
                values=self.float_output.download()[:1024].reshape(512,2)
                compare(values[:,0],expected[:,0],1e-12)
                return compare(values[:,1],expected[:,1])
        elif c.family=='memory':
            if c.size<4096 or c.size>268435456 or c.size%128 or c.warps!=1 or c.op not in ('ca','cg'):
                raise ValueError(c)
            if c.size not in self.chains:
                buffer=Buffer(c.size//8,np.uint64)
                # Explicit seeded PCG64 permutation. The C++ archive used
                # std::shuffle/mt19937, so pointer order is intentionally recorded
                # as a new capture, never mixed with the historical samples.
                order=np.random.Generator(np.random.PCG64(20261105+c.size)).permutation(c.size//128)
                buffer.host[order*16]=int(buffer.address)+np.roll(order,-1).astype(np.uint64)*128
                buffer.upload(); self.chains[c.size]=(buffer,order)
            buffer,order=self.chains[c.size]
            work=c.iterations*32; unit='dependent_load'
            first=int(buffer.address)+int(order[0])*128
            expected=int(buffer.address)+int(order[work%len(order)])*128
            kernel=pointer_chase
            args=(c.op,Uint64(first),Int32(c.iterations),tp,uo)
            key=f'memory-{c.op}'
            def validate():
                actual=int(self.uint_output.download()[0])
                if actual!=expected: raise AssertionError('Incorrect pointer chain')
                return 0.
        elif c.family=='shared':
            if c.warps!=1 or c.depth not in (0,1,2,4,8,16,32): raise ValueError(c)
            kernel=shared_chase
            args=(Int32(c.depth),Int32(c.iterations),tp,uo)
            key='shared'; work=c.iterations*32; unit='warp_load'
            def validate():
                if not np.array_equal(self.uint_output.download()[:32],np.arange(32)*4*c.depth):
                    raise AssertionError('Incorrect shared chain')
                return 0.
        elif c.family=='sync':
            if c.op not in ('cta','mbarrier','fence'): raise ValueError(c)
            kernel=synchronization
            args=(c.op,Int32(c.iterations),tp,uo)
            key=f'sync-{c.op}-w{c.warps}'; work=c.iterations*16; unit='sync_round'
            def validate():
                if self.uint_output.download()[0]!=0: raise AssertionError('Incorrect phase')
                return 0.
        elif c.family=='copy':
            if c.warps!=4 or c.size not in (1024,4096,8192) or c.depth not in (1,2,4,8): raise ValueError(c)
            total=c.size*c.depth; words=total//4
            self.source.host[:words]=np.arange(words,dtype=np.uint32)+0x12340000
            self.source.upload(); self.destination.host.fill(0); self.destination.upload()
            descriptor=check(cu.cuTensorMapEncodeTiled(cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
                cu.cuuint32_t(2),int(self.source.address),
                [cu.cuuint64_t(v) for v in [64,total//256]], [cu.cuuint64_t(256)],
                [cu.cuuint32_t(v) for v in [64,c.size//256]], [cu.cuuint32_t(1),cu.cuuint32_t(1)],
                cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
                cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
                cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
                cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
            check(cu.cuMemcpyHtoD(self.tensor_map.address,descriptor.getPtr(),128))
            kernel=copies
            args=(c.op,total,self.tensor_map.pointer(Uint64),self.source.pointer(Uint32),self.destination.pointer(Uint32),
                  Int32(c.size),Int32(c.depth),Int32(c.iterations),tp)
            key=f'copy-{c.op}-bytes{total}'; work=total*c.iterations; unit='byte'
            def validate():
                if not np.array_equal(self.destination.download()[:words],self.source.host[:words]):
                    raise AssertionError(f'Incorrect copy: {c}')
                return 0.
        else:
            raise ValueError(c.family)
        if key not in self.compiled:
            print(f'Compile {key}',file=sys.stderr,flush=True)
            self.compiled[key]=compile_probe(kernel,args,c.warps*32,key)
        compiled=self.compiled[key]
        def launch():
            compiled(args)
        return key,compiled,launch,validate,work,unit

    def collect(self,c,warmups):
        key,compiled,launch,validate,work,unit=self.prepare(c)
        for sample in range(-warmups,c.samples):
            self.timing.host.fill(0); self.timing.upload()
            launch(); check(cu.cuCtxSynchronize())
            timing=self.timing.download()
            error=validate()
            if key not in self.attributes:
                self.attributes[key]=resources(compiled,c.warps*32,self.artifacts,key,int(timing[3]))
                if self.attributes[key]['local_bytes']:
                    raise AssertionError(f'Register spill in {key}')
            if sample<0: continue
            row={k:getattr(c,k) for k in ('family','op','warps','chains','unroll','size','depth','iterations','variant')}
            row.update(sample=sample,cycles=int(timing[0]),sm_id=int(timing[1]),nanoseconds=int(timing[2]),
                       max_abs_error=error,work=work,unit=unit,**self.attributes[key])
            if c.family=='mma':
                flops=2*16*8*SPECS[c.op][0]
                row.update(mmas=work,flops_per_mma=flops,mma_per_sm_cycle=work/int(timing[0]),
                           flops_per_sm_cycle=work*flops/int(timing[0]))
            yield row


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('plan',type=Path,nargs='?')
    parser.add_argument('--suite',choices=['mma','extended'],default='mma')
    parser.add_argument('--artifacts',type=Path)
    parser.add_argument('--info',action='store_true')
    args=parser.parse_args()
    bench=Bench(args.artifacts)
    try:
        if args.info:
            import json
            print(json.dumps(device_info())); return
        if not args.plan: parser.error('A plan CSV is required')
        writer=csv.DictWriter(sys.stdout,fieldnames=MMA_FIELDS if args.suite=='mma' else EXT_FIELDS,extrasaction='ignore')
        writer.writeheader()
        with args.plan.open() as f:
            for row in csv.DictReader(f):
                for result in bench.collect(Case.from_row(row),3 if args.suite=='mma' else 2):
                    writer.writerow(result); sys.stdout.flush()
    finally:
        bench.close()


if __name__=='__main__':
    main()
