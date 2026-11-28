"""CUDA allocation, CuTe compilation, and native instruction evidence."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cuda.bindings import driver as cu


def check(result):
    status, *values = result
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'CUDA: {status}')
    return values[0] if len(values) == 1 else tuple(values)


def initialize():
    check(cu.cuInit(0))
    context = check(cu.cuDevicePrimaryCtxRetain(0))
    check(cu.cuCtxSetCurrent(context))
    return context


class Buffer:
    def __init__(self, shape, dtype):
        self.host = np.zeros(shape, dtype=dtype)
        self.address = check(cu.cuMemAlloc(self.host.nbytes))
        self.upload()

    def upload(self):
        check(cu.cuMemcpyHtoD(self.address, self.host.ctypes.data, self.host.nbytes))

    def download(self):
        check(cu.cuMemcpyDtoH(self.host.ctypes.data, self.address, self.host.nbytes))
        return self.host

    def pointer(self, dtype):
        return make_ptr(dtype, int(self.address), cute.AddressSpace.gmem, assumed_align=16)

    def close(self):
        if self.address is not None:
            check(cu.cuMemFree(self.address))
            self.address = None


class Entry:
    def __init__(self, kernel, constants, threads):
        self.kernel, self.constants, self.threads = kernel, constants, threads

    @cute.jit
    def __call__(self, arguments: tuple):
        self.kernel(*self.constants, *arguments).launch(grid=(1,1,1), block=(self.threads,1,1))


class CompiledProbe:
    def __init__(self, kernel, arguments, threads, name):
        self.constant_count = 0
        for arg in arguments:
            if type(arg) not in (str, int):
                break
            self.constant_count += 1
        entry = Entry(kernel, arguments[:self.constant_count], threads)
        dump_dir = Path('build/compiler') / name
        dump_dir.mkdir(parents=True,exist_ok=True)
        self.function = cute.compile(entry, arguments[self.constant_count:],
            options=f'--gpu-arch=sm_121a --generate-line-info --keep-ptx --keep-cubin --dump-dir={dump_dir}')

    def __call__(self, arguments):
        self.function(arguments[self.constant_count:])

    def __getattr__(self, name):
        return getattr(self.function, name)


def compile_probe(kernel, arguments, threads, name='probe'):
    return CompiledProbe(kernel, arguments, threads, name)


def resources(compiled, threads, directory=None, name='probe', dynamic_shared=0):
    cubin = compiled.__cubin__
    if isinstance(cubin, str):
        cubin = Path(cubin).read_bytes()
    if not cubin:
        raise RuntimeError('Compiler did not retain cubin')
    ptx = compiled.__ptx__
    if '\n' not in ptx:
        ptx = Path(ptx).read_text()
    symbol = re.search(r'\.entry\s+(\w+)', ptx).group(1)
    module = check(cu.cuModuleLoadData(cubin))
    try:
        function = check(cu.cuModuleGetFunction(module, symbol.encode()))
        def attr(name):
            return check(cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute, 'CU_FUNC_ATTRIBUTE_'+name), function))
        if dynamic_shared > 48*1024:
            check(cu.cuFuncSetAttribute(function, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_shared))
        result = {'registers_per_thread':attr('NUM_REGS'), 'shared_bytes':attr('SHARED_SIZE_BYTES') + dynamic_shared,
                  'local_bytes':attr('LOCAL_SIZE_BYTES'),
                  'resident_cta_capacity':check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, threads,dynamic_shared)),
                  'symbol':symbol, 'cubin_sha256':hashlib.sha256(cubin).hexdigest()}
    finally:
        check(cu.cuModuleUnload(module))
    if directory:
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        binary=directory/f'{name}.cubin'; binary.write_bytes(cubin)
        (directory/f'{name}.ptx').write_text(ptx)
        sass=subprocess.run(['cuobjdump','--dump-sass',str(binary)],check=True,capture_output=True,text=True).stdout
        (directory/f'{name}.sass').write_text(sass)
        (directory/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def device_info():
    def attr(name):
        return check(cu.cuDeviceGetAttribute(getattr(cu.CUdevice_attribute,'CU_DEVICE_ATTRIBUTE_'+name),0))
    name=check(cu.cuDeviceGetName(128,0)).decode().rstrip('\0')
    return {'name':name,'cc':f'{attr("COMPUTE_CAPABILITY_MAJOR")}.{attr("COMPUTE_CAPABILITY_MINOR")}',
            'sm_count':attr('MULTIPROCESSOR_COUNT'),'l2_bytes':attr('L2_CACHE_SIZE'),
            'shared_bytes_per_sm':attr('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR'),
            'threads_per_sm':attr('MAX_THREADS_PER_MULTIPROCESSOR')}
