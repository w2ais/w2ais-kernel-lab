"""Explicit experiment grids shared by timing, profiling, and validation."""
ALU = ['ffma','dfma','hfma2','rcp','exp2']
COPY = ['vector','cp_async','bulk','tma2d','store']
OPS = ['fp16_k8','fp16','bf16','tf32','fp4_k32','mxfp4','nvfp4']
MMA_FIELDS = ['op','warps','chains','unroll','iterations','samples','variant']
EXT_FIELDS = ['family','op','warps','chains','size','depth','iterations','samples','variant']

def extended_configurations(samples):
    rows = set()
    for op in ALU:
        for warps in [1, 2, 4, 8, 16, 32]:
            for chains in [1, 2, 4, 8]:
                rows.add(("alu", op, warps, chains, 0, 0, 1024, samples, 0))
        for length in [256, 4096]:
            rows.add(("alu", op, 1, 1, 0, 0, length, samples, 0))
    for op in ["ffma", "exp2"]:
        for ratio in [1, 4, 8]:
            for mode in [0, 1, 2]:
                rows.add(("mixed", op, 16, 4, 0, ratio, 1024, samples, mode))
    for size in [4096, 65536, 1048576, 8388608, 67108864, 268435456]:
        for policy in ["ca", "cg"]:
            # Every timed interval traverses the whole random cycle, plus 224
            # nodes. Repeats cannot accidentally time only a cache-sized prefix.
            iterations = max(size // 128, 65536) // 32 + 7
            rows.add(("memory", policy, 1, 1, size, 0, iterations, samples, 0))
    for stride in [0, 1, 2, 4, 8, 16, 32]:
        for length in [128, 512, 2048]:
            rows.add(("shared", "lds", 1, 1, 4096, stride, length, samples, 0))
    for length in [256, 1024, 4096]:
        for warps in [1, 4, 8, 16, 32]:
            rows.add(("sync", "cta", warps, 1, 0, 0, length, samples, 0))
        for op in ["mbarrier", "fence"]:
            rows.add(("sync", op, 1, 1, 0, 0, length, samples, 0))
    for op in COPY:
        for size in [1024, 4096, 8192]:
            for depth in [1, 2, 4, 8]:
                rows.add(("copy", op, 4, 1, size, depth, 128, samples, 0))
    return sorted(rows)

def extended_validation():
    rows = [("alu", op, w, n, 0, 0, 32, 1, v) for op in ALU
            for w, n in [(1, 1), (16, 4)] for v in [0, 1]]
    rows += [("mixed", op, 16, 4, 0, 4, 32, 1, mode)
             for op in ["ffma", "exp2"] for mode in [0, 1, 2]]
    rows += [("memory", op, 1, 1, 1048576, 0, 519, 1, 0) for op in ["ca", "cg"]]
    rows += [("shared", "lds", 1, 1, 4096, stride, 128, 1, 0) for stride in [0, 1, 2, 4, 8, 16, 32]]
    rows += [("sync", op, w, 1, 0, 0, 32, 1, 0) for op, w in [("cta", 1), ("cta", 16), ("mbarrier", 1), ("fence", 1)]]
    rows += [("copy", op, 4, 1, size, depth, 8, 1, 0) for op in COPY for size, depth in [(1024, 1), (8192, 8)]]
    return rows

def mma_configurations(samples):
    rows = {(op,w,n,16,2048,samples,0) for op in OPS
            for w in [1,2,4,8,16,32] for n in [1,2,4,8]}
    for op in OPS:
        for length in [256,1024,4096]:
            for w,n in [(1,1),(16,4)]:
                rows.add((op,w,n,16,length,samples,0))
        for w,n in [(1,1),(16,4)]:
            rows.add((op,w,n,1,32768,samples,0))
            rows.add((op,w,n,16,2048,samples,1))
    return sorted(rows)


def mma_validation():
    return [(op,w,n,16,128,1,v) for op in OPS
            for w,n in [(1,1),(16,4)] for v in [0,1]]
