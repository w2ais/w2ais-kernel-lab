"""Check the native instruction loops, not just the intended PTX operations."""
from pathlib import Path
import gzip
import json
import re


def backward_loops(sass):
    lines=[(int(m.group(1),16),line) for line in sass.splitlines()
           if (m:=re.search(r'/\*([0-9a-f]+)\*/\s+',line))]
    result=[]
    for end,line in lines:
        branch=re.search(r'\bBRA(?:\.U)?\s+(?:[^;]*?,\s*)?(0x[0-9a-f]+)\s*;',line)
        if branch and int(branch.group(1),16)<end:
            start=int(branch.group(1),16)
            result.append('\n'.join(text for addr,text in lines if start<=addr<=end))
    return result


def count(text,pattern):
    return len(re.findall(pattern,text))


def audit_file(path):
    name=path.stem; sass=path.read_text()
    local=[line.strip() for line in sass.splitlines() if re.search(r'\b(?:LDL|STL)\b',line)]
    if local: raise AssertionError(f'Local memory invalidates {name}: {local[:2]}')
    result={'name':name,'local_load_store':local}
    if name.startswith('mma-'):
        m=re.fullmatch(r'mma-(.+)-w(\d+)-c(\d+)-u(\d+)',name)
        expected=int(m[3])*int(m[4]); actual=count(sass,r'\b[HOQ]MMA\.')
        assert actual==expected,(name,expected,actual)
        result.update(static_mma_instructions=actual)
    elif name.startswith('alu-'):
        m=re.fullmatch(r'alu-(.+)-w(\d+)-c(\d+)',name)
        op,chains=m[1],int(m[3]); loops=backward_loops(sass)
        pattern={'ffma':r'\bFFMA\b','dfma':r'\bDFMA\b','hfma2':r'\bHFMA2\b',
                 'exp2':r'\bMUFU\.EX2\b','rcp':r'\bMUFU\.RCP\b'}[op]
        matches=[loop for loop in loops if count(loop,pattern)]
        assert len(matches)==1,(name,'loops',len(matches))
        actual=count(matches[0],pattern)
        assert actual==16*chains,(name,actual,16*chains)
        result.update(main_opcode_count=actual,operation=op,chains=chains)
        if op=='rcp':
            additions=count(matches[0],r'\bFADD\b')
            assert additions==actual,(name,additions)
            result['recurrence_additions']=additions
    elif name.startswith('mixed-'):
        m=re.fullmatch(r'mixed-(.+)-r(\d+)-mode(\d+)',name)
        op,ratio,mode=m[1],int(m[2]),int(m[3])
        pattern=r'\bFFMA\b' if op=='ffma' else r'\bMUFU\.EX2\b'
        matches=[loop for loop in backward_loops(sass) if count(loop,r'\bHMMA\.') or count(loop,pattern)]
        assert len(matches)==1,(name,'loops',len(matches))
        tensor=count(matches[0],r'\bHMMA\.'); scalar=count(matches[0],pattern)
        assert tensor==(0 if mode==1 else 64),(name,'mma',tensor)
        assert scalar==(0 if mode==0 else 64*ratio),(name,'scalar',scalar)
        result.update(mma_count=tensor,scalar_count=scalar)
    elif name.startswith('memory-'):
        assert count(sass,r'\bLDG\.')>=32,name
    elif name=='shared':
        assert count(sass,r'\bLDS\b')>=32,name
    return result


def collect(artifacts,capture):
    paths=sorted(Path(artifacts).glob('*.sass'))
    if not paths: raise ValueError('No disassembly was retained')
    checks=[audit_file(path) for path in paths]
    with gzip.open(capture/'sass.txt.gz','wt') as f:
        for path in paths:
            f.write(f'\n// PROBE {path.stem}\n'+path.read_text())
    (capture/'sass-audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    resources={p.stem:json.loads(p.read_text()) for p in sorted(Path(artifacts).glob('*.json'))}
    (capture/'resources.json').write_text(json.dumps(resources,indent=2)+'\n')
    return checks
