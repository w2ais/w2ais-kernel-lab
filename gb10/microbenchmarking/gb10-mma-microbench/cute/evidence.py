"""Verify exact source snapshots instead of tying old data to today's checkout."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCES=['bench.py','device.py','capture.py','plans.py','audit.py','evidence.py',
         'run.py','run_extended.py','pyproject.toml','uv.lock','.python-version']


def snapshot(capture):
    paths=[ROOT/name for name in SOURCES]+sorted((ROOT/'probes').glob('*.py'))
    hashes={}
    with zipfile.ZipFile(capture/'sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            name=str(path.relative_to(ROOT)); data=path.read_bytes()
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    return hashes


def verify_sources(capture,manifest=None,current=False):
    capture=Path(capture)
    manifest=manifest or json.loads((capture/'manifest.json').read_text())
    if manifest.get('implementation') == 'cute-dsl' and not current:
        with zipfile.ZipFile(capture / 'sources.zip') as archive:
            for name, digest in manifest['source_sha256'].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f'Source hash mismatch: {name}')
    else:
        source_dir = ROOT if current else ROOT.parent / 'cuda'
        for name, digest in manifest['source_sha256'].items():
            if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Source hash mismatch: {name}')
    return manifest
