"""Run the single-GPU B300 microbenchmarks on Modal and download each capture."""

import io
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GB10 = ROOT / "gb10/microbenchmarking/gb10-mma-microbench"
REMOTE = Path("/opt/b300-microbench")
app = modal.App("kernel-lab-b300-microbench")
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("patch")
    .add_local_dir(GB10 / "cuda", str(REMOTE / "cuda"), ignore=["**/__pycache__/**", "**/build/**"])
    .add_local_dir(GB10 / "supplemental", str(REMOTE / "supplemental"), ignore=["**/__pycache__/**"])
    .add_local_dir(HERE, str(REMOTE / "b300"), ignore=["**/__pycache__/**"])
)


@app.function(gpu="B300", image=image, timeout=1800, max_containers=1, retries=0)
def execute(suite: str, rounds: int, samples: int):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as work:
        output = Path(work) / "capture"
        process = subprocess.run(
            [sys.executable, str(REMOTE / "b300/run.py"), suite, "--output", str(output),
             "--rounds", str(rounds), "--samples", str(samples)],
            cwd=work,
        )
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output))
        return process.returncode, bundle.getvalue()


@app.local_entrypoint()
def main(suite: str = "smoke", rounds: int = 5, samples: int = 7):
    if suite not in {"smoke", "all", "portable", "tcgen05", "extended", "scheduling"}:
        raise ValueError("Unknown benchmark suite")
    if min(rounds, samples) < 1:
        raise ValueError("rounds and samples must be positive")
    status, data = execute.remote(suite, rounds, samples)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/b300/microbenchmarking" / f"{stamp}-{suite}"
    output.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(output)
    print(f"Downloaded {output}")
    if (output / "device.json").exists():
        print((output / "device.json").read_text())
    if status:
        raise RuntimeError(f"Benchmark exited with status {status}. See {output}")
