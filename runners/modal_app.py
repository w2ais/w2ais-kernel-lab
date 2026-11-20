"""B300 execution: modal run runners/modal_app.py --mode check|bench|profile."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("kernel-lab-b300")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-compute-13-0")
    .uv_pip_install("torch==2.15.0", "triton==3.9.0", "pytest>=8", "numpy>=2.5.4")
    .env({"PYTHONPATH": "/workspace", "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .workdir("/workspace")
    .add_local_dir(ROOT / "shared", "/workspace/shared")
    .add_local_dir(ROOT / "b300", "/workspace/b300")
    .add_local_dir(ROOT / "runners", "/workspace/runners")
)


@app.function(image=image, gpu="B300", timeout=1800)
def execute(mode: str, batch: int, heads: int, dim: int, context: int,
            page_size: int, dtype: str, ncu_set: str, provenance: dict):
    import subprocess
    import sys
    output = Path("/tmp/kernel-lab-output")
    command = [sys.executable, "-m", "runners.local", "b300", mode,
               "--batch", str(batch), "--heads", str(heads), "--dim", str(dim),
               "--context", str(context), "--page-size", str(page_size),
               "--dtype", dtype, "--ncu-set", ncu_set, "--output", str(output)]
    subprocess.run(command, check=True)
    import json
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(provenance)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return {str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*") if path.is_file()}


@app.local_entrypoint()
def main(mode: str = "check", batch: int = 4, heads: int = 8, dim: int = 64,
         context: int = 1024, page_size: int = 16, dtype: str = "float16",
         ncu_set: str = "basic"):
    from datetime import datetime, timezone
    if mode not in {"check", "bench", "profile"}:
        raise ValueError("mode must be check, bench, or profile")
    import subprocess
    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    provenance = {"git_commit": git("rev-parse", "HEAD"),
                  "git_dirty": bool(git("status", "--porcelain"))}
    files = execute.remote(mode, batch, heads, dim, context, page_size, dtype, ncu_set, provenance)
    directory = ROOT / "artifacts" / "b300" / "flash_attention_decode" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for name, data in files.items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(f"Downloaded artifacts to {directory}")
