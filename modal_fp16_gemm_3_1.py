import subprocess

import modal


TUTORIAL_DIR = "cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm"
REMOTE_SCRIPT = "/workspace/tutorial_gemm/fp16_gemm_3_1.py"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.13")
    .entrypoint([])
    .pip_install(
        "nvidia-cutlass-dsl[cu13]==4.5.2",
        "torch==2.12.0",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir(
        TUTORIAL_DIR,
        remote_path="/workspace/tutorial_gemm",
        copy=False,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App("fp16-gemm-3-1-b200", image=image)


@app.function(gpu="B200", timeout=1800)
def run_gemm(mnk: str, tolerance: float):
    subprocess.run(
        ["python", REMOTE_SCRIPT, "--mnk", mnk, "--tolerance", str(tolerance)],
        check=True,
    )


@app.local_entrypoint()
def main(mnk: str = "8192,8192,8192", tolerance: float = 1e-1):
    run_gemm.remote(mnk, tolerance)
