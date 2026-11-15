"""
Run the tests on a Modal GPU, against the working tree as it is (uncommitted changes included).

pytest runs with 4 workers (pytest-xdist) that share the GPU::

    uvx modal run ci/modal_gpu.py                              # every test, 4 workers on an L4
    uvx modal run ci/modal_gpu.py --workers 8 --gpu L40S       # more workers, another GPU
    uvx modal run ci/modal_gpu.py --pytest-args "-k velocity"  # appended to the pytest command
    CUBAM_PYTHON=3.11 uvx modal run ci/modal_gpu.py            # another Python
    uvx modal run ci/modal_gpu.py::build                       # build the image only, on CPU

`modal token new` sets up the credentials once. The function is preemptible
(Modal's default), and waits for an L4 when none is free.

The image is a CUDA base image with cubam's dependencies (the
``rapids-cu12`` and ``anndata`` extras and the ``test`` group), installed
as one layer and rebuilt only when ``pyproject.toml`` changes them. ``src``
and ``tests`` are added when the container starts, so editing them rebuilds
nothing; cubam is imported from ``src`` (it is not installed, so its
version is unknown). The workers share the GPU: each holds a CUDA context
and allocates from RMM's stream-ordered pool, which reserves nothing up
front.

A container that crashes while it starts (before the function runs) is
rescheduled by Modal again and again; ``retries`` does not cover that. So
the function reports that it started, and the run is cancelled if it does
not within ``--startup-deadline`` seconds (30 minutes by default, which
also covers the wait for a free GPU).
"""

import os
import sys
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PYTHON = os.environ.get("CUBAM_PYTHON", "3.13")
# Its CUDA must not be newer than the driver of Modal's GPUs
BASE_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu22.04"
EXTRAS = ("rapids-cu12", "anndata")
GROUPS = ("test",)
SRC = "/root/src"  # the working tree, as added to the container
WORK = "/tmp/cubam"  # its writable copy, where pytest runs
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.pytest_cache"]


def requirements() -> list[str]:
    """The dependencies of cubam, its extras ``EXTRAS`` and its dependency groups ``GROUPS``."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = project["dependency-groups"]

    def group(name):
        for dep in groups[name]:
            yield from group(dep["include-group"]) if isinstance(dep, dict) else [dep]

    return (
        project["project"]["dependencies"]
        + [d for e in EXTRAS for d in project["project"]["optional-dependencies"][e]]
        + [d for g in GROUPS for d in group(g)]
    )


# Modal re-imports this module inside the container, where the script sits at
# /root/modal_gpu.py: the image is defined locally only.
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=PYTHON)
    .uv_pip_install(*requirements())
    .add_local_file(ROOT / "pyproject.toml", f"{SRC}/pyproject.toml")  # pytest's configuration
    .add_local_dir(ROOT / "src", f"{SRC}/src", ignore=IGNORE)
    .add_local_dir(ROOT / "tests", f"{SRC}/tests", ignore=IGNORE)
    if modal.is_local()
    else modal.Image.debian_slim()
)
app = modal.App("cubam-gpu-tests", image=image)


# The GPU is set with with_options in main
@app.function(gpu="L4", timeout=30 * 60)
def tests(workers: int, pytest_args: str, started: modal.Queue | None = None) -> int:
    """Run pytest with ``workers`` processes; its output streams to the local terminal."""
    import shlex
    import shutil
    import subprocess

    if started is not None:
        started.put(True)
    shutil.copytree(SRC, WORK)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    # PYTHONPATH: cubam from src, and not Modal's runtime
    env = os.environ | {"PYTHONPATH": f"{WORK}/src", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [sys.executable, "-m", "pytest", "-ra", "-n", str(workers), *shlex.split(pytest_args)]
    return subprocess.run(cmd, cwd=WORK, env=env, check=False).returncode


@app.function()
def build():
    """Nothing to run: invoking it builds and caches the image without a GPU."""


@app.local_entrypoint()
def main(workers: int = 4, gpu: str = "L4", pytest_args: str = "", startup_deadline: int = 1800):
    import queue

    print(f"pytest -n {workers} on {gpu}, Python {PYTHON}", flush=True)
    with modal.Queue.ephemeral() as started:
        call = tests.with_options(gpu=gpu).spawn(workers, pytest_args, started)
        try:
            started.get(timeout=startup_deadline)
        except queue.Empty:
            call.cancel(terminate_containers=True)
            sys.exit(
                f"No container started within {startup_deadline} s, so the run was cancelled; "
                "the logs above show why (raise --startup-deadline if a GPU was only slow to come up)."
            )
        rc = call.get()
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    print(f"image: {BASE_IMAGE} + Python {PYTHON}, with:", *requirements(), sep="\n  ")
