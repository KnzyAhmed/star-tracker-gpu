"""Syntax/type-check every CUDA kernel with clang, on machines WITHOUT a GPU (e.g. the Mac).
Catches typos before a Colab round trip; says nothing about runtime behaviour (test_gpu.py does).
Skipped if clang is missing or cannot target CUDA."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KERNELS = sorted((ROOT / "tracker" / "kernels").glob("*.cu"))
CMD = ["clang", "-fsyntax-only", "-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=sm_75",
       "-nocudainc", "-nocudalib", "-Wall", "-Werror", "-include", str(ROOT / "tools" / "cuda_shim.h")]


def clang_can_do_cuda():
    if shutil.which("clang") is None:
        return False
    probe = subprocess.run(CMD + ["-"], input='extern "C" __global__ void k(float* x) { x[threadIdx.x] = 1.0f; }',
                           capture_output=True, text=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not clang_can_do_cuda(), reason="clang cannot syntax-check CUDA here")


def test_there_are_kernels():
    assert KERNELS, "no .cu files found"


@pytest.mark.parametrize("cu", KERNELS, ids=lambda p: p.name)
def test_kernel_compiles(cu):
    r = subprocess.run(CMD + [str(cu)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_earth_kernel_compiles_in_float32_too():
    """gpu.render_earth_batch(precision="float32") compiles earth.cu with -DREAL_T=float (Day 12 comparison)."""
    r = subprocess.run(CMD + ["-DREAL_T=float", str(ROOT / "tracker" / "kernels" / "earth.cu")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
