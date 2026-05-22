"""Package wrapper that JIT-compiles the LIF CUDA extension on first import.

Usage:

    from lif_kernel import lif_cuda           # compiles on first call
    spikes, mem_new = lif_cuda.lif_forward(input, mem_prev, beta, threshold)
    grad_v_pre      = lif_cuda.lif_backward(grad_spikes, v_pre, threshold, slope)

If CUDA is not available, importing the extension is skipped and
``lif_cuda`` is ``None``. Downstream code (lif_kernel.lif_function)
checks this and raises a clear error if the kernel is used without CUDA.

The package is named ``lif_kernel`` rather than ``cuda`` to avoid a
shadowing collision with NVIDIA's ``cuda-python`` package, which newer
PyTorch builds import internally as ``cuda.bindings``.
"""

from __future__ import annotations

from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent

lif_cuda = None

if torch.cuda.is_available():
    from torch.utils.cpp_extension import load

    lif_cuda = load(
        name="lif_cuda",
        sources=[
            str(_HERE / "lif_binding.cpp"),
            str(_HERE / "lif_forward.cu"),
            str(_HERE / "lif_backward.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
