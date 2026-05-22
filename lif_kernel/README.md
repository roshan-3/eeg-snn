# `lif_kernel` — custom CUDA LIF kernel

Hand-written CUDA implementation of one timestep of a subtract-on-reset
Leaky Integrate-and-Fire (LIF) neuron, with a fast-sigmoid surrogate
gradient for backpropagation through time. Built as a PyTorch C++/CUDA
extension.

The package is named `lif_kernel` (not `cuda`) to avoid shadowing
NVIDIA's `cuda-python` package, which newer PyTorch builds import
internally as `cuda.bindings`.

## Math

Forward (one timestep, per neuron):

```
v_pre   = beta * mem_prev + current
spike   = 1 if v_pre >= threshold else 0   (detached for the reset)
mem_new = v_pre - spike * threshold
```

Backward (surrogate `sg = 1 / (1 + slope * |v_pre - threshold|)^2`):

```
d_loss/d_v_pre   = grad_spike * sg + grad_mem_new
d_loss/d_current = d_loss/d_v_pre
d_loss/d_mem_prev = d_loss/d_v_pre * beta
```

This exactly matches `snntorch.Leaky(reset_mechanism="subtract")` with
`surrogate.fast_sigmoid(slope=slope)`.

## Files

| File | Purpose |
|---|---|
| `lif_forward.cu` | Forward kernel + launch wrapper. |
| `lif_backward.cu` | Backward kernel (surrogate gradient on `v_pre`). |
| `lif_binding.cpp` | pybind11 module exposing `lif_forward`, `lif_backward`. |
| `__init__.py` | JIT-compiles the extension on import via `torch.utils.cpp_extension.load`. Sets `lif_cuda = None` on CPU-only installs. |
| `lif_function.py` | `LIFCudaFn` autograd.Function + `LIFCudaCell` nn.Module (drop-in for `snntorch.Leaky`). |

## Requirements

- NVIDIA GPU with compute capability >= 6.0 (4070 is sm_89, fine).
- CUDA toolkit matching the PyTorch build (e.g. CUDA 12.x for the
  `cu124` wheel).
- PyTorch built with CUDA support (`pip install torch --index-url
  https://download.pytorch.org/whl/cu124` or similar).
- On Windows: a matching MSVC toolchain (Visual Studio Build Tools 2022)
  on PATH so nvcc can find `cl.exe`.

## Build

JIT (default, recommended):

```
python -c "import lif_kernel; print(lif_kernel.lif_cuda)"
```

The first call compiles `~/torch_extensions/lif_cuda/lif_cuda.pyd`
(Windows) or `.so` (Linux). Subsequent imports reuse the cached build.

Ahead-of-time:

```
python setup.py build_ext --inplace
```

## Validation

```
pytest tests/test_cuda_lif.py -v
```

Tests assert that `LIFClassifierCuda` matches the snnTorch reference
`LIFClassifier` to fp32 tolerance on:

1. Single-step forward (spikes and membrane potentials).
2. Multi-step forward over 100 timesteps (logits + per-layer firing rates).
3. Backward weight gradients on a cross-entropy loss.

All tests are auto-skipped when CUDA is not available.

## Benchmark

After validating correctness:

```
python -m metrics.report --compare-backends --latency-iters 30
```

Prints latency for snnTorch CPU, custom CUDA kernel on GPU, and the
CPU-to-CUDA speedup at batch sizes 1 and 32.
