"""Autograd wrapper around the custom CUDA LIF kernel.

``LIFCudaFn`` implements one timestep of a subtract-on-reset Leaky
neuron, matching snnTorch's ``Leaky(reset_mechanism="subtract")`` math:

    v_pre   = beta * mem_prev + current
    spike   = (v_pre >= threshold).float()        (detached for the reset)
    mem_new = v_pre - spike * threshold

Gradients (with surrogate ``sg = 1 / (1 + slope * |v_pre - threshold|)^2``):

    d_loss/d_v_pre   = grad_spike * sg + grad_mem_new
    d_loss/d_current = d_loss/d_v_pre
    d_loss/d_mem_prev = d_loss/d_v_pre * beta

``LIFCudaCell`` is an ``nn.Module`` drop-in for ``snntorch.Leaky``: it
keeps the same ``(spike, mem_new)`` return signature and exposes
``init_leaky`` for parity with the snnTorch API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import lif_cuda


class LIFCudaFn(torch.autograd.Function):
    """One timestep of a subtract-on-reset LIF neuron on CUDA."""

    @staticmethod
    def forward(ctx, current, mem_prev, beta, threshold, surrogate_slope):
        if lif_cuda is None:
            raise RuntimeError(
                "lif_cuda extension is not loaded (CUDA not available?). "
                "LIFCudaFn requires a CUDA-enabled PyTorch build."
            )
        current = current.contiguous()
        mem_prev = mem_prev.contiguous()
        spikes, mem_new = lif_cuda.lif_forward(
            current, mem_prev, float(beta), float(threshold)
        )
        ctx.save_for_backward(current, mem_prev)
        ctx.beta = float(beta)
        ctx.threshold = float(threshold)
        ctx.surrogate_slope = float(surrogate_slope)
        return spikes, mem_new

    @staticmethod
    def backward(ctx, grad_spikes, grad_mem_new):
        current, mem_prev = ctx.saved_tensors
        v_pre = ctx.beta * mem_prev + current

        grad_from_spike = lif_cuda.lif_backward(
            grad_spikes.contiguous(),
            v_pre.contiguous(),
            ctx.threshold,
            ctx.surrogate_slope,
        )
        # d_loss/d_v_pre = grad_spike * sg + grad_mem_new
        grad_v_pre = grad_from_spike + grad_mem_new
        # v_pre = beta * mem_prev + current
        grad_current = grad_v_pre
        grad_mem_prev = grad_v_pre * ctx.beta
        # No gradient for the scalar hyperparameters.
        return grad_current, grad_mem_prev, None, None, None


class LIFCudaCell(nn.Module):
    """Drop-in replacement for ``snntorch.Leaky(reset_mechanism="subtract")``.

    Maintains its own forward signature ``cell(current, mem_prev)`` and
    returns ``(spike, mem_new)`` like snnTorch.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)

    @staticmethod
    def init_leaky() -> torch.Tensor:
        """API parity with snnTorch's Leaky.init_leaky()."""
        return torch.zeros(1)  # broadcast-compatible; replaced on first forward

    def forward(
        self, current: torch.Tensor, mem_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mem_prev.shape != current.shape:
            mem_prev = torch.zeros_like(current)
        return LIFCudaFn.apply(
            current, mem_prev, self.beta, self.threshold, self.surrogate_slope
        )
