"""Numerical equivalence: custom CUDA LIF kernel vs snnTorch reference.

All tests are skipped automatically when CUDA is not available, so this
file is safe to import in a CPU-only environment.

What we assert:

1. Single-step forward: spikes and post-reset membrane match exactly
   (same float32 ops; the kernel does no fused multiply/add tricks
   that would diverge from PyTorch's elementwise ops).
2. Multi-step forward over a full SNN: hidden spike trains and readout
   logits match to fp32 tolerance.
3. Backward: weight gradients of the two models on identical loss agree
   to surrogate-gradient tolerance.

Run with: pytest tests/test_cuda_lif.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA not available; kernel tests skipped"
)

if CUDA_AVAILABLE:
    import snntorch as snn
    from snntorch import surrogate

    from lif_kernel.lif_function import LIFCudaCell
    from stack_validation.model import LIFClassifier
    from stack_validation.model_cuda import LIFClassifierCuda


BETA = 0.9
THRESHOLD = 1.0
SLOPE = 25.0
FORWARD_TOL = 1e-6
GRAD_TOL = 1e-4


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def test_single_step_forward_matches_snntorch():
    _seed(0)
    device = torch.device("cuda")

    spike_grad = surrogate.fast_sigmoid(slope=SLOPE)
    ref = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad).to(device)
    ours = LIFCudaCell(beta=BETA, threshold=THRESHOLD, surrogate_slope=SLOPE).to(device)

    current = torch.randn(4, 64, device=device)
    mem_prev = torch.randn(4, 64, device=device)

    s_ref, mem_ref = ref(current, mem_prev)
    s_ours, mem_ours = ours(current, mem_prev)

    assert torch.equal(s_ref, s_ours), "spike outputs differ"
    assert torch.allclose(mem_ref, mem_ours, atol=FORWARD_TOL), \
        f"post-reset membrane differs by {(mem_ref - mem_ours).abs().max().item()}"


def test_multi_step_forward_matches_snntorch():
    _seed(1)
    device = torch.device("cuda")

    in_features, num_classes, t_steps, batch = 32, 2, 100, 8

    ref = LIFClassifier(in_features=in_features, num_classes=num_classes,
                        beta=BETA, threshold=THRESHOLD, surrogate_slope=SLOPE).to(device)
    ours = LIFClassifierCuda(in_features=in_features, num_classes=num_classes,
                             beta=BETA, threshold=THRESHOLD, surrogate_slope=SLOPE).to(device)
    # Tie weights so the two models compute identical math.
    ours.load_state_dict(ref.state_dict(), strict=False)

    spikes = (torch.rand(t_steps, batch, in_features, device=device) < 0.1).float()

    logits_ref, (r1_ref, r2_ref) = ref(spikes)
    logits_ours, (r1_ours, r2_ours) = ours(spikes)

    max_diff = (logits_ref - logits_ours).abs().max().item()
    assert max_diff < FORWARD_TOL, f"logit max diff = {max_diff}"
    assert math.isclose(r1_ref.item(), r1_ours.item(), abs_tol=FORWARD_TOL)
    assert math.isclose(r2_ref.item(), r2_ours.item(), abs_tol=FORWARD_TOL)


def test_backward_grads_match_snntorch():
    _seed(2)
    device = torch.device("cuda")

    in_features, num_classes, t_steps, batch = 16, 2, 50, 4

    ref = LIFClassifier(in_features=in_features, num_classes=num_classes,
                        beta=BETA, threshold=THRESHOLD, surrogate_slope=SLOPE).to(device)
    ours = LIFClassifierCuda(in_features=in_features, num_classes=num_classes,
                             beta=BETA, threshold=THRESHOLD, surrogate_slope=SLOPE).to(device)
    ours.load_state_dict(ref.state_dict(), strict=False)

    spikes = (torch.rand(t_steps, batch, in_features, device=device) < 0.1).float()
    targets = torch.randint(0, num_classes, (batch,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    logits_ref, _ = ref(spikes)
    loss_fn(logits_ref, targets).backward()
    ref_grads = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}

    logits_ours, _ = ours(spikes)
    loss_fn(logits_ours, targets).backward()
    ours_grads = {k: p.grad.detach().clone() for k, p in ours.named_parameters() if p.grad is not None}

    assert ref_grads.keys() == ours_grads.keys(), "parameter sets differ between models"
    for k in ref_grads:
        diff = (ref_grads[k] - ours_grads[k]).abs().max().item()
        assert diff < GRAD_TOL, f"grad {k} differs by {diff}"
