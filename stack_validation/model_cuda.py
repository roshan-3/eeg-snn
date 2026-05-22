"""LIF SNN classifier using the custom CUDA LIF kernel.

Architecturally identical to ``stack_validation.model.LIFClassifier``;
only the hidden Leaky cells are swapped for ``LIFCudaCell`` which calls
into the hand-written CUDA kernel under ``lif_kernel/``. The readout stays as
snntorch's leaky integrator (no spikes, no kernel benefit).

Used by the latency report to compare custom CUDA against the snnTorch
reference implementation on identical inputs.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn

from lif_kernel.lif_function import LIFCudaCell

LARGE_THRESHOLD: float = 1e9  # readout never spikes


class LIFClassifierCuda(nn.Module):
    """LIFClassifier variant whose hidden LIFs use the custom CUDA kernel."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_1: int = 64,
        hidden_2: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.lif1 = LIFCudaCell(beta=beta, threshold=threshold,
                                surrogate_slope=surrogate_slope)

        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.lif2 = LIFCudaCell(beta=beta, threshold=threshold,
                                surrogate_slope=surrogate_slope)

        self.fc_out = nn.Linear(hidden_2, num_classes)
        # Readout stays on snntorch; it never spikes (huge threshold) so
        # there is no kernel benefit to porting it.
        self.li_out = snn.Leaky(
            beta=beta,
            threshold=LARGE_THRESHOLD,
            reset_mechanism="none",
        )

    def forward(
        self, spikes: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if spikes.dim() != 3:
            raise ValueError(f"spikes must be (T, B, F), got shape {tuple(spikes.shape)}")

        t_steps, batch_size, _ = spikes.shape
        hidden_1 = self.fc1.out_features
        hidden_2 = self.fc2.out_features

        mem1 = torch.zeros(batch_size, hidden_1, device=spikes.device)
        mem2 = torch.zeros(batch_size, hidden_2, device=spikes.device)
        mem_out = self.li_out.init_leaky()

        readout_history: list[torch.Tensor] = []
        spikes1_history: list[torch.Tensor] = []
        spikes2_history: list[torch.Tensor] = []

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            s1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(s1)
            s2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(s2)
            _, mem_out = self.li_out(cur_out, mem_out)

            spikes1_history.append(s1)
            spikes2_history.append(s2)
            readout_history.append(mem_out)

        logits = torch.stack(readout_history, dim=0).mean(dim=0)
        rate1 = torch.stack(spikes1_history, dim=0).mean()
        rate2 = torch.stack(spikes2_history, dim=0).mean()
        return logits, (rate1, rate2)
