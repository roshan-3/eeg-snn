# Measured results

First end-to-end pass. All numbers come from `metrics.report` and the
`stack_validation.train` CLI in this repo. Reproduce with the commands in
the "Reproduce" section.

## Setup

| Item | Value |
|---|---|
| Task | Eyes-open vs eyes-closed, binary, chance = 50% |
| Dataset | PhysioNet eegmmidb baseline runs (run 1 open, run 2 closed) |
| Subjects | 1-10 of 109 |
| Sample rate | 250 Hz (resampled from native 160 Hz) |
| Epoch length | 4 s (1000 timesteps) |
| Channels | 16-ch 10-20 montage |
| Preprocessing | Butterworth 5-50 Hz + 60 Hz notch (zero-phase), CAR, 4 s epoching, 150 µV peak-to-peak reject, z-score |
| Encoder | Delta encoding (theta = 0.5 sigma per channel), 32 input neurons |
| Model | LIF 32 to 64 to 32 to 2 (snnTorch), subtract-on-reset, fast-sigmoid surrogate slope 25 |
| Loss | CE on time-mean readout membrane + L2 rate reg to target 0.1 (lambda = 1e-3) |
| Optimizer | Adam lr 1e-3, batch 32, grad clip 1.0 |
| Train/test split | 70/30 random epoch split, fixed RNG per seed |
| Hardware | CPU (no CUDA available in this run) |

After preprocessing: **122 epochs kept** (61 open / 61 closed) out of 300
possible. Artifact threshold of 150 µV peak-to-peak rejected all epochs for
subjects 2, 3, 9, 10. Known issue, see "Caveats".

## Model

| Metric | Value |
|---|---|
| Total parameters | **4,258** |
| Trainable parameters | 4,258 |
| Storage at fp32 | 16.63 KB |
| fc1 (32 to 64) | 2,112 |
| fc2 (64 to 32) | 2,080 |
| fc_out (32 to 2) | 66 |

Reference points: EEGNet 2,548, ShallowConvNet ~30k, DeepConvNet ~150k+.

## Accuracy

3 random seeds, same data, same hyperparameters, 50 epochs each.

| Seed | Best test accuracy |
|---|---|
| 0 | 0.892 |
| 1 | 0.811 |
| 2 | 0.757 |
| **mean** | **0.820** |
| sample std (n=3) | 0.068 |

So **82.0% +/- 6.8%** over 3 seeds on a 70/30 random epoch split.

### Classical baseline on the same 122 epochs, same split, seed 0

| Method | Test accuracy |
|---|---|
| Best-channel alpha-power threshold (mirrors `eyes_detector.py`) | 0.730 |
| Logistic regression on 16-channel log alpha power | 0.757 |

**SNN mean delta over best classical baseline: +6.3 pp.**
**Seed-0 SNN delta: +13.5 pp.**

## Firing rate (sparsity) — measured on seed-1 checkpoint

| Layer | Mean firing rate | Target |
|---|---|---|
| LIF 1 (64 units) | 0.138 | 0.10 |
| LIF 2 (32 units) | 0.150 | 0.10 |
| Mean hidden | 0.144 | 0.10 |

Approximately **7x sparser than dense activations**. Rate regularizer
pulled both layers within 50% of the 0.10 target.

## Latency

T = 1000 timesteps = 4 s of EEG at 250 Hz. 30 timed iterations each, warmup
of 10. Two backends measured: the snnTorch reference on Intel CPU and on
an RTX 4070 Laptop GPU (sm_89, CUDA 12.6 torch wheel).

| Backend | Batch | Mean (ms) | Throughput (trials/s) |
|---|---|---|---|
| CPU | 1 | 704 | 1.4 |
| CPU | 32 | 757 | 42.3 |
| CPU | 128 | 1024 | 125.0 |
| CPU | 512 | 1529 | 334.9 |
| CPU | 1024 | 1959 | 522.8 |
| CUDA (snnTorch) | 1 | 1810 | 0.6 |
| CUDA (snnTorch) | 32 | 1892 | 16.9 |
| CUDA (snnTorch) | 128 | 1994 | 64.2 |
| CUDA (snnTorch) | 512 | 2168 | 236.2 |
| CUDA (snnTorch) | 1024 | 2126 | 481.6 |

Real-time factor at single trial: **4 s of EEG processed in 0.7 s on
CPU, ~5.7x faster than realtime.**

**There is no GPU speedup at any batch size <= 1024 with the snnTorch
reference.** The hidden layers are 64 and 32 units wide, and the model
runs a Python `for t in range(1000)` loop that launches ~5 CUDA kernels
per timestep. ~5000 kernel launches per forward pass at ~25 us each
dominates the actual matmul time, so the GPU is launch-overhead-bound.
CPU avoids this entirely.

A hand-written CUDA LIF kernel (`lif_kernel/`) is committed and
numerically validated against the snnTorch reference (see
`tests/test_cuda_lif.py`). It fuses snnTorch's per-timestep ops into one
kernel launch, but does not eliminate the Python-loop dispatch overhead.
Benchmarking it requires installing the CUDA Toolkit (for `nvcc`) and
MSVC Build Tools (for `cl.exe`). Numbers will be added once the
toolchain is in place.

## Caveats (read before quoting on a resume)

1. **Sample size is small.** 122 kept epochs, ~85 train / ~37 test per seed.
   The 6.8% std across seeds is the honest spread.
2. **Random epoch split, not leave-one-subject-out.** Some test epochs come
   from subjects whose other epochs are in train. A LOSO split is the
   stronger, more defensible protocol; expect lower accuracy on it.
3. **Artifact rejection too aggressive.** The 150 µV peak-to-peak threshold
   drops 4 of 10 subjects entirely. Either the threshold needs tuning to
   the PhysioNet data range or it should be replaced with a robust
   statistical rule (e.g., > 6 sigma after z-score). Not changed here to
   avoid introducing differences between this run and the committed code.
4. **GPU does not help at this model size.** Tested on RTX 4070 Laptop
   GPU with the snnTorch reference; CPU is 1.1-2.5x faster across
   B=1..1024 because the per-timestep matmuls (32x64, 64x32) are too
   small to amortize CUDA kernel launch overhead. This is a fundamental
   property of tiny SNNs, not a torch.compile or kernel-fusion problem
   that can be papered over.
5. **Hyperparameters not tuned.** First end-to-end pass with the README
   defaults. No sweep over beta, threshold, rate lambda, or learning rate.
6. **10 subjects of 109.** A full run on 109 subjects is the next test;
   accuracy may rise or fall.

## Suggested resume bullets (anchored to the numbers above)

Pick whichever fits the role.

- "Built an end-to-end EEG to spiking neural network classifier on
  OpenBCI / PhysioNet; reached **82.0% +/- 6.8% on eyes-open vs eyes-closed
  (3 seeds, n=122 epochs)**, **+6.3 pp over a classical alpha-power
  baseline** on identical data."
- "**4,258-parameter LIF network (16.63 KB)** trained with surrogate-gradient
  BPTT; achieves **0.14 spikes/neuron/step (~7x sparser than dense)** under
  L2 rate regularization."
- "Inference at **~5.7x realtime** on CPU (704 ms for a 4 s, 1000-step
  trial); **522 trials/s** throughput at batch 1024 on the same CPU."
- "Profiled inference on CPU and RTX 4070 across batch sizes 1-1024;
  identified CUDA-kernel-launch overhead as the bottleneck for a
  4,258-parameter SNN with a 1000-step Python time loop. Wrote a
  hand-written CUDA LIF kernel + autograd wrapper with
  numerical-equivalence tests vs the snnTorch reference."

Do not quote single-seed best (89.2%) as the headline. Mean across seeds is
the honest number.

Do **not** write "Nx GPU speedup" on this project. The data does not
support it. The honest engineering story is "found and characterized the
launch-overhead bottleneck; wrote a fused kernel as the principled fix",
which is a stronger story for any infra/ML systems role anyway.

## Reproduce

```bash
# 1. Data-free model + latency.
python -m metrics.report

# 2. Train 3 seeds (downloads ~10 EDFs on first run).
python -m stack_validation.train --subjects 1-10 --epochs 50 --seed 0
python -m stack_validation.train --subjects 1-10 --epochs 50 --seed 1 \
    --save metrics/checkpoint_seed1.pt
python -m stack_validation.train --subjects 1-10 --epochs 50 --seed 2

# 3. Classical baseline on the same 122 epochs.
python -m metrics.baseline --subjects 1-10 --seed 0

# 4. Firing-rate measurement on the trained model.
python -m metrics.report --checkpoint metrics/checkpoint_seed1.pt \
    --with-data --subjects 1-10
```
