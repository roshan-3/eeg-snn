"""Classical alpha-power baseline for the eyes-open/closed task.

Operates on the same PhysioNet epochs the SNN consumes, so the SNN-vs-baseline
delta is computed on identical train/test splits, identical preprocessing,
identical labels. Provides the "+X percentage points over classical baseline"
number for the resume.

Two classifiers, both trivial:

- ``best_channel_threshold``: pick the channel with largest train-set
  alpha-power t-statistic between classes, threshold at the midpoint of
  class means. This mirrors ``eyes_detector.py`` on continuous OpenBCI data.
- ``logistic_regression``: 16-dim alpha-power vector per epoch, plain
  logistic regression (closed form via scipy / numpy).

The two results bracket "naive single-feature" and "modern linear model"
classical performance.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.signal import butter, sosfiltfilt

from stack_validation.loaders.physionet import load_eyes_open_closed

ALPHA_BAND: tuple[float, float] = (8.0, 13.0)
DEFAULT_TARGET_FS_HZ: float = 250.0
DEFAULT_EPOCH_S: float = 4.0


def alpha_power_per_epoch(epochs: np.ndarray, fs: float) -> np.ndarray:
    """Return mean-square alpha-band power per epoch per channel.

    Args:
        epochs: ``(N, C, T)`` z-scored EEG epochs from the same preprocessor
            used by the SNN.
        fs: sample rate in Hz.

    Returns:
        ``(N, C)`` array of log10 alpha-band power.
    """
    sos = butter(4, list(ALPHA_BAND), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, epochs, axis=-1)
    power = np.mean(filtered ** 2, axis=-1)  # (N, C)
    return np.log10(power + 1e-12)


def split_indices(n: int, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(n * train_frac)
    return perm[:cut], perm[cut:]


def best_channel_threshold(
    features: np.ndarray, labels: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
) -> dict[str, float]:
    """Pick best channel by training-set t-statistic, threshold at class-mean
    midpoint. Returns test accuracy and the chosen channel."""
    f_train, y_train = features[train_idx], labels[train_idx]
    closed = f_train[y_train == 1]
    open_ = f_train[y_train == 0]
    mu_c, mu_o = closed.mean(0), open_.mean(0)
    var_c, var_o = closed.var(0, ddof=1), open_.var(0, ddof=1)
    n_c, n_o = closed.shape[0], open_.shape[0]
    t = np.abs(mu_c - mu_o) / np.sqrt(var_c / n_c + var_o / n_o + 1e-12)
    ch = int(np.argmax(t))
    threshold = 0.5 * (mu_c[ch] + mu_o[ch])
    # Direction: closed > open expected on alpha; allow either by checking train acc.
    pred_train_hi = (f_train[:, ch] > threshold).astype(int)
    acc_hi = float((pred_train_hi == y_train).mean())
    direction = 1 if acc_hi >= 0.5 else -1
    pred_test = ((features[test_idx, ch] - threshold) * direction > 0).astype(int)
    test_acc = float((pred_test == labels[test_idx]).mean())
    return {"test_acc": test_acc, "channel": ch, "threshold": float(threshold)}


def logistic_regression(
    features: np.ndarray, labels: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
    n_iters: int = 500, lr: float = 0.1, l2: float = 1e-3,
) -> dict[str, float]:
    """Hand-rolled binary logistic regression with L2.

    Avoids the sklearn dependency. Standardizes features using train-set
    statistics only.
    """
    x_train = features[train_idx]
    x_test = features[test_idx]
    y_train = labels[train_idx].astype(np.float64)
    y_test = labels[test_idx].astype(np.float64)

    mu, sigma = x_train.mean(0), x_train.std(0) + 1e-9
    x_train_n = (x_train - mu) / sigma
    x_test_n = (x_test - mu) / sigma

    n, d = x_train_n.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iters):
        z = x_train_n @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = x_train_n.T @ (p - y_train) / n + l2 * w
        grad_b = float((p - y_train).mean())
        w -= lr * grad_w
        b -= lr * grad_b

    p_test = 1.0 / (1.0 + np.exp(-(x_test_n @ w + b)))
    pred = (p_test >= 0.5).astype(int)
    test_acc = float((pred == y_test.astype(int)).mean())
    return {"test_acc": test_acc, "n_features": int(d)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="1-10")
    parser.add_argument("--epoch-s", type=float, default=DEFAULT_EPOCH_S)
    parser.add_argument("--fs", type=float, default=DEFAULT_TARGET_FS_HZ)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from stack_validation.train import parse_subjects
    subjects = parse_subjects(args.subjects)
    print(f"Loading PhysioNet subjects {subjects[0]}..{subjects[-1]} "
          f"({len(subjects)} total) at {args.fs} Hz...")
    batch = load_eyes_open_closed(
        subjects=subjects, epoch_s=args.epoch_s, target_fs=args.fs
    )
    print(f"Loaded {batch.data.shape[0]} epochs, shape (N,C,T) = {batch.data.shape}")

    features = alpha_power_per_epoch(batch.data, args.fs)
    train_idx, test_idx = split_indices(
        batch.data.shape[0], args.train_frac, args.seed
    )

    bct = best_channel_threshold(features, batch.labels, train_idx, test_idx)
    lr = logistic_regression(features, batch.labels, train_idx, test_idx)

    print("\n=== Classical alpha-power baselines ===")
    print(f"best-channel threshold:   test_acc={bct['test_acc']:.3f}  "
          f"(channel {bct['channel']}, threshold={bct['threshold']:.3f})")
    print(f"logistic regression:      test_acc={lr['test_acc']:.3f}  "
          f"({lr['n_features']} features = 1 alpha-power value per channel)")


if __name__ == "__main__":
    main()
