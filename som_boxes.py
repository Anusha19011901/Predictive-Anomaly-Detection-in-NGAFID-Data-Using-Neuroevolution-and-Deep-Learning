#!/usr/bin/env python3
"""
Self-Organizing Map (SOM) "prototype boxes" for NGAFID windowed data.

What this file does
-------------------
1) Train a 2D SOM on healthy (AFTER-maintenance) windows.
2) For each SOM cell, build per-feature percentile boxes (axis-aligned) from its member windows.
3) Explain new windows by:
   - assigning a prototype_id (BMU cell index)
   - reporting per-feature violations against that cell's box

Key fixes in this version
-------------------------
- Prevents centroid collapse to a single prototype (sigma too wide in minibatch):
  * Default to ONLINE trainer (update per sample).
  * If using epoched/minibatch, either shrink sigma aggressively OR use per-sample updates inside the epoch loop.
- Adds NaN guards around distance computations so argmin won't silently return index 0.
- Verifies the scaler has non-zero variance and is the correct one (time-step feature scaler, not error scaler).

CLI
---
# Train SOM on windows (expects scaler fitted on AFTER windows)
python som_boxes.py train-som \
  --train_npz outputs/after_windows.npz \
  --scaler outputs/scaler.pkl \
  --grid 10x10 --iters 50000 --lr 0.5 --sigma 2.5 --lr-decay 0.05 --sigma-decay 0.05 \
  --out outputs/som/prototypes_som_10x10.npz

# Build percentile boxes per cell from the training windows
python som_boxes.py build-boxes \
  --train_npz outputs/after_windows.npz \
  --som outputs/som/prototypes_som_10x10.npz \
  --pctl 5 95 \
  --out outputs/som/prototypes_som_10x10_p95.npz

# Explain windows from a CSV (sliding window), output prototype ids + violations
python som_boxes.py explain \
  --csv /path/to/before_xxx.csv \
  --scaler outputs/scaler.pkl \
  --som outputs/som/prototypes_som_10x10_p95.npz \
  --win 30 --step 5 \
  --sensors IAS,GndSpd,Pitch,Roll,LatAc,NormAc,TRK \
  --out outputs/explanations/before_xxx_explain.parquet

# Visualize SOM hit heatmap on training windows
python som_boxes.py heatmap \
  --train_npz outputs/after_windows.npz \
  --som outputs/som/prototypes_som_10x10_p95.npz \
  --out outputs/som/hitmap_10x10.png

Notes
-----
- The training NPZ is expected to contain X with shape (n_windows, win*features), already scaled by the same scaler provided.
- The scaler should be the time-step feature scaler (7 features per time step in your pipeline), not an error scaler.
- If you prefer to scale inside this script, pass raw windows and add --fit-scaler to train-som; it will save the scaler.
"""

from __future__ import annotations
import os
import re
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

# -----------------------------
# Utilities
# -----------------------------

def scale_windows_with_timescale(X: np.ndarray, scaler) -> np.ndarray:
    """
    Scale flattened windows with a scaler that may be fit on per-timestep features (e.g., 7 dims).
    If X has shape (n, win*d) and scaler expects (d), we reshape to (n*win, d), transform, then
    reshape back to (n, win*d).
    If X has shape (n, d) and scaler expects (d), we just transform.
    """
    if not hasattr(scaler, "n_features_in_"):
        raise ValueError("Scaler must be a fitted sklearn transformer with n_features_in_.")
    d_s = int(scaler.n_features_in_)
    n, D = X.shape
    if D == d_s:
        Xs = scaler.transform(X)
        return Xs.astype(np.float32)
    if D % d_s == 0:
        # assume flattened windows of length win * d_s
        t = D // d_s
        Xr = X.reshape(n * t, d_s)
        Xsr = scaler.transform(Xr)
        return Xsr.reshape(n, D).astype(np.float32)
    raise ValueError(f"Feature mismatch: X has {D} dims but scaler expects {d_s}. Not divisible.")


def set_seed(seed: int = 1337):
    rng = np.random.default_rng(seed)
    return rng


def parse_grid(s: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)x(\d+)$", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("--grid must look like 10x10")
    return int(m.group(1)), int(m.group(2))


def safe_argmin(d2: np.ndarray, axis: int = 1) -> np.ndarray:
    """Argmin with NaN -> +inf guard to avoid all-zeros due to NaN returning 0."""
    d2 = np.where(np.isnan(d2), np.inf, d2)
    return np.argmin(d2, axis=axis)


def assert_valid_scaler(scaler) -> None:
    if not hasattr(scaler, "scale_"):
        raise ValueError("Provided scaler does not look like a fitted StandardScaler.")
    if not np.all(np.array(scaler.scale_) > 0):
        raise ValueError("Scaler has zero-variance features. Refit on training windows.")


# -----------------------------
# SOM implementation
# -----------------------------
@dataclass
class SOM:
    h: int
    w: int
    d: int  # dimension of input vectors
    rng: np.random.Generator

    def __post_init__(self):
        # Init weights from N(0,1) then scale down
        self.weights = self.rng.normal(0, 1, size=(self.h, self.w, self.d)).astype(np.float32) * 0.01
        # Grid coordinates for Gaussian neighborhood
        ci, cj = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing="ij")
        self.coords = np.stack([ci, cj], axis=-1).astype(np.float32)

    def bmu_index(self, X: np.ndarray) -> np.ndarray:
        """Return BMU flattened indices for X (n, d)."""
        # (n, h, w, d)
        # Compute squared distances: (n, h*w)
        W = self.weights.reshape(self.h * self.w, self.d)  # (hw, d)
        # Use matrix trick: ||x-w||^2 = ||x||^2 + ||w||^2 - 2 x·w
        x2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
        w2 = np.sum(W * W, axis=1, keepdims=True).T  # (1, hw)
        dot = X @ W.T  # (n, hw)
        d2 = x2 + w2 - 2.0 * dot
        bmu_flat = safe_argmin(d2, axis=1)
        return bmu_flat

    def fit(self,
            X: np.ndarray,
            iters: int = 50000,
            lr: float = 0.5,
            sigma: float = 2.5,
            lr_decay: float = 0.05,
            sigma_decay: float = 0.05,
            verbose: bool = True):
        """Online training (per-sample). This is the safest against collapse."""
        n = X.shape[0]
        if n == 0:
            raise ValueError("Empty training set.")
        for t in range(iters):
            x = X[self.rng.integers(0, n)]  # (d,)
            # BMU
            bmu_flat = self.bmu_index(x[None, :])[0]
            ci, cj = divmod(bmu_flat, self.w)
            # Decay
            lr_t = lr * math.exp(-lr_decay * t / max(1, iters))
            sigma_t = max(1e-6, sigma * math.exp(-sigma_decay * t / max(1, iters)))
            # Neighborhood
            d2g = (self.coords[:, :, 0] - ci) ** 2 + (self.coords[:, :, 1] - cj) ** 2
            hood = np.exp(-d2g / (2.0 * sigma_t ** 2)).astype(np.float32)  # (h,w)
            # Update
            self.weights += lr_t * hood[..., None] * (x[None, None, :] - self.weights)
            if verbose and (t % max(10000, iters // 10) == 0):
                print(f"[som.fit] iter={t} lr={lr_t:.4f} sigma={sigma_t:.4f}")

    def fit_epoched(self,
                    X: np.ndarray,
                    epochs: int = 25,
                    lr: float = 0.6,
                    sigma: float = 1.5,
                    lr_decay: float = 0.95,
                    sigma_decay: float = 0.90,
                    batch_size: int = 1024,
                    per_sample_updates: bool = True,
                    verbose: bool = True):
        """
        Mini-batch training. IMPORTANT:
        - Default per_sample_updates=True to avoid global-mean collapse.
        - If per_sample_updates=False, make sure sigma is small and decays fast.
        """
        n = X.shape[0]
        if n == 0:
            raise ValueError("Empty training set.")
        for ep in range(epochs):
            # Shuffle
            idx = self.rng.permutation(n)
            Xs = X[idx]
            # Decays per epoch
            lr_t = lr * (lr_decay ** ep)
            sigma_t = max(1e-6, sigma * (sigma_decay ** ep))
            if verbose:
                print(f"[som.fit_epoched] epoch={ep+1}/{epochs} lr={lr_t:.4f} sigma={sigma_t:.4f}")
            # Iterate in batches
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]  # (b,d)
                # BMUs for the batch
                bmu_flat = self.bmu_index(xb)  # (b,)
                bi, bj = np.divmod(bmu_flat, self.w)
                if per_sample_updates:
                    # Safer: apply updates per sample
                    for b in range(xb.shape[0]):
                        ci, cj = int(bi[b]), int(bj[b])
                        d2g = (self.coords[:, :, 0] - ci) ** 2 + (self.coords[:, :, 1] - cj) ** 2
                        hood = np.exp(-d2g / (2.0 * sigma_t ** 2)).astype(np.float32)
                        self.weights += lr_t * hood[..., None] * (xb[b][None, None, :] - self.weights)
                else:
                    # Riskier but faster: aggregate one target per batch (can collapse if sigma too big)
                    # Compute soft assignment map per cell
                    votes = np.zeros((self.h, self.w), dtype=np.float32)
                    target = np.zeros_like(self.weights)
                    for b in range(xb.shape[0]):
                        ci, cj = int(bi[b]), int(bj[b])
                        votes[ci, cj] += 1.0
                        target[ci, cj] += xb[b]
                    mask = votes > 0
                    target[mask] = target[mask] / votes[mask][..., None]
                    d2g = (self.coords[:, :, 0][..., None, None] - self.coords[:, :, 0]) ** 2 + \
                          (self.coords[:, :, 1][..., None, None] - self.coords[:, :, 1]) ** 2
                    hood = np.exp(-d2g / (2.0 * sigma_t ** 2)).astype(np.float32)
                    # Smooth the target field by neighborhood
                    num = np.sum(hood[..., None] * target[None, None, ...], axis=(2, 3))
                    den = np.sum(hood, axis=(2, 3)) + 1e-8
                    smooth_target = num / den[..., None]
                    self.weights += lr_t * (smooth_target - self.weights)

# -----------------------------
# Prototype boxes
# -----------------------------

def build_boxes(X: np.ndarray, bmu_flat: np.ndarray, h: int, w: int,
                p_lo: float = 5.0, p_hi: float = 95.0) -> Dict[str, np.ndarray]:
    """Build percentile boxes per cell. Returns dict with 'lo', 'hi', 'counts'."""
    d = X.shape[1]
    lo = np.full((h, w, d), np.nan, dtype=np.float32)
    hi = np.full((h, w, d), np.nan, dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)
    for k in range(h * w):
        idx = np.where(bmu_flat == k)[0]
        if idx.size == 0:
            continue
        i, j = divmod(k, w)
        counts[i, j] = idx.size
        sub = X[idx]
        lo[i, j] = np.percentile(sub, p_lo, axis=0)
        hi[i, j] = np.percentile(sub, p_hi, axis=0)
    return {"lo": lo, "hi": hi, "counts": counts}


def violations_against_box(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return signed violation (0 inside; negative if below, positive if above)."""
    below = np.where(x < lo, x - lo, 0.0)
    above = np.where(x > hi, x - hi, 0.0)
    return below + above  # (d,)


# -----------------------------
# Data loading & windowing for CSVs
# -----------------------------

def sliding_windows_from_csv(csv_path: str,
                              sensors: Optional[list[str]] = None,
                              win: int = 30,
                              step: int = 5) -> Tuple[np.ndarray, list[Tuple[int, int]]]:
    """
    Load CSV and create flattened sliding windows over the selected sensors.
    Returns (X_flat, spans) where spans is list of (start_idx, end_idx_exclusive) rows in original CSV.
    """
    df = pd.read_csv(csv_path)
    if sensors is None:
        # Default: keep numeric columns only, drop common timestamp cols
        drop_cols = {"Lcl Date", "Lcl Time", "UTCOfst", "AtvWpt"}
        sensors = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    else:
        for s in sensors:
            if s not in df.columns:
                raise ValueError(f"Sensor '{s}' not found in CSV. Available: {list(df.columns)}")
    V = df[sensors].to_numpy(dtype=np.float32)
    n = V.shape[0]
    spans = []
    X = []
    for start in range(0, max(0, n - win + 1), step):
        end = start + win
        if end > n:
            break
        block = V[start:end]  # (win, d)
        X.append(block.reshape(-1))  # flatten to (win*d,)
        spans.append((start, end))
    if not X:
        return np.empty((0, win * (V.shape[1] if V.ndim == 2 else 0)), dtype=np.float32), []
    X = np.stack(X, axis=0)
    return X, spans


# -----------------------------
# Save/Load helpers for SOM/prototypes
# -----------------------------

def save_som(path: str, som: SOM):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        h=som.h, w=som.w, d=som.d,
                        weights=som.weights)


def load_som(path: str, rng: Optional[np.random.Generator] = None) -> SOM:
    z = np.load(path)
    h, w, d = int(z["h"]), int(z["w"]), int(z["d"])
    som = SOM(h, w, d, rng or set_seed())
    som.weights = z["weights"].astype(np.float32)
    return som


def save_boxes(path: str, boxes: Dict[str, np.ndarray], som: SOM):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path,
                        h=som.h, w=som.w, d=som.d,
                        lo=boxes["lo"], hi=boxes["hi"], counts=boxes["counts"],
                        weights=som.weights)


def load_boxes(path: str, rng: Optional[np.random.Generator] = None) -> Tuple[SOM, Dict[str, np.ndarray]]:
    z = np.load(path)
    h, w, d = int(z["h"]), int(z["w"]), int(z["d"])
    som = SOM(h, w, d, rng or set_seed())
    som.weights = z["weights"].astype(np.float32)
    boxes = {"lo": z["lo"].astype(np.float32),
             "hi": z["hi"].astype(np.float32),
             "counts": z["counts"].astype(np.int32)}
    return som, boxes


# -----------------------------
# Commands
# -----------------------------

def cmd_train_som(args):
    rng = set_seed(args.seed)
    # Load training windows (npz with X)
    data = np.load(args.train_npz)
    if "X" in data:
        X = data["X"].astype(np.float32)
    else:
        # allow raw key name
        X = next(iter(data.values())).astype(np.float32)
    # Scale
    if args.fit_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)
        assert_valid_scaler(scaler)
        joblib.dump(scaler, args.scaler)
    scaler = joblib.load(args.scaler)
    assert_valid_scaler(scaler)
    Xs = scale_windows_with_timescale(X, scaler)

    h, w = parse_grid(args.grid)
    som = SOM(h, w, Xs.shape[1], rng)
    if args.trainer == "online":
        som.fit(Xs, iters=args.iters, lr=args.lr, sigma=args.sigma,
                lr_decay=args.lr_decay, sigma_decay=args.sigma_decay,
                verbose=not args.quiet)
    elif args.trainer == "epoched":
        som.fit_epoched(Xs, epochs=args.epochs, lr=args.lr, sigma=args.sigma,
                        lr_decay=args.lr_decay, sigma_decay=args.sigma_decay,
                        batch_size=args.batch_size, per_sample_updates=not args.batch_average,
                        verbose=not args.quiet)
    else:
        raise ValueError("Unknown trainer")

    # Quick sanity: BMU spread
    bmu = som.bmu_index(Xs)
    uniq = np.unique(bmu).size
    print(f"[train-som] unique BMUs: {uniq}/{h*w}")

    save_som(args.out, som)
    print(f"Saved SOM to {args.out}")


def cmd_build_boxes(args):
    rng = set_seed(args.seed)
    som = load_som(args.som, rng)
    data = np.load(args.train_npz)
    if "X" in data:
        X = data["X"].astype(np.float32)
    else:
        X = next(iter(data.values())).astype(np.float32)
    scaler = joblib.load(args.scaler)
    assert_valid_scaler(scaler)
    Xs = scale_windows_with_timescale(X, scaler)
    bmu = som.bmu_index(Xs)
    boxes = build_boxes(Xs, bmu, som.h, som.w, args.pctl[0], args.pctl[1])
    save_boxes(args.out, boxes, som)
    print(f"Saved boxes to {args.out} (lo/hi percentiles {args.pctl})")


def cmd_explain(args):
    rng = set_seed(args.seed)
    som, boxes = load_boxes(args.som, rng)
    scaler = joblib.load(args.scaler)
    assert_valid_scaler(scaler)
    sensors = args.sensors.split(',') if args.sensors else None
    Xraw, spans = sliding_windows_from_csv(args.csv, sensors=sensors, win=args.win, step=args.step)
    if Xraw.shape[0] == 0:
        raise ValueError("No windows produced. Check --win/--step and CSV length.")
    Xs = scale_windows_with_timescale(Xraw, scaler)
    bmu = som.bmu_index(Xs)
    # Per-window violations
    viol = []
    for n in range(Xs.shape[0]):
        k = int(bmu[n])
        i, j = divmod(k, som.w)
        v = violations_against_box(Xs[n], boxes["lo"][i, j], boxes["hi"][i, j])
        viol.append(v)
    viol = np.stack(viol, axis=0)  # (N, d)

    out = pd.DataFrame({
        "start": [s for s, _ in spans],
        "end": [e for _, e in spans],
        "prototype_id": bmu,
    })
    # Store per-dimension L1 magnitude of violation (sum over dims also useful)
    mag = np.abs(viol)
    out["violation_L1"] = mag.sum(axis=1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote explanations to {args.out}")


def cmd_heatmap(args):
    import matplotlib.pyplot as plt
    rng = set_seed(args.seed)
    som, _ = load_boxes(args.som, rng)
    data = np.load(args.train_npz)
    if "X" in data:
        X = data["X"].astype(np.float32)
    else:
        X = next(iter(data.values())).astype(np.float32)
    scaler = joblib.load(args.scaler)
    assert_valid_scaler(scaler)
    Xs = scale_windows_with_timescale(X, scaler)
    bmu = som.bmu_index(Xs)
    hits = np.zeros((som.h, som.w), dtype=np.int32)
    for k in bmu:
        i, j = divmod(int(k), som.w)
        hits[i, j] += 1
    plt.figure(figsize=(6, 6))
    plt.imshow(hits, origin='upper')
    plt.title('SOM Hit Heatmap')
    plt.colorbar()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches='tight')
    print(f"Saved heatmap to {args.out}")


# -----------------------------
# Argparse
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="SOM prototype boxes for NGAFID")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train-som
    ps = sub.add_parser("train-som", help="Train a SOM on (scaled) windows")
    ps.add_argument("--train_npz", required=True, help="NPZ with X (n, d) windows")
    ps.add_argument("--scaler", required=True, help="Path to fitted StandardScaler.pkl")
    ps.add_argument("--fit-scaler", action="store_true", help="Fit and save scaler on the input X before training")
    ps.add_argument("--grid", type=str, default="10x10", help="Grid size HxW, e.g., 10x10")
    ps.add_argument("--trainer", choices=["online", "epoched"], default="online")
    ps.add_argument("--iters", type=int, default=50000)
    ps.add_argument("--epochs", type=int, default=25)
    ps.add_argument("--batch-size", type=int, default=1024)
    ps.add_argument("--batch-average", action="store_true", help="Use risky batch-averaged updates (faster, may collapse)")
    ps.add_argument("--lr", type=float, default=0.5)
    ps.add_argument("--sigma", type=float, default=2.5)
    ps.add_argument("--lr-decay", type=float, default=0.05)
    ps.add_argument("--sigma-decay", type=float, default=0.05)
    ps.add_argument("--seed", type=int, default=1337)
    ps.add_argument("--quiet", action="store_true")
    ps.add_argument("--out", required=True, help="Output NPZ path for SOM weights")
    ps.set_defaults(func=cmd_train_som)

    # build-boxes
    pb = sub.add_parser("build-boxes", help="Build percentile boxes per SOM cell")
    pb.add_argument("--train_npz", required=True)
    pb.add_argument("--som", required=True)
    pb.add_argument("--scaler", required=True)
    pb.add_argument("--pctl", nargs=2, type=float, default=[5.0, 95.0])
    pb.add_argument("--seed", type=int, default=1337)
    pb.add_argument("--out", required=True)
    pb.set_defaults(func=cmd_build_boxes)

    # explain
    pe = sub.add_parser("explain", help="Assign prototype ids and violations for a CSV")
    pe.add_argument("--csv", required=True)
    pe.add_argument("--scaler", required=True)
    pe.add_argument("--som", required=True, help="Path to SOM+boxes npz (use build-boxes output)")
    pe.add_argument("--win", type=int, default=30)
    pe.add_argument("--step", type=int, default=5)
    pe.add_argument("--sensors", type=str, default=None, help="Comma-separated sensor column names; default picks numeric")
    pe.add_argument("--seed", type=int, default=1337)
    pe.add_argument("--out", required=True)
    pe.set_defaults(func=cmd_explain)

    # heatmap
    ph = sub.add_parser("heatmap", help="Save SOM hit heatmap for training windows")
    ph.add_argument("--train_npz", required=True)
    ph.add_argument("--som", required=True)
    ph.add_argument("--scaler", required=True)
    ph.add_argument("--seed", type=int, default=1337)
    ph.add_argument("--out", required=True)
    ph.set_defaults(func=cmd_heatmap)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
