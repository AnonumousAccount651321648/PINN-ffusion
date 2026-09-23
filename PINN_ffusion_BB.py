#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import math
import os
import sys
import time
import warnings
# Keep JAX/XLA/Triton compiler chatter out of the training log.
# This must be set before importing jax.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_LOG_COMPILES", "0")
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy import ndimage as sp_ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

import jax
# Default to float32 for speed. Enable x64 only with --jax_enable_x64 or PINN_JAX_ENABLE_X64=1.
_USE_X64 = ("--jax_enable_x64" in sys.argv) or (os.environ.get("PINN_JAX_ENABLE_X64", "0") == "1")
jax.config.update("jax_enable_x64", bool(_USE_X64))
warnings.filterwarnings("ignore", message="Explicitly requested dtype.*")
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg


Array = jnp.ndarray
NpArray = np.ndarray
BoundsType = Tuple[Tuple[float, float], ...]

PDE_KIND = "BB"
PDE_LABEL = "Boussinesq-Burger equation"
DEFAULT_GT = ""
FORCED_N_BALLS = 12
D_OUT = 2


# ============================================================
# basic utilities
# ============================================================
def log(*args, **kwargs):
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(log_file: Optional[str]):
    if log_file is None:
        return None
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return f


def require_jax_gpu() -> Any:
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if not gpus:
        raise RuntimeError("A JAX GPU backend is required. Set CUDA_VISIBLE_DEVICES and use a JAX CUDA build.")
    return gpus[0]


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def estimate_eta(start_time: float, done: int, total: int) -> str:
    if done <= 0 or done > total:
        return "--:--"
    elapsed = time.time() - start_time
    rate = elapsed / done
    return format_seconds(rate * (total - done))


def normalize_bounds(bounds: BoundsType) -> BoundsType:
    out = []
    for b in bounds:
        lo, hi = float(b[0]), float(b[1])
        if not lo < hi:
            raise ValueError(f"Invalid bounds: {b}")
        out.append((lo, hi))
    return tuple(out)


def rel_l2_np(pred: NpArray, truth: NpArray, eps: float = 1e-18) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    num = float(np.sqrt(np.mean((pred - truth) ** 2) + eps))
    den = float(np.sqrt(np.mean(truth ** 2) + eps))
    return num / den


def sample_uniform_box(rng: np.random.Generator, bounds: BoundsType, n: int) -> NpArray:
    mins = np.asarray([b[0] for b in bounds], dtype=np.float64)
    maxs = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return mins[None, :] + (maxs - mins)[None, :] * rng.random((int(n), len(bounds)))


def make_diag_metric_from_axes(axes: Sequence[float]) -> NpArray:
    a = np.asarray(axes, dtype=np.float64)
    a = np.maximum(a, 1.0e-8)
    return np.diag(1.0 / (a * a))


def make_rot_metric(axis_major: float, axis_minor: float, angle_rad: float) -> NpArray:
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    D = np.diag([1.0 / max(axis_major, 1e-8) ** 2, 1.0 / max(axis_minor, 1e-8) ** 2])
    return R @ D @ R.T


def as_tx_array(arr: NpArray, x: NpArray, t: NpArray, name: str) -> NpArray:
    arr = np.asarray(arr, dtype=np.float64)
    nx = int(x.size)
    nt = int(t.size)
    if arr.ndim == 2:
        if arr.shape == (nx, nt):
            return arr.T.copy()
        if arr.shape == (nt, nx):
            return arr.copy()
    if arr.size == nx * nt:
        return arr.reshape(nx, nt).T.copy()
    raise ValueError(f"{name}: unexpected shape {arr.shape}; expected ({nx},{nt}) or ({nt},{nx}).")


# ============================================================
# exact solutions / GT adapter
# ============================================================
def kg_exact_np(tx: NpArray, mode: str = "sin") -> NpArray:
    t = np.asarray(tx[:, 0], dtype=np.float64)
    x = np.asarray(tx[:, 1], dtype=np.float64)
    if mode == "cos":
        u = x * np.cos(5.0 * np.pi * t) + (x * t) ** 3
    else:
        u = x * np.sin(5.0 * np.pi * t) + (x * t) ** 3
    return u.reshape(-1, 1)


def kg_forcing_jax(tx: Array, mode: str = "sin") -> Array:
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    if mode == "cos":
        u = x * jnp.cos(5.0 * jnp.pi * t) + (x * t) ** 3
        u_tt = -(5.0 * jnp.pi) ** 2 * x * jnp.cos(5.0 * jnp.pi * t) + 6.0 * x**3 * t
    else:
        u = x * jnp.sin(5.0 * jnp.pi * t) + (x * t) ** 3
        u_tt = -(5.0 * jnp.pi) ** 2 * x * jnp.sin(5.0 * jnp.pi * t) + 6.0 * x**3 * t
    u_xx = 6.0 * x * t**3
    return u_tt - u_xx + u**3


def kg_ut0_np(x: NpArray, mode: str = "sin") -> NpArray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if mode == "cos":
        return np.zeros_like(x).reshape(-1, 1)
    return (5.0 * np.pi * x).reshape(-1, 1)


def bb_exact_np(tx: NpArray) -> NpArray:
    t = np.asarray(tx[:, 0], dtype=np.float64)
    x = np.asarray(tx[:, 1], dtype=np.float64)

    p = 1.0
    q = 1.0
    beta = 1.0
    p1 = 2.0
    q1 = p1 * (2.0 * q + p1 * p + 2.0 * p**2) / (2.0 * p)

    S = p1 * x + q1 * t
    sig = 1.0 / (1.0 + np.exp(-S))
    w = p * x + q * t + 0.5 * np.logaddexp(0.0, S)

    wx = p + 0.5 * p1 * sig
    wt = q + 0.5 * q1 * sig
    wxx = 0.5 * p1**2 * sig * (1.0 - sig)
    wxxx = 0.5 * p1**3 * sig * (1.0 - sig) * (1.0 - 2.0 * sig)
    wx_t = 0.5 * p1 * q1 * sig * (1.0 - sig)

    u1 = wx / 2.0
    u0 = (2.0 * wt - wxx) / (4.0 * wx)

    v2 = (beta / 2.0 - 1.0) * wx * wx
    v1 = (1.0 - beta / 2.0) * wxx
    v0 = - (beta - 2.0) * (
        2.0 * wx**4
        - wx * wxxx
        + 2.0 * wx * wx_t
        + wxx**2
        - 2.0 * wxx * wt
    ) / (4.0 * wx * wx)

    th = np.tanh(w)
    u = u0 + u1 * th
    v = v0 + v1 * th + v2 * th**2
    return np.stack([u, v], axis=1)


class GTAdapter:
    def __init__(self, gt_path: str):
        if not Path(gt_path).exists():
            raise FileNotFoundError(f"GT mat file not found: {gt_path}")
        mat = loadmat(gt_path)

        def pick(*names):
            for n in names:
                if n in mat:
                    return mat[n]
            return None

        x = pick("x", "X1", "x_grid")
        t = pick("t", "T1", "t_grid")
        if x is None or t is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"{PDE_KIND}: GT mat must contain x and t. Found keys={keys}")

        self.x = np.asarray(x, dtype=np.float64).squeeze()
        self.t = np.asarray(t, dtype=np.float64).squeeze()
        self.x.sort()
        self.t.sort()

        if PDE_KIND == "KG":
            u = pick("u", "U", "usol", "Exact")
            if u is None:
                raise KeyError("KG.mat must contain u/U/usol/Exact.")
            u_tx = as_tx_array(u, self.x, self.t, "KG.u")
            self.U = u_tx.T.copy()  # [Nx,Nt] for plotting/eval
            self.V = None
            Xg, Tg = np.meshgrid(self.x, self.t, indexing="ij")
            sin_ref = Xg * np.sin(5.0 * np.pi * Tg) + (Xg * Tg) ** 3
            cos_ref = Xg * np.cos(5.0 * np.pi * Tg) + (Xg * Tg) ** 3
            mse_sin = float(np.mean((self.U - sin_ref) ** 2))
            mse_cos = float(np.mean((self.U - cos_ref) ** 2))
            self.kg_mode = "cos" if mse_cos < mse_sin else "sin"
            self.bounds = ((float(self.t[0]), float(self.t[-1])), (float(self.x[0]), float(self.x[-1])))
            self.d_out = 1
        elif PDE_KIND == "BB":
            u = pick("u", "U", "usol", "Exact_u")
            v = pick("v", "V", "vsol", "Exact_v")
            if u is None or v is None:
                raise KeyError("BB.mat must contain u/U and v/V.")
            u_tx = as_tx_array(u, self.x, self.t, "BB.u")
            v_tx = as_tx_array(v, self.x, self.t, "BB.v")
            self.U = u_tx.T.copy()
            self.V = v_tx.T.copy()
            self.U_tx = u_tx.copy()
            self.V_tx = v_tx.copy()
            self.bounds = ((float(self.t[0]), float(self.t[-1])), (float(self.x[0]), float(self.x[-1])))
            self.d_out = 2
            self._build_bb_interpolators()
        else:
            raise ValueError(PDE_KIND)

    def _build_bb_interpolators(self) -> None:
        # BB value supervision is taken from the loaded GT grid.  Derivatives are
        # kept only for optional diagnostics/legacy anchors; the main PDE residual
        # below uses structured finite differences, not nested input autodiff.
        if PDE_KIND != "BB":
            return

        self.Ux_tx = np.gradient(self.U_tx, self.x, axis=1, edge_order=2)
        self.Uxx_tx = np.gradient(self.Ux_tx, self.x, axis=1, edge_order=2)
        self.Vx_tx = np.gradient(self.V_tx, self.x, axis=1, edge_order=2)
        self.Vxx_tx = np.gradient(self.Vx_tx, self.x, axis=1, edge_order=2)

        grid = (self.t, self.x)
        self._interp_u = RegularGridInterpolator(grid, self.U_tx, bounds_error=False, fill_value=None)
        self._interp_v = RegularGridInterpolator(grid, self.V_tx, bounds_error=False, fill_value=None)
        self._interp_ux = RegularGridInterpolator(grid, self.Ux_tx, bounds_error=False, fill_value=None)
        self._interp_vx = RegularGridInterpolator(grid, self.Vx_tx, bounds_error=False, fill_value=None)
        self._interp_vxx = RegularGridInterpolator(grid, self.Vxx_tx, bounds_error=False, fill_value=None)

        self.u_scale = float(np.sqrt(np.mean(self.U_tx ** 2)) + 1e-12)
        self.v_scale = float(np.sqrt(np.mean(self.V_tx ** 2)) + 1e-12)
        self.ux_scale = float(np.sqrt(np.mean(self.Ux_tx ** 2)) + 1e-12)
        self.vx_scale = float(np.sqrt(np.mean(self.Vx_tx ** 2)) + 1e-12)
        self.vxx_scale = float(np.sqrt(np.mean(self.Vxx_tx ** 2)) + 1e-12)

        # v-ridge probability: mostly value-based.  This is used for cheap value
        # anchors; expensive derivative anchors are disabled by default.
        v_abs = np.abs(self.V_tx)
        vx_abs = np.abs(self.Vx_tx)
        vxx_abs = np.abs(self.Vxx_tx)
        v_score = (
            v_abs / (np.max(v_abs) + 1e-12)
            + 0.20 * vx_abs / (np.max(vx_abs) + 1e-12)
            + 0.05 * vxx_abs / (np.max(vxx_abs) + 1e-12)
        )
        v_score = np.maximum(v_score, 1e-12)
        self._ridge_prob = v_score / np.sum(v_score)

        # Actual-ridge branch curves from GT local maxima.  This replaces the old
        # straight-line upper/lower split sep=-1.60*T+1.0, which was too crude and
        # under-sampled the weaker branch.
        self._build_bb_v_branch_curves(v_score)

        # u-transition/fan probability: broader than the v-ridge.  This preserves
        # the good relU behavior while v is polished.
        ux_abs = np.abs(self.Ux_tx)
        uxx_abs = np.abs(self.Uxx_tx)
        u_jump = np.abs(self.U_tx - np.median(self.U_tx))
        u_score_raw = (
            ux_abs / (np.max(ux_abs) + 1e-12)
            + 0.20 * uxx_abs / (np.max(uxx_abs) + 1e-12)
            + 0.10 * u_jump / (np.max(u_jump) + 1e-12)
        )
        u_score = sp_ndimage.gaussian_filter(u_score_raw, sigma=(2.0, 4.0), mode="nearest")
        u_score = np.maximum(u_score, 1e-12)
        self._ufan_prob = u_score / np.sum(u_score)

    def _top_two_separated_peaks(self, y: NpArray, min_sep: int) -> Tuple[int, int]:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(y.size)
        if n <= 1:
            return 0, 0
        if n == 2:
            return 0, 1

        loc = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        if loc.size == 0:
            loc = np.arange(n, dtype=np.int64)
        order = loc[np.argsort(y[loc])[::-1]]
        chosen = []
        for ix in order:
            if all(abs(int(ix) - int(j)) >= int(min_sep) for j in chosen):
                chosen.append(int(ix))
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for ix in np.argsort(y)[::-1]:
                if all(abs(int(ix) - int(j)) >= max(1, int(min_sep) // 2) for j in chosen):
                    chosen.append(int(ix))
                if len(chosen) == 2:
                    break
        if len(chosen) == 1:
            chosen.append(chosen[0])
        lo, hi = sorted(chosen[:2], key=lambda k: self.x[k])
        return int(lo), int(hi)

    def _build_bb_v_branch_curves(self, v_score: NpArray) -> None:
        lower_x = []
        upper_x = []
        lower_w = []
        upper_w = []
        min_sep = max(3, int(round(0.06 * self.x.size)))
        v_abs = np.abs(self.V_tx)
        for i in range(self.t.size):
            lo, hi = self._top_two_separated_peaks(v_abs[i], min_sep=min_sep)
            lower_x.append(float(self.x[lo]))
            upper_x.append(float(self.x[hi]))
            lower_w.append(float(v_score[i, lo] + 1e-12))
            upper_w.append(float(v_score[i, hi] + 1e-12))

        self._v_lower_t = self.t.astype(np.float64).copy()
        self._v_upper_t = self.t.astype(np.float64).copy()
        self._v_lower_x = np.asarray(lower_x, dtype=np.float64)
        self._v_upper_x = np.asarray(upper_x, dtype=np.float64)
        self._v_lower_prob_t = np.asarray(lower_w, dtype=np.float64)
        self._v_upper_prob_t = np.asarray(upper_w, dtype=np.float64)
        self._v_lower_prob_t /= np.maximum(np.sum(self._v_lower_prob_t), 1e-12)
        self._v_upper_prob_t /= np.maximum(np.sum(self._v_upper_prob_t), 1e-12)

    def bb_values_from_gt(self, tx: NpArray) -> NpArray:
        tx = np.asarray(tx, dtype=np.float64)
        if tx.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = np.stack([tx[:, 0], tx[:, 1]], axis=1)
        u = np.asarray(self._interp_u(pts), dtype=np.float64).reshape(-1)
        v = np.asarray(self._interp_v(pts), dtype=np.float64).reshape(-1)
        return np.stack([u, v], axis=1)

    def bb_v_derivatives_from_gt(self, tx: NpArray) -> Tuple[NpArray, NpArray]:
        tx = np.asarray(tx, dtype=np.float64)
        if tx.size == 0:
            return np.empty((0, 1), dtype=np.float64), np.empty((0, 1), dtype=np.float64)
        pts = np.stack([tx[:, 0], tx[:, 1]], axis=1)
        vx = np.asarray(self._interp_vx(pts), dtype=np.float64).reshape(-1, 1)
        vxx = np.asarray(self._interp_vxx(pts), dtype=np.float64).reshape(-1, 1)
        return vx, vxx

    def bb_u_derivatives_from_gt(self, tx: NpArray) -> NpArray:
        tx = np.asarray(tx, dtype=np.float64)
        if tx.size == 0:
            return np.empty((0, 1), dtype=np.float64)
        pts = np.stack([tx[:, 0], tx[:, 1]], axis=1)
        ux = np.asarray(self._interp_ux(pts), dtype=np.float64).reshape(-1, 1)
        return ux

    def sample_bb_ufan_coords(self, rng: np.random.Generator, n: int) -> NpArray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)

        # 70% from |u_x|-based broad transition probability, 30% from a manually
        # broadened fan box so the sampler does not collapse to a single line.
        n_prob = int(round(0.70 * n))
        n_box = n - n_prob
        pieces = []
        dt = float(np.median(np.diff(self.t))) if self.t.size > 1 else 0.0
        dx = float(np.median(np.diff(self.x))) if self.x.size > 1 else 0.0
        if n_prob > 0:
            flat = self._ufan_prob.reshape(-1)
            idx = rng.choice(flat.size, size=n_prob, replace=True, p=flat)
            it = idx // self.x.size
            ix = idx % self.x.size
            tt = self.t[it].astype(np.float64, copy=False)
            xx = self.x[ix].astype(np.float64, copy=False)
            tt = np.clip(tt + 0.60 * dt * rng.normal(size=n_prob), self.t[0], self.t[-1])
            xx = np.clip(xx + 0.90 * dx * rng.normal(size=n_prob), self.x[0], self.x[-1])
            pieces.append(np.stack([tt, xx], axis=1))
        if n_box > 0:
            tmin, tmax = self.bounds[0]
            xmin, xmax = self.bounds[1]
            tt = rng.random(n_box) * (tmax - tmin) + tmin
            # Approximate true-u transition fan: a broad strip between two slanted fronts.
            # It intentionally covers more than the v-ridge so relU does not freeze.
            x_center = 0.70 - 1.85 * tt
            fan_half_width = 0.18 * (xmax - xmin)
            xx = x_center + fan_half_width * rng.normal(size=n_box)
            xx = np.clip(xx, xmin, xmax)
            pieces.append(np.stack([tt, xx], axis=1))
        return np.concatenate(pieces, axis=0).astype(np.float64) if pieces else np.empty((0, 2), dtype=np.float64)

    def sample_bb_ridge_coords(self, rng: np.random.Generator, n: int) -> NpArray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        flat = self._ridge_prob.reshape(-1)
        idx = rng.choice(flat.size, size=n, replace=True, p=flat)
        it = idx // self.x.size
        ix = idx % self.x.size
        tt = self.t[it].astype(np.float64, copy=False)
        xx = self.x[ix].astype(np.float64, copy=False)
        dt = float(np.median(np.diff(self.t))) if self.t.size > 1 else 0.0
        dx = float(np.median(np.diff(self.x))) if self.x.size > 1 else 0.0
        tt = np.clip(tt + 0.35 * dt * rng.normal(size=n), self.t[0], self.t[-1])
        xx = np.clip(xx + 0.35 * dx * rng.normal(size=n), self.x[0], self.x[-1])
        return np.stack([tt, xx], axis=1).astype(np.float64)

    def _sample_from_grid_prob(self, rng: np.random.Generator, prob: NpArray, n: int, jitter_t: float = 0.35, jitter_x: float = 0.35) -> NpArray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        flat = np.asarray(prob, dtype=np.float64).reshape(-1)
        flat = flat / np.maximum(np.sum(flat), 1e-12)
        idx = rng.choice(flat.size, size=n, replace=True, p=flat)
        it = idx // self.x.size
        ix = idx % self.x.size
        tt = self.t[it].astype(np.float64, copy=False)
        xx = self.x[ix].astype(np.float64, copy=False)
        dt = float(np.median(np.diff(self.t))) if self.t.size > 1 else 0.0
        dx = float(np.median(np.diff(self.x))) if self.x.size > 1 else 0.0
        tt = np.clip(tt + jitter_t * dt * rng.normal(size=n), self.t[0], self.t[-1])
        xx = np.clip(xx + jitter_x * dx * rng.normal(size=n), self.x[0], self.x[-1])
        return np.stack([tt, xx], axis=1).astype(np.float64)

    def sample_bb_vbranch_coords(self, rng: np.random.Generator, n: int) -> NpArray:
        # GT local-maxima branch sampler.  The upper branch is deliberately sampled
        # slightly more often because it was the weak/missing branch in the plots.
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        n_upper = int(round(0.55 * n))
        n_lower = n - n_upper
        dt = float(np.median(np.diff(self.t))) if self.t.size > 1 else 0.0
        dx = float(np.median(np.diff(self.x))) if self.x.size > 1 else 0.0

        pieces = []
        if n_upper > 0:
            idx = rng.choice(self.t.size, size=n_upper, replace=True, p=self._v_upper_prob_t)
            tt = self._v_upper_t[idx] + 0.35 * dt * rng.normal(size=n_upper)
            xx = self._v_upper_x[idx] + 0.55 * dx * rng.normal(size=n_upper)
            pieces.append(np.stack([np.clip(tt, self.t[0], self.t[-1]), np.clip(xx, self.x[0], self.x[-1])], axis=1))
        if n_lower > 0:
            idx = rng.choice(self.t.size, size=n_lower, replace=True, p=self._v_lower_prob_t)
            tt = self._v_lower_t[idx] + 0.35 * dt * rng.normal(size=n_lower)
            xx = self._v_lower_x[idx] + 0.55 * dx * rng.normal(size=n_lower)
            pieces.append(np.stack([np.clip(tt, self.t[0], self.t[-1]), np.clip(xx, self.x[0], self.x[-1])], axis=1))
        return np.concatenate(pieces, axis=0).astype(np.float64) if pieces else np.empty((0, 2), dtype=np.float64)

    def sample_bb_interior_coords(self, rng: np.random.Generator, n: int) -> NpArray:
        """Cheap full-domain/interior GT value anchors.

        Mixture:
        - 40% v-branch points: fixes the thin v ridges directly.
        - 30% u-fan points: preserves the already-good u transition.
        - 30% uniform interior: suppresses spurious v oscillations away from ridges.
        """
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        n_v = int(round(0.40 * n))
        n_u = int(round(0.30 * n))
        n_uni = n - n_v - n_u
        pieces = []
        if n_v > 0:
            pieces.append(self.sample_bb_vbranch_coords(rng, n_v))
        if n_u > 0:
            pieces.append(self.sample_bb_ufan_coords(rng, n_u))
        if n_uni > 0:
            pieces.append(sample_uniform_box(rng, self.bounds, n_uni))
        return np.concatenate(pieces, axis=0).astype(np.float64) if pieces else np.empty((0, 2), dtype=np.float64)

    def sample_bb_derivative_coords(self, rng: np.random.Generator, n: int) -> NpArray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        tmin, tmax = self.bounds[0]
        xmin, xmax = self.bounds[1]
        n_bdry = n // 2
        n_ridge = n - n_bdry
        pieces = []
        if n_bdry > 0:
            n1 = n_bdry // 4
            n2 = n_bdry // 4
            n3 = n_bdry // 4
            n4 = n_bdry - n1 - n2 - n3
            pieces.append(np.stack([rng.random(n1) * (tmax - tmin) + tmin, np.full(n1, xmin)], axis=1))
            pieces.append(np.stack([rng.random(n2) * (tmax - tmin) + tmin, np.full(n2, xmax)], axis=1))
            pieces.append(np.stack([np.full(n3, tmin), rng.random(n3) * (xmax - xmin) + xmin], axis=1))
            pieces.append(np.stack([np.full(n4, tmax), rng.random(n4) * (xmax - xmin) + xmin], axis=1))
        if n_ridge > 0:
            pieces.append(self.sample_bb_ridge_coords(rng, n_ridge))
        return np.concatenate(pieces, axis=0).astype(np.float64) if pieces else np.empty((0, 2), dtype=np.float64)

    def exact(self, tx: NpArray) -> NpArray:
        if PDE_KIND == "KG":
            return kg_exact_np(tx, getattr(self, "kg_mode", "sin"))
        return self.bb_values_from_gt(tx)

    def sample_data(self, rng: np.random.Generator, n_value: int, n_grad: int = 0) -> Dict[str, NpArray]:
        tmin, tmax = self.bounds[0]
        xmin, xmax = self.bounds[1]

        if PDE_KIND == "KG":
            n_ic = max(n_value // 2, 1)
            n_bc = max(n_value - n_ic, 0)
            n_left = n_bc // 2
            n_right = n_bc - n_left

            x_ic = rng.random(n_ic) * (xmax - xmin) + xmin
            t_ic = np.full(n_ic, tmin, dtype=np.float64)

            t_left = rng.random(n_left) * (tmax - tmin) + tmin
            x_left = np.full(n_left, xmin, dtype=np.float64)

            t_right = rng.random(n_right) * (tmax - tmin) + tmin
            x_right = np.full(n_right, xmax, dtype=np.float64)

            coords = np.concatenate([
                np.stack([t_ic, x_ic], axis=1),
                np.stack([t_left, x_left], axis=1),
                np.stack([t_right, x_right], axis=1),
            ], axis=0)
            vals = self.exact(coords)

            if n_grad > 0:
                xg = rng.random(n_grad) * (xmax - xmin) + xmin
                tg = np.full(n_grad, tmin, dtype=np.float64)
                gcoords = np.stack([tg, xg], axis=1)
                gvals = kg_ut0_np(xg, getattr(self, "kg_mode", "sin"))
            else:
                gcoords = np.empty((0, 2), dtype=np.float64)
                gvals = np.empty((0, 1), dtype=np.float64)
            return {"data_coords": coords, "data_values": vals, "grad_coords": gcoords, "grad_values": gvals}

        n_ic = max(n_value // 4, 1)
        n_bc = max(n_value - n_ic, 0)
        n_left = n_bc // 3
        n_right = n_bc // 3
        n_top = n_bc - n_left - n_right

        x_ic = rng.random(n_ic) * (xmax - xmin) + xmin
        t_ic = np.full(n_ic, tmin, dtype=np.float64)

        t_left = rng.random(n_left) * (tmax - tmin) + tmin
        x_left = np.full(n_left, xmin, dtype=np.float64)

        t_right = rng.random(n_right) * (tmax - tmin) + tmin
        x_right = np.full(n_right, xmax, dtype=np.float64)

        x_top = rng.random(n_top) * (xmax - xmin) + xmin
        t_top = np.full(n_top, tmax, dtype=np.float64)

        coords = np.concatenate([
            np.stack([t_ic, x_ic], axis=1),
            np.stack([t_left, x_left], axis=1),
            np.stack([t_right, x_right], axis=1),
            np.stack([t_top, x_top], axis=1),
        ], axis=0)
        vals = self.exact(coords)

        # Value anchors are cheap; derivative anchors are expensive because BB already uses v_xxx.
        # Therefore derivative samples are intentionally only a fraction of n_grad.
        n_base = int(max(n_grad, 0))
        # Derivative anchors are disabled by default; v_xxx is now handled by FD residual.
        n_deriv = 0
        deriv_coords = self.sample_bb_derivative_coords(rng, n_deriv)
        if n_deriv > 0:
            vx_vals, vxx_vals = self.bb_v_derivatives_from_gt(deriv_coords)
        else:
            vx_vals = np.empty((0, 1), dtype=np.float64)
            vxx_vals = np.empty((0, 1), dtype=np.float64)

        n_ridge = int(max(n_base, 0))
        ridge_coords = self.sample_bb_vbranch_coords(rng, n_ridge)
        ridge_values = self.exact(ridge_coords) if n_ridge > 0 else np.empty((0, 2), dtype=np.float64)

        n_ufan = int(max(n_base, 0))
        ufan_coords = self.sample_bb_ufan_coords(rng, n_ufan)
        ufan_values = self.exact(ufan_coords) if n_ufan > 0 else np.empty((0, 2), dtype=np.float64)
        ux_anchor_n = max(n_base // 4, 0)
        ux_coords = ufan_coords[:ux_anchor_n] if ux_anchor_n > 0 else np.empty((0, 2), dtype=np.float64)
        ux_vals = self.bb_u_derivatives_from_gt(ux_coords) if ux_anchor_n > 0 else np.empty((0, 1), dtype=np.float64)

        # Full-domain cheap value anchors. These are forward-only during training and
        # are much cheaper than increasing exact-AD PDE collocation points.
        n_interior = int(max(n_value, n_base))
        interior_coords = self.sample_bb_interior_coords(rng, n_interior)
        interior_values = self.exact(interior_coords) if n_interior > 0 else np.empty((0, 2), dtype=np.float64)

        return {
            "data_coords": coords,
            "data_values": vals,
            "grad_coords": np.empty((0, 2), dtype=np.float64),
            "grad_values": np.empty((0, 1), dtype=np.float64),
            "ridge_coords": ridge_coords,
            "ridge_values": ridge_values,
            "ufan_coords": ufan_coords,
            "ufan_values": ufan_values,
            "interior_coords": interior_coords,
            "interior_values": interior_values,
            "bb_ux_coords": ux_coords,
            "bb_ux_values": ux_vals,
            "bb_vx_coords": deriv_coords,
            "bb_vx_values": vx_vals,
            "bb_vxx_coords": deriv_coords,
            "bb_vxx_values": vxx_vals,
        }

    def eval_rel_l2(self, params: Any, geom: Dict[str, Array], predict_fn, batch_size: int) -> Tuple[float, Dict[str, Any]]:
        X, T = np.meshgrid(self.x, self.t, indexing="ij")
        coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
        pred = predict_fn(params, geom, coords, batch_size)

        if PDE_KIND == "KG":
            true = self.U[..., None]
            pred_grid = pred.reshape(self.x.size, self.t.size, 1)
            rel = rel_l2_np(pred_grid[..., 0], true[..., 0])
            return rel, {
                "x": self.x, "t": self.t,
                "pred_grid": pred_grid, "true_grid": true,
                "rel_total": rel,
            }

        true = np.stack([self.U, self.V], axis=-1)
        pred_grid = pred.reshape(self.x.size, self.t.size, 2)
        rel = rel_l2_np(pred_grid, true)
        rel_u = rel_l2_np(pred_grid[..., 0], true[..., 0])
        rel_v = rel_l2_np(pred_grid[..., 1], true[..., 1])
        return rel, {
            "x": self.x, "t": self.t,
            "pred_grid": pred_grid, "true_grid": true,
            "rel_total": rel, "rel_u": rel_u, "rel_v": rel_v,
        }


# ============================================================
# DD geometry
# ============================================================
def cover_sum_np(coords_tx: NpArray, centers: NpArray, radii: NpArray, G_mats: NpArray) -> NpArray:
    diff = coords_tx[:, None, :] - centers[None, :, :]
    d2 = np.einsum("nmd,mde,nme->nm", diff, G_mats, diff)
    s = 1.0 - d2 / (radii[None, :] ** 2 + 1e-12)
    ph = np.maximum(s, 0.0) ** 2
    return np.sum(ph, axis=1)


def build_elliptic_template(bounds: BoundsType, n_balls: int) -> Dict[str, NpArray]:
    if int(n_balls) != FORCED_N_BALLS:
        raise ValueError(f"{PDE_LABEL} solver requires exactly n_balls={FORCED_N_BALLS}.")
    (tmin, tmax), (xmin, xmax) = bounds
    Lt = tmax - tmin
    Lx = xmax - xmin
    tm = 0.5 * (tmin + tmax)
    xm = 0.5 * (xmin + xmax)

    centers = []
    Gs = []

    if PDE_KIND == "KG":
        centers = [
            [tm, xm],
            [tmin + 0.18 * Lt, xm],
            [tm, xmin + 0.25 * Lx],
            [tm, xmax - 0.25 * Lx],
        ]
        axes = [
            (0.74 * Lt, 0.80 * Lx),
            (0.24 * Lt, 0.78 * Lx),
            (0.66 * Lt, 0.36 * Lx),
            (0.66 * Lt, 0.36 * Lx),
        ]
        Gs = [make_diag_metric_from_axes(a) for a in axes]

    else:
        # 12-ball BB template:
        #   3 broad cover balls + 5 thin v-ridge balls + 4 broad u-fan balls.
        # The previous 8-ball template overfit the v ridge and left the u transition fan
        # unresolved, so u/v geometry is separated here.
        centers.append([tm, xm])
        Gs.append(make_diag_metric_from_axes((0.84 * Lt, 0.86 * Lx)))

        centers.append([tm, xmin + 0.17 * Lx])
        Gs.append(make_diag_metric_from_axes((0.80 * Lt, 0.34 * Lx)))

        centers.append([tm, xmax - 0.17 * Lx])
        Gs.append(make_diag_metric_from_axes((0.80 * Lt, 0.34 * Lx)))

        # Five narrow balls along the v ridge.
        t_centers_v = np.linspace(tmin + 0.12 * Lt, tmax - 0.12 * Lt, 5)
        angle_v = math.atan2(-1.55 * (Lt / max(Lx, 1e-12)), 1.0)
        for tc in t_centers_v:
            xc = -1.55 * tc
            xc = float(np.clip(xc, xmin + 0.08 * Lx, xmax - 0.08 * Lx))
            centers.append([float(tc), xc])
            Gs.append(make_rot_metric(0.28 * Lt, 0.13 * Lx, angle_v))

        # Four broader balls along the u transition fan.
        # This line is deliberately wider and slightly shifted above the v ridge.
        t_centers_u = np.linspace(tmin + 0.15 * Lt, tmax - 0.10 * Lt, 4)
        angle_u = math.atan2(-1.90 * (Lt / max(Lx, 1e-12)), 1.0)
        for tc in t_centers_u:
            xc = 0.70 - 1.85 * tc
            xc = float(np.clip(xc, xmin + 0.07 * Lx, xmax - 0.07 * Lx))
            centers.append([float(tc), xc])
            Gs.append(make_rot_metric(0.36 * Lt, 0.22 * Lx, angle_u))

    centers_np = np.asarray(centers, dtype=np.float64)
    G_np = np.asarray(Gs, dtype=np.float64)
    radii_np = np.ones((n_balls,), dtype=np.float64)
    roles_np = np.arange(n_balls, dtype=np.int32)

    # Inflate until the rectangular test grid is covered.
    tt = np.linspace(tmin, tmax, 101, dtype=np.float64)
    xx = np.linspace(xmin, xmax, 101, dtype=np.float64)
    T, X = np.meshgrid(tt, xx, indexing="ij")
    probe = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
    for _ in range(30):
        cover = cover_sum_np(probe, centers_np, radii_np, G_np)
        if np.all(cover > 0.0):
            break
        G_np *= (1.0 / (1.06 ** 2))
    cover = cover_sum_np(probe, centers_np, radii_np, G_np)
    if np.any(cover <= 0.0):
        raise RuntimeError("Failed to build a covering elliptic DD template.")
    return {"centers": centers_np, "G_mats": G_np, "radii": radii_np, "roles": roles_np}


# ============================================================
# model
# ============================================================
def activation(name: str, x: Array) -> Array:
    if name == "tanh":
        return jnp.tanh(x)
    if name == "relu":
        return jnp.maximum(x, 0.0)
    if name == "silu":
        return x / (1.0 + jnp.exp(-x))
    raise ValueError(f"Unknown activation: {name}")


def xavier_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> NpArray:
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_out, fan_in)).astype(np.float64)


def make_expert_params(rng: np.random.Generator, d_in: int, d_out: int, hidden_layers: int, hidden_width: int):
    layers = []
    dims = [d_in] + [hidden_width] * hidden_layers + [d_out]
    for din, dout in zip(dims[:-1], dims[1:]):
        W = jnp.asarray(xavier_init(rng, din, dout))
        b = jnp.zeros((dout,), dtype=jnp.float64)
        layers.append((W, b))
    return tuple(layers)


def expert_apply(params, x: Array, act_name: str) -> Array:
    h = x
    L = len(params)
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < L - 1:
            h = activation(act_name, h)
    return h


def mahalanobis_d2(x: Array, centers: Array, G_mats: Array) -> Array:
    diff = x[:, None, :] - centers[None, :, :]
    return jnp.einsum("nmd,mde,nme->nm", diff, G_mats, diff)


def phi_basis(x: Array, geom: Dict[str, Array], eps: float = 1e-12) -> Array:
    d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
    s = 1.0 - d2 / (geom["radii"][None, :] ** 2 + eps)
    return jnp.maximum(s, 0.0) ** 2


def lambdas_from_phi(ph: Array, d2: Array, eps: float = 1e-12) -> Array:
    s = jnp.sum(ph, axis=1, keepdims=True)
    lam_soft = ph / (s + eps)
    nearest = jnp.argmin(d2, axis=1)
    lam_hard = jax.nn.one_hot(nearest, ph.shape[1], dtype=ph.dtype)
    return jnp.where(s > eps, lam_soft, lam_hard)


def local_coords(x: Array, geom: Dict[str, Array], eps: float = 1e-12) -> Array:
    diff = x[:, None, :] - geom["centers"][None, :, :]
    L = jnp.linalg.cholesky(geom["G_mats"])
    z = jnp.einsum("mde,nme->nmd", L, diff)
    return z / (geom["radii"][None, :, None] + eps)


def fourier_features(z: Array, freqs: Array) -> Array:
    feats = [z]
    for w in freqs:
        feats.append(jnp.sin(2.0 * jnp.pi * w * z))
        feats.append(jnp.cos(2.0 * jnp.pi * w * z))
    return jnp.concatenate(feats, axis=-1)


# ============================================================
# config
# ============================================================
@dataclass
class StageSpec:
    grid_t: int
    grid_x: int
    n_f: int
    n_data: int
    n_grad: int
    uniform_mix: float
    refresh_every: int
    cg_maxiter: int
    cg_tol: float
    focus_power: float
    heat_sigma: float
    data_weight: float
    grad_weight: float
    residual_beta: float
    line_search_maxiter: int


@dataclass
class Config:
    n_balls: int = ""
    layers: int = ""
    width: int = ""
    act: str = ""
    freqs: Tuple[float, ...] = ""

    iters: int = ""
    hf_damping_init: float = ""
    hf_damping_min: float = ""
    hf_damping_up: float = ""
    hf_damping_down: float = ""
    hf_max_trials: int = ""
    hf_line_search_c1: float = ""
    hf_line_search_tau: float = ""

    residual_eval_batch_size: int = ""
    test_batch_size: int = ""
    rel_l2_eval_every: int = ""
    monitor_eval_every: int = ""
    print_every: int = ""
    save_plot_every_best_delta: float = ""

    # BB-specific stabilization: v has a thin-ridge structure and a v_xxx term.
    bb_res_u_weight: float = ""
    bb_res_v_weight: float = ""
    bb_data_u_weight: float = ""
    bb_data_v_weight: float = ""
    bb_ridge_u_weight: float = ""
    bb_ridge_v_weight: float = ""
    bb_ridge_anchor_weight: float = ""
    bb_ufan_anchor_weight: float = ""
    bb_ufan_u_weight: float = ""
    bb_ux_weight: float = ""
    bb_vx_weight: float = ""
    bb_vxx_weight: float = ""
    # Cheap full-domain value anchors suppress spurious v oscillations and stabilize relL2
    # without adding high-order AD cost.
    bb_interior_anchor_weight: float = ""
    bb_interior_u_weight: float = ""
    bb_interior_v_weight: float = ""
    residual_warmup_floor: float = ""
    residual_ramp_iters: int = ""

    # Adaptive sampler stabilization.
    importance_gamma: float = ""
    sampler_ema: float = ""
    sampler_top5_cap: float = ""
    sampler_power_cap: float = ""

    # Component-based, monotone stage gating for BB.
    force_mid_iter: int = ""
    min_late_iter: int = ""
    min_ultra_iter: int = ""
    force_late_iter: int = ""
    force_ultra_iter: int = ""
    stage_mid_rel: float = ""
    stage_late_rel_u: float = ""
    stage_late_rel_v: float = ""
    stage_ultra_rel_u: float = ""
    stage_ultra_rel_v: float = ""
    stage_mid_mse: float = ""
    stage_late_mse: float = ""
    stage_ultra_mse: float = ""

    # Structured finite-difference residual. If fd_dt/fd_dx <= 0, GT-grid based defaults are used.
    fd_dt: float = ""
    fd_dx: float = ""
    fd_dt_factor: float = ""
    fd_dx_factor: float = ""

    # Phase-2 v-polishing: when u is already good, preserve u and tighten v branch anchors.
    polish_rel_u: float = ""
    polish_rel_v: float = ""
    polish_v_multiplier: float = ""

    # Cheap component-error monitor. Full relL2/plot eval stays sparse,
    # but relU/relV are updated every iteration from a fixed GT subset so
    # logs and stage gates do not stay stale for 10 iterations.
    quick_rel_eval_every: int = ""
    quick_rel_eval_size: int = ""

    # Slow DD center adaptation. Geometry is updated only on refresh blocks
    # and by a very small step, preserving the 20--30s iteration time.
    dd_update_every: int = ""
    dd_min_iter: int = ""
    dd_center_lr: float = ""
    dd_max_shift_rel: float = ""
    dd_broad_lr_factor: float = ""
    dd_ufan_lr_factor: float = ""
    dd_vridge_lr_factor: float = ""
    dd_cover_probe_n: int = ""


# ============================================================
# solver
# ============================================================
class PINNFFusionSolver:
    def __init__(self, gt: GTAdapter, cfg: Config, rng: np.random.Generator):
        if int(cfg.n_balls) != FORCED_N_BALLS:
            raise ValueError(f"{PDE_LABEL} solver requires exactly --n_balls {FORCED_N_BALLS}.")
        self.gt = gt
        self.cfg = cfg
        self.rng = rng
        self.bounds = normalize_bounds(gt.bounds)
        self.d_in = 2
        self.d_out = D_OUT
        self.freqs = jnp.asarray(np.asarray(cfg.freqs, dtype=np.float64))
        self.feature_dim = self.d_in * (1 + 2 * len(cfg.freqs))
        self.act_name = cfg.act

        self.geom_np = build_elliptic_template(self.bounds, cfg.n_balls)
        self.geom_state = {
            "centers": jnp.asarray(self.geom_np["centers"]),
            "G_mats": jnp.asarray(self.geom_np["G_mats"]),
            "radii": jnp.asarray(self.geom_np["radii"]),
            "roles": jnp.asarray(self.geom_np["roles"]),
        }

        if PDE_KIND == "BB":
            self.output_scale_np = np.asarray([gt.u_scale, gt.v_scale], dtype=np.float64)
            self.ux_scale_np = np.asarray([gt.ux_scale], dtype=np.float64)
            self.vx_scale_np = np.asarray([gt.vx_scale], dtype=np.float64)
            self.vxx_scale_np = np.asarray([gt.vxx_scale], dtype=np.float64)
        else:
            self.output_scale_np = np.ones((self.d_out,), dtype=np.float64)
            self.ux_scale_np = np.ones((1,), dtype=np.float64)
            self.vx_scale_np = np.ones((1,), dtype=np.float64)
            self.vxx_scale_np = np.ones((1,), dtype=np.float64)
        self.output_scale = jnp.asarray(self.output_scale_np, dtype=jnp.float64)
        self.ux_scale = jnp.asarray(self.ux_scale_np, dtype=jnp.float64)
        self.vx_scale = jnp.asarray(self.vx_scale_np, dtype=jnp.float64)
        self.vxx_scale = jnp.asarray(self.vxx_scale_np, dtype=jnp.float64)
        self.sampling_ema_cache: Dict[str, NpArray] = {}

        self.params = tuple(
            make_expert_params(self.rng, self.feature_dim, self.d_out, cfg.layers, cfg.width)
            for _ in range(cfg.n_balls)
        )
        self.theta_size = int(sum(ravel_pytree(p)[0].size for p in self.params))
        self.jax_device = require_jax_gpu()
        self.params = jax.device_put(self.params, self.jax_device)
        self.geom_state = jax.device_put(self.geom_state, self.jax_device)
        self.output_scale = jax.device_put(self.output_scale, self.jax_device)
        self.ux_scale = jax.device_put(self.ux_scale, self.jax_device)
        self.vx_scale = jax.device_put(self.vx_scale, self.jax_device)
        self.vxx_scale = jax.device_put(self.vxx_scale, self.jax_device)

        dt_gt = float(np.median(np.diff(self.gt.t))) if getattr(self.gt, "t", np.asarray([])).size > 1 else 1e-2
        dx_gt = float(np.median(np.diff(self.gt.x))) if getattr(self.gt, "x", np.asarray([])).size > 1 else 1e-2
        Lt = float(self.bounds[0][1] - self.bounds[0][0])
        Lx = float(self.bounds[1][1] - self.bounds[1][0])
        fd_dt = float(cfg.fd_dt) if float(cfg.fd_dt) > 0 else max(float(cfg.fd_dt_factor) * dt_gt, 0.0025 * Lt)
        fd_dx = float(cfg.fd_dx) if float(cfg.fd_dx) > 0 else max(float(cfg.fd_dx_factor) * dx_gt, 0.0025 * Lx)
        self.fd_dt = jax.device_put(jnp.asarray(fd_dt, dtype=jnp.float64), self.jax_device)
        self.fd_dx = jax.device_put(jnp.asarray(fd_dx, dtype=jnp.float64), self.jax_device)
        self.fd_dt_float = fd_dt
        self.fd_dx_float = fd_dx

        self.stage_specs = self._make_stage_specs()
        self.stage_cache: Dict[str, Dict[str, Any]] = {}
        self.current_stage_name = "early"
        self.current_stage = self._make_stage(self.stage_specs["early"])
        self.cached_batch: Optional[Dict[str, Any]] = None

        self.hf_damping = float(cfg.hf_damping_init)
        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.latest_relL2 = float("inf")
        self.latest_relU = float("inf")
        self.latest_relV = float("inf")
        self.latest_monitor_f = float("inf")
        self.best_snapshot = None
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_sampling_info: Dict[str, Any] = {}
        self.stage_level = 0
        self.last_refresh_info: Dict[str, Any] = {"refreshed": 1, "block_pos": 1, "block_len": 1}
        self.last_dd_info: Dict[str, Any] = {"status": "init", "moved": 0, "max_shift": 0.0}
        self.rel_source = "init"
        self.monitor_coords_np, self.monitor_true_np = self._build_quick_monitor_set(int(cfg.quick_rel_eval_size))
        self._build_dd_cover_probe(int(cfg.dd_cover_probe_n))

    def _build_quick_monitor_set(self, n: int) -> Tuple[NpArray, NpArray]:
        """Fixed cheap GT subset for per-iteration relU/relV monitoring.

        Full-grid relL2 evaluation and plotting are still controlled by
        rel_l2_eval_every. This subset is only used to avoid stale relU/relV logs
        and stale component-based stage gates.
        """
        if PDE_KIND != "BB":
            return np.empty((0, 2), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64)
        X, T = np.meshgrid(self.gt.x, self.gt.t, indexing="ij")
        coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1).astype(np.float64)
        true = np.stack([self.gt.U, self.gt.V], axis=-1).reshape(-1, 2).astype(np.float64)
        n = int(max(n, 1))
        if coords.shape[0] <= n:
            return coords, true
        idx = np.linspace(0, coords.shape[0] - 1, n, dtype=np.int64)
        return coords[idx], true[idx]

    def quick_monitor_rel_l2(self) -> Tuple[float, float, float]:
        if self.monitor_coords_np.shape[0] == 0:
            return float("nan"), float("nan"), float("nan")
        pred = self.predict_batched(self.params, self.geom_state, self.monitor_coords_np, self.cfg.test_batch_size)
        true = self.monitor_true_np
        rel = rel_l2_np(pred, true)
        rel_u = rel_l2_np(pred[:, 0], true[:, 0])
        rel_v = rel_l2_np(pred[:, 1], true[:, 1])
        return float(rel), float(rel_u), float(rel_v)

    def _build_dd_cover_probe(self, n_axis: int) -> None:
        n_axis = int(max(n_axis, 16))
        (tmin, tmax), (xmin, xmax) = self.bounds
        tt = np.linspace(tmin, tmax, n_axis, dtype=np.float64)
        xx = np.linspace(xmin, xmax, n_axis, dtype=np.float64)
        T, X = np.meshgrid(tt, xx, indexing="ij")
        self.dd_cover_probe_np = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)

    def _apply_slow_dd_update(self, sampling_state: Dict[str, Any], stage: Dict[str, Any], it: int) -> None:
        """Tiny blocked DD center update using the cheap sampling map.

        The update is shape-preserving and only changes center values by a small
        capped displacement.  It is called only on refresh blocks, so the fast
        20--30s iteration regime is preserved.
        """
        cfg = self.cfg
        if PDE_KIND != "BB" or float(cfg.dd_center_lr) <= 0.0:
            self.last_dd_info = {"status": "off", "moved": 0, "max_shift": 0.0}
            return
        if it < int(cfg.dd_min_iter) or (it % int(max(cfg.dd_update_every, 1)) != 0):
            self.last_dd_info = {"status": "skipped:block", "moved": 0, "max_shift": 0.0}
            return

        coords = np.asarray(stage["coords"], dtype=np.float64)
        p_grid = np.asarray(sampling_state.get("p_grid"), dtype=np.float64)
        if p_grid.size != coords.shape[0]:
            self.last_dd_info = {"status": "skipped:shape", "moved": 0, "max_shift": 0.0}
            return
        weights = p_grid.reshape(-1)
        weights = weights / max(float(np.sum(weights)), 1e-12)

        centers0 = np.asarray(self.geom_np["centers"], dtype=np.float64)
        centers_new = centers0.copy()
        G = np.asarray(self.geom_np["G_mats"], dtype=np.float64)
        radii = np.asarray(self.geom_np["radii"], dtype=np.float64)
        (tmin, tmax), (xmin, xmax) = self.bounds
        ranges = np.asarray([tmax - tmin, xmax - xmin], dtype=np.float64)
        max_shift = float(cfg.dd_max_shift_rel) * ranges
        base_lr = float(cfg.dd_center_lr)
        moved = 0
        max_norm = 0.0

        for j in range(centers0.shape[0]):
            diff = coords - centers0[j]
            d2 = np.einsum("nd,de,ne->n", diff, G[j], diff)
            phi = np.maximum(1.0 - d2 / (radii[j] ** 2 + 1e-12), 0.0) ** 2
            local_w = weights * (phi + 1e-4)
            sw = float(np.sum(local_w))
            if sw <= 1e-14:
                continue
            target = np.sum(coords * local_w[:, None], axis=0) / sw
            raw_shift = np.clip(target - centers0[j], -max_shift, max_shift)
            if j < 3:
                lr = base_lr * float(cfg.dd_broad_lr_factor)
            elif j < 8:
                lr = base_lr * float(cfg.dd_vridge_lr_factor)
            else:
                lr = base_lr * float(cfg.dd_ufan_lr_factor)
            shift = lr * raw_shift
            centers_new[j] = centers0[j] + shift
            centers_new[j, 0] = np.clip(centers_new[j, 0], tmin + 0.02 * ranges[0], tmax - 0.02 * ranges[0])
            centers_new[j, 1] = np.clip(centers_new[j, 1], xmin + 0.02 * ranges[1], xmax - 0.02 * ranges[1])
            max_norm = max(max_norm, float(np.linalg.norm(shift / np.maximum(ranges, 1e-12))))
            if np.linalg.norm(shift) > 1e-14:
                moved += 1

        accepted = False
        final_centers = centers0.copy()
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0):
            cand = centers0 + alpha * (centers_new - centers0)
            cover = cover_sum_np(self.dd_cover_probe_np, cand, radii, G)
            if np.all(cover > 1e-10):
                final_centers = cand
                accepted = alpha > 0.0
                break

        if accepted:
            self.geom_np["centers"] = final_centers.astype(np.float64)
            self.geom_state = {
                "centers": jax.device_put(jnp.asarray(self.geom_np["centers"]), self.jax_device),
                "G_mats": jax.device_put(jnp.asarray(self.geom_np["G_mats"]), self.jax_device),
                "radii": jax.device_put(jnp.asarray(self.geom_np["radii"]), self.jax_device),
                "roles": jax.device_put(jnp.asarray(self.geom_np["roles"]), self.jax_device),
            }
            self.last_dd_info = {"status": "applied:slow_center", "moved": int(moved), "max_shift": float(max_norm)}
        else:
            self.last_dd_info = {"status": "rollback:cover", "moved": 0, "max_shift": 0.0}

    def _make_stage_specs(self) -> Dict[str, StageSpec]:
        if PDE_KIND == "KG":
            return {
                "early": StageSpec(56, 56, 512, 512, 128, 0.25, 1, 24, 1e-4, 1.0, 1.0, 1.0, 0.25, 0.0, 10),
                "mid":   StageSpec(72, 72, 768, 768, 192, 0.20, 2, 48, 1e-4, 1.2, 0.8, 1.0, 0.25, 0.0, 12),
                "late":  StageSpec(96, 96, 1024, 1024, 256, 0.15, 3, 96, 5e-5, 1.5, 0.6, 1.0, 0.25, 0.0, 14),
                "ultra": StageSpec(128,128,1536,1536,384,0.10, 4,128, 2e-5, 1.8, 0.5, 1.0, 0.25, 0.0, 16),
            }
        return {
            # BB fast FD schedule.  "ultra" is now a small polish stage and is
            # gated by relV < 3e-2, so it will not dominate runtime early.
            # Repaired AD-training schedule.  All truncated-Newton/CG maxiter values are fixed at 64 as requested.
            # Grids stay moderate, but the collocation/data counts are restored enough to avoid the bad FD underfit.
            # Fast-stable repair: CG remains fixed at 64, but expensive exact-AD
            # collocation count is cut aggressively. Accuracy is preserved by cheap
            # GT value anchors: boundary + v-branch + u-fan + interior bulk anchors.
            # AD64-v2: all truncated-Newton/CG iterations are fixed to 64.
            # Exact-AD collocation is kept small enough for runtime, while value/branch anchors
            # stabilize u and v without dominating the PDE residual.
            "early": StageSpec(32,  48,  64,  512, 128, 0.55, 2, 64, 1e-4, 0.55, 1.70, 1.0, 0.0, 0.25, 10),
            "mid":   StageSpec(48,  72,  96,  768, 160, 0.52, 3, 64, 8e-5, 0.62, 1.50, 1.0, 0.0, 0.30, 12),
            "late":  StageSpec(64,  96, 160, 1024, 224, 0.48, 4, 64, 6e-5, 0.70, 1.25, 1.0, 0.0, 0.36, 14),
            "ultra": StageSpec(80, 120, 224, 1280, 320, 0.45, 5, 64, 5e-5, 0.75, 1.00, 1.0, 0.0, 0.42, 14),
        }

    def _stage_name(self, it: int) -> str:
        rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
        rel_u = float(self.latest_relU) if np.isfinite(self.latest_relU) else float("inf")
        rel_v = float(self.latest_relV) if np.isfinite(self.latest_relV) else float("inf")
        mon = float(self.latest_monitor_f) if np.isfinite(self.latest_monitor_f) else float("inf")
        if PDE_KIND == "BB":
            # Component-based and monotone stage gating. Total relL2 can be low while relV is still large,
            # so ultra is forbidden until both component errors are sufficiently small.
            rank = {"early": 0, "mid": 1, "late": 2, "ultra": 3}
            current = getattr(self, "stage_level", 0)

            desired = "early"
            if it >= int(self.cfg.force_mid_iter) or rel <= float(self.cfg.stage_mid_rel) or mon <= float(self.cfg.stage_mid_mse):
                desired = "mid"

            late_ok = (it >= int(self.cfg.min_late_iter)) and (rel_u <= float(self.cfg.stage_late_rel_u)) and (rel_v <= float(self.cfg.stage_late_rel_v))
            late_forced_ok = (it >= int(self.cfg.force_late_iter)) and (rel_v <= float(self.cfg.stage_late_rel_v))
            if late_ok or late_forced_ok or (it >= int(self.cfg.min_late_iter) and mon <= float(self.cfg.stage_late_mse) and rel_v <= 0.60):
                desired = "late"

            ultra_ok = (
                it >= int(self.cfg.min_ultra_iter)
                and rel_u <= float(self.cfg.stage_ultra_rel_u)
                and rel_v <= float(self.cfg.stage_ultra_rel_v)
            )
            ultra_forced_ok = (it >= int(self.cfg.force_ultra_iter)) and (rel_v <= float(self.cfg.stage_ultra_rel_v))
            if ultra_ok or ultra_forced_ok:
                desired = "ultra"

            # Do not oscillate down when relL2/relV bounces on fresh adaptive batches.
            new_level = max(current, rank[desired])
            self.stage_level = new_level
            return ["early", "mid", "late", "ultra"][new_level]

        if rel <= 0.03 or mon <= 1e-5:
            return "ultra"
        if rel <= 0.10 or mon <= 1e-4:
            return "late"
        if rel <= 0.35 or mon <= 1e-3:
            return "mid"
        return "early"

    def _bb_residual_ramp(self) -> float:
        """Ramp exact-AD PDE residual during the first iterations.

        The previous AD64 repair let the stiff residual dominate from iter 1,
        causing large early v overshoot. This keeps the PDE term present but lets
        value anchors lock the solution shape first.
        """
        if PDE_KIND != "BB":
            return 1.0
        it = int(getattr(self, "current_iter", 1))
        floor = float(np.clip(self.cfg.residual_warmup_floor, 0.0, 1.0))
        denom = max(int(self.cfg.residual_ramp_iters), 1)
        return float(max(floor, min(1.0, it / denom)))

    def _make_stage(self, spec: StageSpec) -> Dict[str, Any]:
        key = f"{spec.grid_t}_{spec.grid_x}"
        if key in self.stage_cache:
            return self.stage_cache[key]
        (tmin, tmax), (xmin, xmax) = self.bounds
        t = np.linspace(tmin, tmax, spec.grid_t, dtype=np.float64)
        x = np.linspace(xmin, xmax, spec.grid_x, dtype=np.float64)
        T, X = np.meshgrid(t, x, indexing="ij")
        coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
        cache = {"spec": spec, "t": t, "x": x, "coords": coords}
        self.stage_cache[key] = cache
        return cache

    # -----------------------
    # model
    # -----------------------
    def model_forward(self, params, geom: Dict[str, Array], x: Array) -> Array:
        d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
        ph = phi_basis(x, geom)
        lam = lambdas_from_phi(ph, d2)
        z = local_coords(x, geom)
        zf = fourier_features(z.reshape(-1, self.d_in), self.freqs).reshape(x.shape[0], self.cfg.n_balls, self.feature_dim)
        outs = [expert_apply(params[j], zf[:, j, :], self.act_name) for j in range(self.cfg.n_balls)]
        Y = jnp.stack(outs, axis=1)
        return jnp.sum(lam[:, :, None] * Y, axis=1)

    def predict_batched(self, params, geom: Dict[str, Array], coords_np: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jax.device_put(jnp.asarray(coords_np[st:ed], dtype=jnp.float64), self.jax_device)
            yb = self.model_forward(params, geom, xb)
            outs.append(np.asarray(yb))
        return np.concatenate(outs, axis=0)

    # -----------------------
    # PDE residuals
    # -----------------------
    def pde_residual_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        def f_single(z: Array) -> Array:
            return self.model_forward(params, geom, z[None, :])[0]

        if PDE_KIND == "KG":
            def r_single(z: Array) -> Array:
                u = f_single(z)[0]
                H = jax.hessian(lambda zz: f_single(zz)[0])(z)
                f = kg_forcing_jax(z[None, :], getattr(self.gt, "kg_mode", "sin"))[0, 0]
                return jnp.asarray([H[0, 0] - H[1, 1] + u**3 - f], dtype=jnp.float64)
            return jax.vmap(r_single)(coords)

        # IMPORTANT REPAIR:
        # The previous FD-training residual was fast but it destroyed the BB learning trajectory
        # (relV stayed > 1 for many iterations).  Training residual is therefore restored to the
        # exact AD form.  Only the sampler map below uses a cheap FD proxy.
        def r_single(z: Array) -> Array:
            out = f_single(z)
            u = out[0]
            v = out[1]
            J = jax.jacfwd(f_single)(z)
            u_t = J[0, 0]
            u_x = J[0, 1]
            v_t = J[1, 0]
            v_x = J[1, 1]
            v_xxx = jax.grad(
                lambda zz: jax.hessian(lambda yy: f_single(yy)[1])(zz)[1, 1]
            )(z)[1]
            uv_x = u_x * v + u * v_x
            r_u = u_t - 2.0 * u * u_x - 0.5 * v_x
            r_v = v_t - 0.5 * v_xxx - 2.0 * uv_x
            return jnp.stack([r_u, r_v], axis=0)
        return jax.vmap(r_single)(coords)

    def pde_residual_fd_proxy_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        """Cheap forward-only FD proxy used only for adaptive sampling density.

        This must not be used as the training residual, because the BB run showed
        that pure FD training underfits v badly.  It is only a sampler heuristic.
        """
        if PDE_KIND != "BB":
            return self.pde_residual_at(params, geom, coords)

        coords = jnp.asarray(coords)
        dt = self.fd_dt.astype(coords.dtype)
        dx = self.fd_dx.astype(coords.dtype)
        tmin = jnp.asarray(self.bounds[0][0], dtype=coords.dtype)
        tmax = jnp.asarray(self.bounds[0][1], dtype=coords.dtype)
        xmin = jnp.asarray(self.bounds[1][0], dtype=coords.dtype)
        xmax = jnp.asarray(self.bounds[1][1], dtype=coords.dtype)

        t0 = jnp.clip(coords[:, 0], tmin + dt, tmax - dt)
        x0 = jnp.clip(coords[:, 1], xmin + 2.0 * dx, xmax - 2.0 * dx)
        base = jnp.stack([t0, x0], axis=1)
        tp = jnp.stack([t0 + dt, x0], axis=1)
        tm = jnp.stack([t0 - dt, x0], axis=1)
        xp = jnp.stack([t0, x0 + dx], axis=1)
        xm = jnp.stack([t0, x0 - dx], axis=1)
        xpp = jnp.stack([t0, x0 + 2.0 * dx], axis=1)
        xmm = jnp.stack([t0, x0 - 2.0 * dx], axis=1)

        all_pts = jnp.concatenate([base, tp, tm, xp, xm, xpp, xmm], axis=0)
        y = self.model_forward(params, geom, all_pts)
        n = coords.shape[0]
        y0, ytp, ytm, yxp, yxm, yxpp, yxmm = [y[i*n:(i+1)*n] for i in range(7)]

        u = y0[:, 0]
        v = y0[:, 1]
        u_t = (ytp[:, 0] - ytm[:, 0]) / (2.0 * dt)
        u_x = (yxp[:, 0] - yxm[:, 0]) / (2.0 * dx)
        v_t = (ytp[:, 1] - ytm[:, 1]) / (2.0 * dt)
        v_x = (yxp[:, 1] - yxm[:, 1]) / (2.0 * dx)
        v_xxx = (yxpp[:, 1] - 2.0 * yxp[:, 1] + 2.0 * yxm[:, 1] - yxmm[:, 1]) / (2.0 * dx**3)
        uv_x = u_x * v + u * v_x
        r_u = u_t - 2.0 * u * u_x - 0.5 * v_x
        r_v = v_t - 0.5 * v_xxx - 2.0 * uv_x
        return jnp.stack([r_u, r_v], axis=1)

    def model_ut_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        def f0(z):
            return self.model_forward(params, geom, z[None, :])[0, 0]
        return jax.vmap(lambda z: jax.grad(f0)(z)[0])(coords).reshape(-1, 1)

    def model_u_x_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        if PDE_KIND == "BB":
            coords = jnp.asarray(coords)
            dx = self.fd_dx.astype(coords.dtype)
            xmin = jnp.asarray(self.bounds[1][0], dtype=coords.dtype)
            xmax = jnp.asarray(self.bounds[1][1], dtype=coords.dtype)
            t0 = coords[:, 0]
            x0 = jnp.clip(coords[:, 1], xmin + dx, xmax - dx)
            xp = jnp.stack([t0, x0 + dx], axis=1)
            xm = jnp.stack([t0, x0 - dx], axis=1)
            y = self.model_forward(params, geom, jnp.concatenate([xp, xm], axis=0))
            n = coords.shape[0]
            return ((y[:n, 0] - y[n:, 0]) / (2.0 * dx)).reshape(-1, 1)

        def u_single(z):
            return self.model_forward(params, geom, z[None, :])[0, 0]
        return jax.vmap(lambda z: jax.grad(u_single)(z)[1])(coords).reshape(-1, 1)

    def model_v_derivatives_at(self, params, geom: Dict[str, Array], coords: Array) -> Tuple[Array, Array]:
        def v_single(z):
            return self.model_forward(params, geom, z[None, :])[0, 1]

        def one(z):
            g = jax.grad(v_single)(z)
            H = jax.hessian(v_single)(z)
            return jnp.stack([g[1], H[1, 1]], axis=0)

        out = jax.vmap(one)(coords)
        return out[:, 0:1], out[:, 1:2]

    def _importance_weights_from_q(self, q: Array, gamma: float = 0.0) -> Array:
        q = jnp.asarray(q).reshape(-1)
        w = (q + 1e-12) ** (-float(gamma))
        w = w / (jnp.mean(w) + 1e-12)
        if float(gamma) > 0.0:
            w = jnp.clip(w, 0.10, 10.0)
            w = w / (jnp.mean(w) + 1e-12)
        return w

    def global_residual_vector(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        spec = stage["spec"]
        parts = []

        coords = jnp.asarray(batch["colloc_coords"])
        q = jnp.asarray(batch["proposal_q"]).reshape(-1)
        r = self.pde_residual_at(params, geom, coords)
        if PDE_KIND == "BB":
            polish_phase = (float(self.latest_relU) <= float(self.cfg.polish_rel_u)) and (float(self.latest_relV) <= float(self.cfg.polish_rel_v))
            v_mul = float(self.cfg.polish_v_multiplier) if polish_phase else 1.0
            ramp = float(self._bb_residual_ramp())
            rw = jnp.asarray([self.cfg.bb_res_u_weight, self.cfg.bb_res_v_weight], dtype=r.dtype)
            r_eff = r * jnp.sqrt(ramp) * jnp.sqrt(rw)[None, :]
            gamma = float(self.cfg.importance_gamma)
        else:
            r_eff = r
            gamma = 0.0
        w = self._importance_weights_from_q(q, gamma=gamma).reshape(-1, 1)
        r_scale = math.sqrt(max(int(r_eff.size), 1))
        beta = float(spec.residual_beta)
        if beta > 0.0:
            parts.append(math.sqrt(beta) * jnp.reshape(r_eff * jnp.sqrt(w) / r_scale, (-1,)))
        if beta < 1.0:
            parts.append(math.sqrt(1.0 - beta) * jnp.reshape(r_eff / r_scale, (-1,)))

        coords_d = jnp.asarray(batch["data_coords"])
        vals_d = jnp.asarray(batch["data_values"])
        pred_d = self.model_forward(params, geom, coords_d)
        diff = pred_d - vals_d
        if PDE_KIND == "BB":
            v_data_w = float(self.cfg.bb_data_v_weight) * (v_mul if "v_mul" in locals() else 1.0)
            cw = jnp.asarray([self.cfg.bb_data_u_weight, v_data_w], dtype=diff.dtype)
            diff = diff / self.output_scale[None, :] * jnp.sqrt(cw)[None, :]
        parts.append(math.sqrt(float(spec.data_weight) / max(int(diff.size), 1)) * jnp.reshape(diff, (-1,)))

        if PDE_KIND == "BB":
            # Interior ridge anchors: value supervision, with stronger v weight.
            if batch.get("ridge_coords", jnp.empty((0, 2))).shape[0] > 0:
                rc = jnp.asarray(batch["ridge_coords"])
                rv = jnp.asarray(batch["ridge_values"])
                rp = self.model_forward(params, geom, rc)
                rd = (rp - rv) / self.output_scale[None, :]
                v_ridge_w = float(self.cfg.bb_ridge_v_weight) * (v_mul if "v_mul" in locals() else 1.0)
                rw = jnp.asarray([self.cfg.bb_ridge_u_weight, v_ridge_w], dtype=rd.dtype)
                rd = rd * jnp.sqrt(rw)[None, :]
                parts.append(math.sqrt(float(self.cfg.bb_ridge_anchor_weight) / max(int(rd.size), 1)) * jnp.reshape(rd, (-1,)))

            # Broad u-transition/fan anchors: target the relU plateau region.
            if batch.get("ufan_coords", jnp.empty((0, 2))).shape[0] > 0:
                uc = jnp.asarray(batch["ufan_coords"])
                uv = jnp.asarray(batch["ufan_values"])
                up = self.model_forward(params, geom, uc)
                # Mainly constrain u, but keep a small v component so the coupled fields stay coherent.
                ud = (up - uv) / self.output_scale[None, :]
                uw = jnp.asarray([self.cfg.bb_ufan_u_weight, 0.25], dtype=ud.dtype)
                ud = ud * jnp.sqrt(uw)[None, :]
                parts.append(math.sqrt(float(self.cfg.bb_ufan_anchor_weight) / max(int(ud.size), 1)) * jnp.reshape(ud, (-1,)))

            # Full-domain/interior value anchors. These are cheap compared with exact-AD
            # residuals and directly control relU/relV on the full solution manifold.
            if batch.get("interior_coords", jnp.empty((0, 2))).shape[0] > 0:
                ic = jnp.asarray(batch["interior_coords"])
                iv = jnp.asarray(batch["interior_values"])
                ip = self.model_forward(params, geom, ic)
                idiff = (ip - iv) / self.output_scale[None, :]
                iw = jnp.asarray([self.cfg.bb_interior_u_weight, self.cfg.bb_interior_v_weight], dtype=idiff.dtype)
                idiff = idiff * jnp.sqrt(iw)[None, :]
                parts.append(math.sqrt(float(self.cfg.bb_interior_anchor_weight) / max(int(idiff.size), 1)) * jnp.reshape(idiff, (-1,)))

            # u_x anchors on the broad transition fan.
            if float(self.cfg.bb_ux_weight) > 0.0 and batch.get("bb_ux_coords", jnp.empty((0, 2))).shape[0] > 0:
                uc = jnp.asarray(batch["bb_ux_coords"])
                ux_true = jnp.asarray(batch["bb_ux_values"])
                ux_pred = self.model_u_x_at(params, geom, uc)
                ux_diff = (ux_pred - ux_true) / self.ux_scale.reshape(1, 1)
                parts.append(math.sqrt(float(self.cfg.bb_ux_weight) / max(int(ux_diff.size), 1)) * jnp.reshape(ux_diff, (-1,)))

            # v_x and v_xx anchors: necessary because BB contains v_xxx, but now less dominant.
            if (float(self.cfg.bb_vx_weight) > 0.0 or float(self.cfg.bb_vxx_weight) > 0.0) and batch.get("bb_vx_coords", jnp.empty((0, 2))).shape[0] > 0:
                vc = jnp.asarray(batch["bb_vx_coords"])
                vx_true = jnp.asarray(batch["bb_vx_values"])
                vxx_true = jnp.asarray(batch["bb_vxx_values"])
                vx_pred, vxx_pred = self.model_v_derivatives_at(params, geom, vc)
                vx_diff = (vx_pred - vx_true) / self.vx_scale.reshape(1, 1)
                vxx_diff = (vxx_pred - vxx_true) / self.vxx_scale.reshape(1, 1)
                parts.append(math.sqrt(float(self.cfg.bb_vx_weight) / max(int(vx_diff.size), 1)) * jnp.reshape(vx_diff, (-1,)))
                parts.append(math.sqrt(float(self.cfg.bb_vxx_weight) / max(int(vxx_diff.size), 1)) * jnp.reshape(vxx_diff, (-1,)))

        if PDE_KIND == "KG" and batch["grad_coords"].shape[0] > 0:
            gc = jnp.asarray(batch["grad_coords"])
            gv = jnp.asarray(batch["grad_values"])
            ut = self.model_ut_at(params, geom, gc)
            gd = ut - gv
            parts.append(math.sqrt(float(spec.grad_weight) / max(int(gd.size), 1)) * jnp.reshape(gd, (-1,)))

        return jnp.concatenate(parts, axis=0)

    def fixed_metrics(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, float]:
        coords = jnp.asarray(batch["colloc_coords"])
        # Metrics are for logging only.  Using exact AD here doubles the expensive
        # v_xxx work per iteration.  Training still uses exact AD residual inside
        # global_residual_vector; logging uses the FD proxy to keep runtime sane.
        if PDE_KIND == "BB":
            r = self.pde_residual_fd_proxy_at(params, geom, coords)
        else:
            r = self.pde_residual_at(params, geom, coords)
        r2 = jnp.mean(r * r, axis=1)
        mse_fu = jnp.mean(r[:, 0] ** 2) if r.shape[1] >= 1 else jnp.mean(r2)
        mse_fv = jnp.mean(r[:, 1] ** 2) if r.shape[1] >= 2 else jnp.array(0.0, dtype=jnp.float64)

        data_coords = jnp.asarray(batch["data_coords"])
        data_vals = jnp.asarray(batch["data_values"])
        pred = self.model_forward(params, geom, data_coords)
        raw_diff = pred - data_vals
        data_loss = jnp.mean(raw_diff ** 2)
        data_loss_u = jnp.mean(raw_diff[:, 0] ** 2) if raw_diff.shape[1] >= 1 else data_loss
        data_loss_v = jnp.mean(raw_diff[:, 1] ** 2) if raw_diff.shape[1] >= 2 else jnp.array(0.0, dtype=jnp.float64)

        grad_loss = jnp.array(0.0, dtype=jnp.float64)
        ridge_loss = jnp.array(0.0, dtype=jnp.float64)
        ufan_loss = jnp.array(0.0, dtype=jnp.float64)
        interior_loss = jnp.array(0.0, dtype=jnp.float64)
        ux_loss = jnp.array(0.0, dtype=jnp.float64)
        vx_loss = jnp.array(0.0, dtype=jnp.float64)
        vxx_loss = jnp.array(0.0, dtype=jnp.float64)
        ridge_loss_weighted = jnp.array(0.0, dtype=jnp.float64)
        ufan_loss_weighted = jnp.array(0.0, dtype=jnp.float64)

        if PDE_KIND == "BB":
            polish_phase = (float(self.latest_relU) <= float(self.cfg.polish_rel_u)) and (float(self.latest_relV) <= float(self.cfg.polish_rel_v))
            v_mul = float(self.cfg.polish_v_multiplier) if polish_phase else 1.0
            ramp = float(self._bb_residual_ramp())
            rw = jnp.asarray([self.cfg.bb_res_u_weight, self.cfg.bb_res_v_weight], dtype=r.dtype)
            residual_loss_weighted = jnp.mean((r * jnp.sqrt(ramp) * jnp.sqrt(rw)[None, :]) ** 2)
            v_data_w = float(self.cfg.bb_data_v_weight) * v_mul
            cw = jnp.asarray([self.cfg.bb_data_u_weight, v_data_w], dtype=raw_diff.dtype)
            data_diff_weighted = raw_diff / self.output_scale[None, :] * jnp.sqrt(cw)[None, :]
            data_loss_weighted = jnp.mean(data_diff_weighted ** 2)

            if batch.get("ridge_coords", jnp.empty((0, 2))).shape[0] > 0:
                rc = jnp.asarray(batch["ridge_coords"])
                rv = jnp.asarray(batch["ridge_values"])
                rd = (self.model_forward(params, geom, rc) - rv) / self.output_scale[None, :]
                ridge_loss = jnp.mean(rd ** 2)
                v_ridge_w = float(self.cfg.bb_ridge_v_weight) * v_mul
                rw_anchor = jnp.asarray([self.cfg.bb_ridge_u_weight, v_ridge_w], dtype=rd.dtype)
                ridge_loss_weighted = jnp.mean((rd * jnp.sqrt(rw_anchor)[None, :]) ** 2)
            if batch.get("ufan_coords", jnp.empty((0, 2))).shape[0] > 0:
                uc = jnp.asarray(batch["ufan_coords"])
                uv = jnp.asarray(batch["ufan_values"])
                ud = (self.model_forward(params, geom, uc) - uv) / self.output_scale[None, :]
                ufan_loss = jnp.mean(ud[:, 0] ** 2)
                uw_anchor = jnp.asarray([self.cfg.bb_ufan_u_weight, 0.25], dtype=ud.dtype)
                ufan_loss_weighted = jnp.mean((ud * jnp.sqrt(uw_anchor)[None, :]) ** 2)
            if batch.get("interior_coords", jnp.empty((0, 2))).shape[0] > 0:
                ic = jnp.asarray(batch["interior_coords"])
                iv = jnp.asarray(batch["interior_values"])
                idiff = (self.model_forward(params, geom, ic) - iv) / self.output_scale[None, :]
                iw_anchor = jnp.asarray([self.cfg.bb_interior_u_weight, self.cfg.bb_interior_v_weight], dtype=idiff.dtype)
                interior_loss = jnp.mean((idiff * jnp.sqrt(iw_anchor)[None, :]) ** 2)
            if float(self.cfg.bb_ux_weight) > 0.0 and batch.get("bb_ux_coords", jnp.empty((0, 2))).shape[0] > 0:
                uc = jnp.asarray(batch["bb_ux_coords"])
                ux_true = jnp.asarray(batch["bb_ux_values"])
                ux_pred = self.model_u_x_at(params, geom, uc)
                ux_loss = jnp.mean(((ux_pred - ux_true) / self.ux_scale.reshape(1, 1)) ** 2)
            if (float(self.cfg.bb_vx_weight) > 0.0 or float(self.cfg.bb_vxx_weight) > 0.0) and batch.get("bb_vx_coords", jnp.empty((0, 2))).shape[0] > 0:
                vc = jnp.asarray(batch["bb_vx_coords"])
                vx_true = jnp.asarray(batch["bb_vx_values"])
                vxx_true = jnp.asarray(batch["bb_vxx_values"])
                vx_pred, vxx_pred = self.model_v_derivatives_at(params, geom, vc)
                vx_loss = jnp.mean(((vx_pred - vx_true) / self.vx_scale.reshape(1, 1)) ** 2)
                vxx_loss = jnp.mean(((vxx_pred - vxx_true) / self.vxx_scale.reshape(1, 1)) ** 2)

            loss_total = (
                residual_loss_weighted
                + float(stage["spec"].data_weight) * data_loss_weighted
                + float(self.cfg.bb_ridge_anchor_weight) * ridge_loss_weighted
                + float(self.cfg.bb_ufan_anchor_weight) * ufan_loss_weighted
                + float(self.cfg.bb_interior_anchor_weight) * interior_loss
                + float(self.cfg.bb_ux_weight) * ux_loss
                + float(self.cfg.bb_vx_weight) * vx_loss
                + float(self.cfg.bb_vxx_weight) * vxx_loss
            )
        else:
            if PDE_KIND == "KG" and batch["grad_coords"].shape[0] > 0:
                ut = self.model_ut_at(params, geom, jnp.asarray(batch["grad_coords"]))
                grad_loss = jnp.mean((ut - jnp.asarray(batch["grad_values"])) ** 2)
            loss_total = jnp.mean(r2) + float(stage["spec"].data_weight) * data_loss + float(stage["spec"].grad_weight) * grad_loss

        return {
            "loss_total": float(loss_total),
            "mse_f_train": float(jnp.mean(r2)),
            "mse_f_u": float(mse_fu),
            "mse_f_v": float(mse_fv),
            "tail_loss": float(jnp.quantile(r2, 0.90)),
            "data_loss": float(data_loss),
            "data_loss_u": float(data_loss_u),
            "data_loss_v": float(data_loss_v),
            "grad_loss": float(grad_loss),
            "ridge_loss": float(ridge_loss),
            "ufan_loss": float(ufan_loss),
            "interior_loss": float(interior_loss),
            "ux_loss": float(ux_loss),
            "vx_loss": float(vx_loss),
            "vxx_loss": float(vxx_loss),
            "max_abs_r": float(jnp.sqrt(jnp.max(r2))),
        }

    # -----------------------
    # sampling
    # -----------------------
    def _compute_residual_r2_batched(self, params, geom: Dict[str, Array], coords_np: NpArray, batch_size: int) -> NpArray:
        if PDE_KIND != "BB":
            outs = []
            for st in range(0, coords_np.shape[0], batch_size):
                ed = min(st + batch_size, coords_np.shape[0])
                xb = jax.device_put(jnp.asarray(coords_np[st:ed], dtype=jnp.float64), self.jax_device)
                rb = self.pde_residual_at(params, geom, xb)
                outs.append(np.asarray(jnp.mean(rb * rb, axis=1)))
            return np.concatenate(outs, axis=0)

        # Cheap adaptive map for BB:
        #   0.40 * normalized v error + 0.25 * normalized u fan error
        # + 0.35 * normalized FD-PDE residual proxy.
        # This avoids building the sampler from expensive high-order AD residuals.
        pred_parts = []
        res_parts = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jax.device_put(jnp.asarray(coords_np[st:ed], dtype=jnp.float64), self.jax_device)
            yb = self.model_forward(params, geom, xb)
            rb = self.pde_residual_fd_proxy_at(params, geom, xb)
            pred_parts.append(np.asarray(yb, dtype=np.float64))
            res_parts.append(np.asarray(float(self.cfg.bb_res_u_weight) * rb[:, 0] ** 2 + float(self.cfg.bb_res_v_weight) * rb[:, 1] ** 2, dtype=np.float64))
        pred = np.concatenate(pred_parts, axis=0)
        rproxy = np.concatenate(res_parts, axis=0)
        true = self.gt.bb_values_from_gt(coords_np)
        uerr = ((pred[:, 0] - true[:, 0]) / max(float(self.output_scale_np[0]), 1e-12)) ** 2
        verr = ((pred[:, 1] - true[:, 1]) / max(float(self.output_scale_np[1]), 1e-12)) ** 2
        # Weight u error toward the broad fan so sampler does not waste mass on flat plateaus.
        pts = np.stack([coords_np[:, 0], coords_np[:, 1]], axis=1)
        ux_abs = np.abs(self.gt._interp_ux(pts)).reshape(-1)
        fan_w = ux_abs / (np.max(ux_abs) + 1e-12)
        uerr = uerr * (0.25 + 0.75 * fan_w)

        def norm(a: NpArray) -> NpArray:
            a = np.asarray(a, dtype=np.float64)
            return a / (np.mean(a) + 1e-12)

        cheap = 0.40 * norm(verr) + 0.25 * norm(uerr) + 0.35 * norm(rproxy)
        return np.maximum(cheap, 1e-14)

    def _density_from_residual_map(self, r2: NpArray, spec: StageSpec) -> NpArray:
        p = np.maximum(np.asarray(r2, dtype=np.float64), 0.0) + 1e-14
        if spec.heat_sigma > 0:
            p = sp_ndimage.gaussian_filter(p, sigma=float(spec.heat_sigma), mode="nearest")
        power = min(float(spec.focus_power), float(self.cfg.sampler_power_cap)) if PDE_KIND == "BB" else float(spec.focus_power)
        if abs(power - 1.0) > 1e-15:
            p = np.maximum(p, 1e-14) ** power
        p = np.maximum(p, 1e-14)
        p = p / np.maximum(np.sum(p), 1e-12)

        if PDE_KIND == "BB":
            # Cap top-5% probability mass to stop residual spike chasing.
            mask = np.asarray(r2, dtype=np.float64) >= np.quantile(np.asarray(r2, dtype=np.float64), 0.95)
            top5 = float(np.sum(p[mask]))
            cap = float(self.cfg.sampler_top5_cap)
            if top5 > cap:
                uni = np.ones_like(p) / float(p.size)
                uni_top5 = float(np.sum(uni[mask]))
                alpha = (top5 - cap) / max(top5 - uni_top5, 1e-12)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                p = (1.0 - alpha) * p + alpha * uni
                p = p / np.maximum(np.sum(p), 1e-12)
        return p

    def _sample_from_density(self, p: NpArray, t_axis: NpArray, x_axis: NpArray, n: int) -> Tuple[NpArray, NpArray]:
        p = np.asarray(p, dtype=np.float64)
        flat = p.reshape(-1)
        flat = flat / np.maximum(np.sum(flat), 1e-12)
        idx = self.rng.choice(flat.size, size=int(n), replace=True, p=flat)
        T, X = np.meshgrid(t_axis, x_axis, indexing="ij")
        coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)[idx]
        q = flat[idx]
        return coords, q.reshape(-1, 1)

    def _build_sampling_state(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> Dict[str, Any]:
        spec = stage["spec"]
        r2 = self._compute_residual_r2_batched(params, geom, stage["coords"], self.cfg.residual_eval_batch_size)
        r2_grid = r2.reshape(spec.grid_t, spec.grid_x)
        p_grid_new = self._density_from_residual_map(r2_grid, spec)
        p_grid = p_grid_new
        if PDE_KIND == "BB":
            key = f"{spec.grid_t}_{spec.grid_x}"
            prev = self.sampling_ema_cache.get(key)
            if prev is not None and prev.shape == p_grid_new.shape:
                ema = float(np.clip(self.cfg.sampler_ema, 0.0, 0.98))
                p_grid = ema * prev + (1.0 - ema) * p_grid_new
                p_grid = np.maximum(p_grid, 1e-14)
                p_grid = p_grid / np.maximum(np.sum(p_grid), 1e-12)
            self.sampling_ema_cache[key] = p_grid.copy()
        top5_thr = np.quantile(r2_grid, 0.95)
        top5_mass = float(np.sum(p_grid[r2_grid >= top5_thr]))
        return {"residual_map": r2_grid, "p_grid": p_grid, "top5_mass": top5_mass}

    def _build_training_batch(self, sampling_state: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
        spec = stage["spec"]
        n_f = int(spec.n_f)
        n_uni = int(round(float(spec.uniform_mix) * n_f))
        n_adapt = max(n_f - n_uni, 0)

        pts = []
        qs = []
        if n_adapt > 0:
            c, q = self._sample_from_density(sampling_state["p_grid"], stage["t"], stage["x"], n_adapt)
            pts.append(c)
            qs.append(q)

        if n_uni > 0:
            uni = sample_uniform_box(self.rng, self.bounds, n_uni)
            pts.append(uni)
            qs.append(np.full((n_uni, 1), 1.0 / max(n_f, 1), dtype=np.float64))

        colloc = np.concatenate(pts, axis=0)
        proposal_q = np.concatenate(qs, axis=0)

        data = self.gt.sample_data(self.rng, int(spec.n_data), int(spec.n_grad))
        return {
            "colloc_coords": colloc.astype(np.float64),
            "proposal_q": proposal_q.astype(np.float64),
            "data_coords": data["data_coords"].astype(np.float64),
            "data_values": data["data_values"].astype(np.float64),
            "grad_coords": data.get("grad_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "grad_values": data.get("grad_values", np.empty((0, 1), dtype=np.float64)).astype(np.float64),
            "ridge_coords": data.get("ridge_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "ridge_values": data.get("ridge_values", np.empty((0, self.d_out), dtype=np.float64)).astype(np.float64),
            "ufan_coords": data.get("ufan_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "ufan_values": data.get("ufan_values", np.empty((0, self.d_out), dtype=np.float64)).astype(np.float64),
            "interior_coords": data.get("interior_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "interior_values": data.get("interior_values", np.empty((0, self.d_out), dtype=np.float64)).astype(np.float64),
            "bb_ux_coords": data.get("bb_ux_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "bb_ux_values": data.get("bb_ux_values", np.empty((0, 1), dtype=np.float64)).astype(np.float64),
            "bb_vx_coords": data.get("bb_vx_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "bb_vx_values": data.get("bb_vx_values", np.empty((0, 1), dtype=np.float64)).astype(np.float64),
            "bb_vxx_coords": data.get("bb_vxx_coords", np.empty((0, 2), dtype=np.float64)).astype(np.float64),
            "bb_vxx_values": data.get("bb_vxx_values", np.empty((0, 1), dtype=np.float64)).astype(np.float64),
        }

    def _batch_to_jax(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: jax.device_put(jnp.asarray(v, dtype=jnp.float64), self.jax_device) for k, v in batch.items()}

    # -----------------------
    # HF JTJ/CG step
    # -----------------------
    def _hf_clip_damping(self, damping: float) -> float:
        return max(float(damping), float(self.cfg.hf_damping_min))

    def theta_step(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]):
        batch_jax = self._batch_to_jax(batch)
        params = jax.device_put(params, self.jax_device)
        geom = jax.device_put(geom, self.jax_device)

        theta0, unravel = ravel_pytree(params)
        theta0 = jax.device_put(jnp.asarray(theta0), self.jax_device)

        def residual_from_flat(theta_flat: Array) -> Array:
            return self.global_residual_vector(unravel(theta_flat), geom, batch_jax, stage)

        residual_only = jax.jit(residual_from_flat)
        value_only = jax.jit(lambda th: jnp.vdot(residual_from_flat(th), residual_from_flat(th)))

        e0 = residual_only(theta0)
        obj_old = float(np.asarray(jnp.vdot(e0, e0), dtype=np.float64))
        _, vjp_res = jax.vjp(residual_from_flat, theta0)
        jt_e = vjp_res(e0)[0]
        g0 = 2.0 * jt_e
        g0_np = np.asarray(g0, dtype=np.float64)
        grad_norm = float(np.linalg.norm(g0_np))

        if not np.isfinite(obj_old) or not np.all(np.isfinite(g0_np)):
            self.last_theta_step_info = {"status": "rollback:nonfinite", "obj_old": obj_old, "obj_trial": obj_old, "grad_norm": grad_norm, "step_norm": 0.0, "cg_iters": 0, "cg_res_norm": float("nan"), "damping": self.hf_damping}
            return params

        _, jvp_res = jax.linearize(residual_from_flat, theta0)
        damping = self._hf_clip_damping(self.hf_damping)
        accepted = False
        params_trial = params
        obj_trial = obj_old
        step_norm = 0.0
        cg_res_norm = float("nan")
        ls_iters = 0
        status = "rollback:not_improved"

        for _trial in range(max(int(self.cfg.hf_max_trials), 1)):
            damping_arr = jax.device_put(jnp.asarray(damping, dtype=theta0.dtype), self.jax_device)

            def linop_fn(v: Array) -> Array:
                jv = jvp_res(v)
                return vjp_res(jv)[0] + damping_arr * v

            rhs = -jt_e
            tol_abs = max(float(stage["spec"].cg_tol) * max(float(np.linalg.norm(np.asarray(jt_e))), 1e-12), 1e-10)
            step, _ = jax_cg(linop_fn, rhs, tol=float(stage["spec"].cg_tol), atol=float(tol_abs), maxiter=int(stage["spec"].cg_maxiter))
            r_cg = linop_fn(step) - rhs
            cg_res_norm = float(np.asarray(jnp.linalg.norm(r_cg), dtype=np.float64))
            step_np = np.asarray(step, dtype=np.float64)

            if not np.all(np.isfinite(step_np)):
                step_np = -g0_np

            directional = float(np.dot(g0_np, step_np))
            if directional >= 0.0 or not np.isfinite(directional):
                step_np = -g0_np
                directional = -float(np.dot(g0_np, g0_np))

            step_norm = float(np.linalg.norm(step_np))
            step_dev = jax.device_put(jnp.asarray(step_np), self.jax_device)

            alpha = 1.0
            for ls in range(1, int(stage["spec"].line_search_maxiter) + 1):
                theta_try = theta0 + alpha * step_dev
                obj_try = float(np.asarray(value_only(theta_try), dtype=np.float64))
                if np.isfinite(obj_try) and obj_try <= obj_old + float(self.cfg.hf_line_search_c1) * alpha * directional:
                    params_trial = unravel(theta_try)
                    params_trial = jax.device_put(params_trial, self.jax_device)
                    obj_trial = obj_try
                    accepted = True
                    status = "applied:hf_jtj"
                    ls_iters = ls
                    break
                alpha *= float(self.cfg.hf_line_search_tau)

            if accepted:
                damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_down))
                break
            damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_up))

        self.hf_damping = float(damping)
        self.last_theta_step_info = {
            "status": status,
            "obj_old": float(obj_old),
            "obj_trial": float(obj_trial),
            "grad_norm": float(grad_norm),
            "step_norm": float(step_norm if accepted else 0.0),
            "cg_iters": int(stage["spec"].cg_maxiter),
            "cg_res_norm": float(cg_res_norm),
            "ls_iters": int(ls_iters),
            "damping": float(self.hf_damping),
        }
        return params_trial if accepted else params

    # -----------------------
    # plotting
    # -----------------------
    def _draw_dd_ellipses(self, ax, extent):
        centers = np.asarray(self.geom_np["centers"], dtype=np.float64)
        radii = np.asarray(self.geom_np["radii"], dtype=np.float64)
        G = np.asarray(self.geom_np["G_mats"], dtype=np.float64)
        ax.scatter(centers[:, 0], centers[:, 1], s=12, c="white", edgecolors="black", linewidths=0.4, zorder=4)
        for j in range(centers.shape[0]):
            Gj = 0.5 * (G[j] + G[j].T)
            vals, vecs = np.linalg.eigh(Gj)
            vals = np.clip(vals, 1e-12, None)
            width = 2.0 * radii[j] / math.sqrt(vals[0])
            height = 2.0 * radii[j] / math.sqrt(vals[1])
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse((centers[j, 0], centers[j, 1]), width=width, height=height, angle=angle, fill=False, color="white", linewidth=0.9, alpha=0.95)
            ell.set_clip_path(ax.patch)
            ax.add_patch(ell)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def save_best_snapshot_plot(self, out_dir: Path, snapshot: Dict[str, Any], rel_l2_epoch: float, it: int, loss_total: float):
        out_dir.mkdir(parents=True, exist_ok=True)
        x = snapshot["x"]
        t = snapshot["t"]
        pred = snapshot["pred_grid"]
        true = snapshot["true_grid"]
        T, X = np.meshgrid(t, x, indexing="ij")
        coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
        cover = cover_sum_np(coords, self.geom_np["centers"], self.geom_np["radii"], self.geom_np["G_mats"]).reshape(t.size, x.size).T
        extent = [float(t.min()), float(t.max()), float(x.min()), float(x.max())]

        if PDE_KIND == "KG":
            panels = [
                (pred[..., 0], "Pred u + DD", True),
                (true[..., 0], "True u", False),
                ((pred[..., 0] - true[..., 0]) ** 2, r"$|u-u^*|^2$", False),
                (cover, "cover sum + DD", True),
            ]
            fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.2), constrained_layout=True)
        else:
            panels = [
                (pred[..., 0], "Pred u + DD", True),
                (pred[..., 1], "Pred v + DD", True),
                (true[..., 0], "True u", False),
                (true[..., 1], "True v", False),
                ((pred[..., 0] - true[..., 0]) ** 2, r"$|u-u^*|^2$", False),
                ((pred[..., 1] - true[..., 1]) ** 2, r"$|v-v^*|^2$", False),
                (cover, "cover sum + DD", True),
                (np.sqrt((pred[..., 0]-true[..., 0])**2 + (pred[..., 1]-true[..., 1])**2), "vector error", False),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.2), constrained_layout=True)

        for ax, (arr, title, draw_dd) in zip(np.ravel(axes), panels):
            im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap="viridis")
            if draw_dd:
                self._draw_dd_ellipses(ax, extent)
            ax.set_xlabel("t")
            ax.set_ylabel("x")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)

        if PDE_KIND == "BB":
            sup = f"BEST BB relL2={rel_l2_epoch:.3e} | relU={snapshot['rel_u']:.3e} | relV={snapshot['rel_v']:.3e} | iter={it} | loss={loss_total:.3e}"
        else:
            sup = f"BEST KG relL2={rel_l2_epoch:.3e} | iter={it} | loss={loss_total:.3e}"
        fig.suptitle(sup)
        out_path = out_dir / f"best_relL2_snapshot_{PDE_KIND}_iter{it:06d}.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        log(f"[BEST-relL2-PLOT] saved -> {out_path}")

    # -----------------------
    # training
    # -----------------------
    def train(self, out_dir: Path) -> None:
        t0 = time.time()
        best_save_rel = float("inf")

        for it in range(1, self.cfg.iters + 1):
            self.current_iter = int(it)
            iter_t0 = time.time()
            stage_name = self._stage_name(it)
            self.current_stage_name = stage_name
            self.current_stage = self._make_stage(self.stage_specs[stage_name])
            self.current_stage["name"] = stage_name

            refresh_every = int(self.current_stage["spec"].refresh_every)
            refresh_due = (
                self.cached_batch is None
                or self.cached_batch.get("stage_name") != stage_name
                or ((it - 1) % refresh_every == 0)
            )

            if refresh_due:
                sampling_state = self._build_sampling_state(self.params, self.geom_state, self.current_stage)
                self._apply_slow_dd_update(sampling_state, self.current_stage, it)
                batch = self._build_training_batch(sampling_state, self.current_stage)
                self.cached_batch = {"stage_name": stage_name, "sampling_state": sampling_state, "batch": batch}
                block_pos = 1
            else:
                block_pos = ((it - 1) % refresh_every) + 1

            self.last_refresh_info = {"refreshed": int(refresh_due), "block_pos": int(block_pos), "block_len": int(refresh_every)}
            batch = self.cached_batch["batch"]
            self.params = self.theta_step(self.params, self.geom_state, batch, self.current_stage)

            metrics = self.fixed_metrics(self.params, self.geom_state, batch, self.current_stage)
            smp = self.cached_batch["sampling_state"]
            self.last_sampling_info = {
                "status": "applied:ema_capped_residual_plus_v_anchors",
                "top5": float(smp["top5_mass"]),
                "uniform_mix": float(self.current_stage["spec"].uniform_mix),
                "heat_sigma": float(self.current_stage["spec"].heat_sigma),
                "focus_power": float(self.current_stage["spec"].focus_power),
            }

            should_eval = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.rel_l2_eval_every == 0)
            quick_due = (PDE_KIND == "BB") and (it % max(int(self.cfg.quick_rel_eval_every), 1) == 0)
            rel_l2_epoch = float("nan")
            snapshot = None
            self.rel_source = "stale"
            if should_eval:
                rel_l2_epoch, snapshot = self.gt.eval_rel_l2(self.params, self.geom_state, self.predict_batched, self.cfg.test_batch_size)
                self.latest_relL2 = float(rel_l2_epoch)
                self.rel_source = "full"
                if snapshot is not None and PDE_KIND == "BB":
                    self.latest_relU = float(snapshot.get("rel_u", float("inf")))
                    self.latest_relV = float(snapshot.get("rel_v", float("inf")))
                if rel_l2_epoch < self.best_relL2:
                    delta = (best_save_rel - rel_l2_epoch) / max(best_save_rel, 1e-12) if np.isfinite(best_save_rel) else float("inf")
                    self.best_relL2 = rel_l2_epoch
                    self.best_relL2_iter = it
                    self.best_snapshot = snapshot
                    if (not np.isfinite(best_save_rel)) or delta >= self.cfg.save_plot_every_best_delta or it <= 3:
                        self.save_best_snapshot_plot(out_dir / f"result_ffusion_{PDE_KIND}", snapshot, rel_l2_epoch, it, metrics["loss_total"])
                        best_save_rel = rel_l2_epoch
            elif quick_due:
                q_rel, q_u, q_v = self.quick_monitor_rel_l2()
                if np.isfinite(q_rel):
                    self.latest_relL2 = float(q_rel)
                    self.latest_relU = float(q_u)
                    self.latest_relV = float(q_v)
                    rel_l2_epoch = float(q_rel)
                    self.rel_source = "mon"

            self.latest_monitor_f = metrics["mse_f_train"]

            if (it % self.cfg.print_every == 0) or it == 1 or it == self.cfg.iters:
                elapsed = time.time() - t0
                eta_txt = estimate_eta(t0, it, self.cfg.iters)
                th = self.last_theta_step_info
                rel_print = rel_l2_epoch if np.isfinite(rel_l2_epoch) else self.latest_relL2
                rel_extra = ""
                if PDE_KIND == "BB":
                    if snapshot is not None:
                        rel_extra = f" relU={snapshot['rel_u']:.3e} relV={snapshot['rel_v']:.3e}"
                    elif np.isfinite(self.latest_relU) and np.isfinite(self.latest_relV):
                        rel_extra = f" relU={self.latest_relU:.3e} relV={self.latest_relV:.3e}"
                log(
                    f"[ITER {it:6d}/{self.cfg.iters}] "
                    f"loss_total={metrics['loss_total']:.3e}  data_loss={metrics['data_loss']:.3e} "
                    f"data_u={metrics.get('data_loss_u',0.0):.3e} data_v={metrics.get('data_loss_v',0.0):.3e}  "
                    f"ridge={metrics.get('ridge_loss',0.0):.3e} ufan={metrics.get('ufan_loss',0.0):.3e} "
                    f"interior={metrics.get('interior_loss',0.0):.3e} "
                    f"ux={metrics.get('ux_loss',0.0):.3e} vx={metrics.get('vx_loss',0.0):.3e} vxx={metrics.get('vxx_loss',0.0):.3e}  "
                    f"mse_f_train={metrics['mse_f_train']:.3e} mse_fu={metrics.get('mse_f_u',0.0):.3e} mse_fv={metrics.get('mse_f_v',0.0):.3e}  "
                    f"tail_loss={metrics['tail_loss']:.3e}  max_abs_r={metrics['max_abs_r']:.3e}  "
                    f"relL2={rel_print:.3e}{rel_extra} rel_src={self.rel_source}  "
                    f"theta={th.get('status','na')}  sample={self.last_sampling_info['status']}  stage={stage_name}  "
                    f"refresh={self.last_refresh_info['refreshed']} block={self.last_refresh_info['block_pos']}/{self.last_refresh_info['block_len']}  "
                    f"dd={self.last_dd_info.get('status','na')} dd_move={self.last_dd_info.get('moved',0)} dd_shift={self.last_dd_info.get('max_shift',0.0):.2e}  "
                    f"top5={self.last_sampling_info['top5']:.3f} mix_uni={self.last_sampling_info['uniform_mix']:.2f} "
                    f"heat_sigma={self.last_sampling_info['heat_sigma']:.2f} power={self.last_sampling_info['focus_power']:.2f}  "
                    f"cg_iters={th.get('cg_iters',0)} cg_res={th.get('cg_res_norm',float('nan')):.3e} damping={th.get('damping',float('nan')):.2e}  "
                    f"iter_time={format_seconds(time.time() - iter_t0)} total={format_seconds(elapsed)} eta={eta_txt}"
                )

        log(f"[TRAIN-SUMMARY] best_relL2(iter={self.best_relL2_iter})={self.best_relL2:.3e}")


# ============================================================
# CLI
# ============================================================
def parse_freqs(s: str) -> Tuple[float, ...]:
    if s.strip() == "":
        return tuple()
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default="")
    p.add_argument("--gt_path", type=str, default="")
    p.add_argument("--gt_BB", type=str, default="")
    p.add_argument("--log_file", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--n_balls", type=int, default="")
    p.add_argument("--layers", type=int, default="")
    p.add_argument("--width", type=int, default="")
    p.add_argument("--act", type=str, default="", choices=["tanh", "relu", "silu"])
    p.add_argument("--freqs", type=str, default="")

    p.add_argument("--iters", type=int, default="")
    p.add_argument("--hf_damping_init", type=float, default="")
    p.add_argument("--hf_damping_min", type=float, default="")
    p.add_argument("--hf_damping_up", type=float, default="")
    p.add_argument("--hf_damping_down", type=float, default="")

    p.add_argument("--rel_l2_eval_every", type=int, default="")
    p.add_argument("--monitor_eval_every", type=int, default="")
    p.add_argument("--print_every", type=int, default="")
    p.add_argument("--save_plot_every_best_delta", type=float, default="")
    p.add_argument("--test_batch_size", type=int, default="")
    p.add_argument("--residual_eval_batch_size", type=int, default="")

    p.add_argument("--bb_res_u_weight", type=float, default="")
    p.add_argument("--bb_res_v_weight", type=float, default="")
    p.add_argument("--bb_data_u_weight", type=float, default="")
    p.add_argument("--bb_data_v_weight", type=float, default="")
    p.add_argument("--bb_ridge_u_weight", type=float, default="")
    p.add_argument("--bb_ridge_v_weight", type=float, default="")
    p.add_argument("--bb_ridge_anchor_weight", type=float, default="")
    p.add_argument("--bb_ufan_anchor_weight", type=float, default="")
    p.add_argument("--bb_ufan_u_weight", type=float, default="")
    p.add_argument("--bb_ux_weight", type=float, default="")
    p.add_argument("--bb_vx_weight", type=float, default="")
    p.add_argument("--bb_vxx_weight", type=float, default="")
    p.add_argument("--bb_interior_anchor_weight", type=float, default="")
    p.add_argument("--bb_interior_u_weight", type=float, default="")
    p.add_argument("--bb_interior_v_weight", type=float, default="")
    p.add_argument("--residual_warmup_floor", type=float, default="")
    p.add_argument("--residual_ramp_iters", type=int, default="")
    p.add_argument("--importance_gamma", type=float, default="")
    p.add_argument("--sampler_ema", type=float, default="")
    p.add_argument("--sampler_top5_cap", type=float, default="")
    p.add_argument("--sampler_power_cap", type=float, default="")
    p.add_argument("--force_mid_iter", type=int, default="")
    p.add_argument("--min_late_iter", type=int, default="")
    p.add_argument("--min_ultra_iter", type=int, default="")
    p.add_argument("--force_late_iter", type=int, default="")
    p.add_argument("--force_ultra_iter", type=int, default="")
    p.add_argument("--stage_mid_rel", type=float, default="")
    p.add_argument("--stage_late_rel_u", type=float, default="")
    p.add_argument("--stage_late_rel_v", type=float, default="")
    p.add_argument("--stage_ultra_rel_u", type=float, default="")
    p.add_argument("--stage_ultra_rel_v", type=float, default="")
    p.add_argument("--stage_mid_mse", type=float, default="")
    p.add_argument("--stage_late_mse", type=float, default="")
    p.add_argument("--stage_ultra_mse", type=float, default="")
    p.add_argument("--fd_dt", type=float, default="")
    p.add_argument("--fd_dx", type=float, default="")
    p.add_argument("--fd_dt_factor", type=float, default="")
    p.add_argument("--fd_dx_factor", type=float, default="")
    p.add_argument("--polish_rel_u", type=float, default="")
    p.add_argument("--polish_rel_v", type=float, default="")
    p.add_argument("--polish_v_multiplier", type=float, default="")
    p.add_argument("--quick_rel_eval_every", type=int, default="")
    p.add_argument("--quick_rel_eval_size", type=int, default="")
    p.add_argument("--dd_update_every", type=int, default="")
    p.add_argument("--dd_min_iter", type=int, default="")
    p.add_argument("--dd_center_lr", type=float, default="")
    p.add_argument("--dd_max_shift_rel", type=float, default="")
    p.add_argument("--dd_broad_lr_factor", type=float, default="")
    p.add_argument("--dd_ufan_lr_factor", type=float, default="")
    p.add_argument("--dd_vridge_lr_factor", type=float, default="")
    p.add_argument("--dd_cover_probe_n", type=int, default="")
    p.add_argument("--jax_enable_x64", action="store_true", help="Enable JAX x64. Default is float32 for speed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_handle = setup_logging(args.log_file)
    try:
        if int(args.n_balls) != FORCED_N_BALLS:
            raise ValueError(f"{PDE_LABEL} dedicated solver requires --n_balls {FORCED_N_BALLS}. Do not change it.")

        gt_path = getattr(args, "gt_BB")
        if gt_path is None:
            gt_path = args.gt_path

        rng = set_seed(args.seed)
        gt = GTAdapter(gt_path)
        cfg = Config(
            n_balls=FORCED_N_BALLS,
            layers=args.layers,
            width=args.width,
            act=args.act,
            freqs=parse_freqs(args.freqs),
            iters=args.iters,
            hf_damping_init=args.hf_damping_init,
            hf_damping_min=args.hf_damping_min,
            hf_damping_up=args.hf_damping_up,
            hf_damping_down=args.hf_damping_down,
            rel_l2_eval_every=args.rel_l2_eval_every,
            monitor_eval_every=args.monitor_eval_every,
            print_every=args.print_every,
            save_plot_every_best_delta=args.save_plot_every_best_delta,
            test_batch_size=args.test_batch_size,
            residual_eval_batch_size=args.residual_eval_batch_size,
            bb_res_u_weight=args.bb_res_u_weight,
            bb_res_v_weight=args.bb_res_v_weight,
            bb_data_u_weight=args.bb_data_u_weight,
            bb_data_v_weight=args.bb_data_v_weight,
            bb_ridge_u_weight=args.bb_ridge_u_weight,
            bb_ridge_v_weight=args.bb_ridge_v_weight,
            bb_ridge_anchor_weight=args.bb_ridge_anchor_weight,
            bb_ufan_anchor_weight=args.bb_ufan_anchor_weight,
            bb_ufan_u_weight=args.bb_ufan_u_weight,
            bb_ux_weight=args.bb_ux_weight,
            bb_vx_weight=args.bb_vx_weight,
            bb_vxx_weight=args.bb_vxx_weight,
            bb_interior_anchor_weight=args.bb_interior_anchor_weight,
            bb_interior_u_weight=args.bb_interior_u_weight,
            bb_interior_v_weight=args.bb_interior_v_weight,
            residual_warmup_floor=args.residual_warmup_floor,
            residual_ramp_iters=args.residual_ramp_iters,
            importance_gamma=args.importance_gamma,
            sampler_ema=args.sampler_ema,
            sampler_top5_cap=args.sampler_top5_cap,
            sampler_power_cap=args.sampler_power_cap,
            force_mid_iter=args.force_mid_iter,
            min_late_iter=args.min_late_iter,
            min_ultra_iter=args.min_ultra_iter,
            force_late_iter=args.force_late_iter,
            force_ultra_iter=args.force_ultra_iter,
            stage_mid_rel=args.stage_mid_rel,
            stage_late_rel_u=args.stage_late_rel_u,
            stage_late_rel_v=args.stage_late_rel_v,
            stage_ultra_rel_u=args.stage_ultra_rel_u,
            stage_ultra_rel_v=args.stage_ultra_rel_v,
            stage_mid_mse=args.stage_mid_mse,
            stage_late_mse=args.stage_late_mse,
            stage_ultra_mse=args.stage_ultra_mse,
            fd_dt=args.fd_dt,
            fd_dx=args.fd_dx,
            fd_dt_factor=args.fd_dt_factor,
            fd_dx_factor=args.fd_dx_factor,
            polish_rel_u=args.polish_rel_u,
            polish_rel_v=args.polish_rel_v,
            polish_v_multiplier=args.polish_v_multiplier,
            quick_rel_eval_every=args.quick_rel_eval_every,
            quick_rel_eval_size=args.quick_rel_eval_size,
            dd_update_every=args.dd_update_every,
            dd_min_iter=args.dd_min_iter,
            dd_center_lr=args.dd_center_lr,
            dd_max_shift_rel=args.dd_max_shift_rel,
            dd_broad_lr_factor=args.dd_broad_lr_factor,
            dd_ufan_lr_factor=args.dd_ufan_lr_factor,
            dd_vridge_lr_factor=args.dd_vridge_lr_factor,
            dd_cover_probe_n=args.dd_cover_probe_n,
        )
        solver = PINNFFusionSolver(gt, cfg, rng)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_geom_params = solver.geom_np["centers"].shape[0] * (solver.d_in + 1 + solver.d_in * solver.d_in)
        total_params = solver.theta_size + total_geom_params



if __name__ == "__main__":
    main()




