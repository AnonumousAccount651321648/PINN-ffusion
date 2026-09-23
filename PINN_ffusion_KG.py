#!/usr/bin/env python3
from __future__ import annotations


import argparse
import builtins
import math
import sys
import time
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
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg


Array = jnp.ndarray
NpArray = np.ndarray
BoundsType = Tuple[Tuple[float, float], ...]

PDE_KIND = "KG"
PDE_LABEL = "Klein-Gordon equation"
DEFAULT_GT = ""
FORCED_N_BALLS = 4
D_OUT = 1


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




def kg_boundary_base_jax(tx: Array, mode: str = "sin") -> Array:
    """Hard KG boundary/initial-condition base term."""
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    if mode == "cos":
        g = jnp.cos(5.0 * jnp.pi * t) + t**3
    else:
        g = jnp.sin(5.0 * jnp.pi * t) + t**3
    return x * g


def kg_hard_envelope_jax(tx: Array) -> Array:
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    return x * (1.0 - x) * t**2

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
            self.bounds = ((float(self.t[0]), float(self.t[-1])), (float(self.x[0]), float(self.x[-1])))
            self.d_out = 2
        else:
            raise ValueError(PDE_KIND)

    def exact(self, tx: NpArray) -> NpArray:
        if PDE_KIND == "KG":
            return kg_exact_np(tx, getattr(self, "kg_mode", "sin"))
        return bb_exact_np(tx)

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
        return {
            "data_coords": coords,
            "data_values": vals,
            "grad_coords": np.empty((0, 2), dtype=np.float64),
            "grad_values": np.empty((0, 1), dtype=np.float64),
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
        centers.append([tm, xm])
        Gs.append(make_diag_metric_from_axes((0.82 * Lt, 0.82 * Lx)))

        centers.append([tm, xmin + 0.17 * Lx])
        Gs.append(make_diag_metric_from_axes((0.78 * Lt, 0.32 * Lx)))

        centers.append([tm, xmax - 0.17 * Lx])
        Gs.append(make_diag_metric_from_axes((0.78 * Lt, 0.32 * Lx)))

        # Five balls along the analytical steep tanh transition band w(t,x)=0.
        # The band is close to x≈-t for the left/early part and x≈-2t for the right/late part.
        t_centers = np.linspace(tmin + 0.12 * Lt, tmax - 0.12 * Lt, 5)
        angle = math.atan2(-1.55 * (Lt / max(Lx, 1e-12)), 1.0)
        for tc in t_centers:
            xc = -1.55 * tc
            xc = float(np.clip(xc, xmin + 0.08 * Lx, xmax - 0.08 * Lx))
            centers.append([float(tc), xc])
            Gs.append(make_rot_metric(0.28 * Lt, 0.14 * Lx, angle))

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
    top5_cap: float = ""


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

    enable_rel_rollback: bool = ""
    rollback_rel_growth: float = ""
    rollback_best_growth: float = ""
    rollback_damping_up: float = ""


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

        self.params = tuple(
            make_expert_params(self.rng, self.feature_dim, self.d_out, cfg.layers, cfg.width)
            for _ in range(cfg.n_balls)
        )
        self.theta_size = int(sum(ravel_pytree(p)[0].size for p in self.params))
        self.jax_device = require_jax_gpu()
        self.params = jax.device_put(self.params, self.jax_device)
        self.geom_state = jax.device_put(self.geom_state, self.jax_device)

        self.stage_specs = self._make_stage_specs()
        self.stage_cache: Dict[str, Dict[str, Any]] = {}
        self.current_stage_name = "early"
        self.current_stage = self._make_stage(self.stage_specs["early"])
        self.cached_batch: Optional[Dict[str, Any]] = None
        self.stage_level = 0
        self.stage_names_order = ("early", "mid", "late", "ultra")

        self.hf_damping = float(cfg.hf_damping_init)
        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.latest_relL2 = float("inf")
        self.latest_monitor_f = float("inf")
        self.best_snapshot = None
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_sampling_info: Dict[str, Any] = {}
        self.last_refresh_info: Dict[str, Any] = {"refreshed": 1, "block_pos": 1, "block_len": 1}

    def _make_stage_specs(self) -> Dict[str, StageSpec]:
        if PDE_KIND == "KG":
            # KG is smooth, not shock-like. Keep sampling mostly uniform and
            # cap residual concentration. Hard ansatz enforces IC/BC exactly,
            # so data/grad losses are logging-only with zero training weight.
            return {
                "early": StageSpec(64, 64, 1024, 256, 128, 0.90, 2, 32, 1e-4, 0.70, 1.50, 0.0, 0.0, 0.0, 12, 0.25),
                "mid":   StageSpec(80, 80, 1536, 384, 128, 0.85, 4, 64, 1e-4, 0.80, 1.20, 0.0, 0.0, 0.0, 14, 0.30),
                "late":  StageSpec(96, 96, 2048, 512, 128, 0.80, 6, 96, 7e-5, 0.90, 1.00, 0.0, 0.0, 0.0, 16, 0.35),
                "ultra": StageSpec(128,128,3072,512, 128, 0.75, 8,128, 5e-5, 1.00, 0.80, 0.0, 0.0, 0.0, 18, 0.35),
            }
        return {
            "early": StageSpec(56, 72, 512, 768, 0, 0.25, 1, 24, 1e-4, 1.0, 1.2, 1.0, 0.0, 0.0, 10, 0.35),
            "mid":   StageSpec(72, 96, 768,1024, 0, 0.20, 2, 48, 1e-4, 1.2, 1.0, 1.0, 0.0, 0.0, 12, 0.35),
            "late":  StageSpec(96,128,1024,1536,0, 0.15, 3, 96, 5e-5, 1.5, 0.8, 1.0, 0.0, 0.0, 14, 0.35),
            "ultra": StageSpec(120,160,1280,2048,0,0.10, 4,128, 2e-5, 1.8, 0.6, 1.0, 0.0, 0.0, 16, 0.35),
        }

    def _stage_name(self, it: int) -> str:
        # Monotone stage hysteresis. Do not bounce mid <-> late on temporary
        # residual-refresh spikes. Late starts only after the global KG shape is
        # already reasonably stable.
        rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
        mon = float(self.latest_monitor_f) if np.isfinite(self.latest_monitor_f) else float("inf")
        target_level = self.stage_level
        if target_level < 1 and (rel <= 0.60 or mon <= 5e-2):
            target_level = 1
        if target_level < 2 and (rel <= 0.08 or mon <= 5e-3):
            target_level = 2
        if target_level < 3 and (rel <= 0.015 or mon <= 5e-4):
            target_level = 3
        self.stage_level = max(self.stage_level, target_level)
        return self.stage_names_order[self.stage_level]

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
    def raw_network_forward(self, params, geom: Dict[str, Array], x: Array) -> Array:
        d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
        ph = phi_basis(x, geom)
        lam = lambdas_from_phi(ph, d2)
        z = local_coords(x, geom)
        zf = fourier_features(z.reshape(-1, self.d_in), self.freqs).reshape(x.shape[0], self.cfg.n_balls, self.feature_dim)
        outs = [expert_apply(params[j], zf[:, j, :], self.act_name) for j in range(self.cfg.n_balls)]
        Y = jnp.stack(outs, axis=1)
        return jnp.sum(lam[:, :, None] * Y, axis=1)

    def model_forward(self, params, geom: Dict[str, Array], x: Array) -> Array:
        raw = self.raw_network_forward(params, geom, x)
        if PDE_KIND == "KG":
            mode = getattr(self.gt, "kg_mode", "sin")
            return kg_boundary_base_jax(x, mode) + kg_hard_envelope_jax(x) * raw[:, 0:1]
        return raw

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

    def model_ut_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        def f0(z):
            return self.model_forward(params, geom, z[None, :])[0, 0]
        return jax.vmap(lambda z: jax.grad(f0)(z)[0])(coords).reshape(-1, 1)

    def _importance_weights_from_q(self, q: Array, gamma: float = 0.0) -> Array:
        q = jnp.asarray(q).reshape(-1)
        w = (q + 1e-12) ** (-float(gamma))
        w = w / (jnp.mean(w) + 1e-12)
        return w

    def global_residual_vector(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        spec = stage["spec"]
        parts = []

        coords = jnp.asarray(batch["colloc_coords"])
        q = jnp.asarray(batch["proposal_q"]).reshape(-1)
        r = self.pde_residual_at(params, geom, coords)
        w = self._importance_weights_from_q(q, gamma=0.0).reshape(-1, 1)
        r_scale = math.sqrt(max(int(r.size), 1))
        beta = float(spec.residual_beta)
        if beta > 0.0:
            parts.append(math.sqrt(beta) * jnp.reshape(r * jnp.sqrt(w) / r_scale, (-1,)))
        if beta < 1.0:
            parts.append(math.sqrt(1.0 - beta) * jnp.reshape(r / r_scale, (-1,)))

        if float(spec.data_weight) > 0.0 and batch["data_coords"].shape[0] > 0:
            coords_d = jnp.asarray(batch["data_coords"])
            vals_d = jnp.asarray(batch["data_values"])
            pred_d = self.model_forward(params, geom, coords_d)
            diff = pred_d - vals_d
            parts.append(math.sqrt(float(spec.data_weight) / max(int(diff.size), 1)) * jnp.reshape(diff, (-1,)))

        if float(spec.grad_weight) > 0.0 and PDE_KIND == "KG" and batch["grad_coords"].shape[0] > 0:
            gc = jnp.asarray(batch["grad_coords"])
            gv = jnp.asarray(batch["grad_values"])
            ut = self.model_ut_at(params, geom, gc)
            gd = ut - gv
            parts.append(math.sqrt(float(spec.grad_weight) / max(int(gd.size), 1)) * jnp.reshape(gd, (-1,)))

        return jnp.concatenate(parts, axis=0)

    def fixed_metrics(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, float]:
        coords = jnp.asarray(batch["colloc_coords"])
        r = self.pde_residual_at(params, geom, coords)
        r2 = jnp.mean(r * r, axis=1)
        data_coords = jnp.asarray(batch["data_coords"])
        data_vals = jnp.asarray(batch["data_values"])
        pred = self.model_forward(params, geom, data_coords)
        data_loss = jnp.mean((pred - data_vals) ** 2)
        grad_loss = jnp.array(0.0, dtype=jnp.float64)
        if PDE_KIND == "KG" and batch["grad_coords"].shape[0] > 0:
            ut = self.model_ut_at(params, geom, jnp.asarray(batch["grad_coords"]))
            grad_loss = jnp.mean((ut - jnp.asarray(batch["grad_values"])) ** 2)

        loss_total = jnp.mean(r2) + float(stage["spec"].data_weight) * data_loss + float(stage["spec"].grad_weight) * grad_loss
        return {
            "loss_total": float(loss_total),
            "mse_f_train": float(jnp.mean(r2)),
            "tail_loss": float(jnp.quantile(r2, 0.90)),
            "data_loss": float(data_loss),
            "grad_loss": float(grad_loss),
            "max_abs_r": float(jnp.sqrt(jnp.max(r2))),
        }

    # -----------------------
    # sampling
    # -----------------------
    def _compute_residual_r2_batched(self, params, geom: Dict[str, Array], coords_np: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jax.device_put(jnp.asarray(coords_np[st:ed], dtype=jnp.float64), self.jax_device)
            rb = self.pde_residual_at(params, geom, xb)
            r2 = jnp.mean(rb * rb, axis=1)
            outs.append(np.asarray(r2))
        return np.concatenate(outs, axis=0)

    def _density_from_residual_map(self, r2: NpArray, spec: StageSpec) -> NpArray:
        r2 = np.maximum(np.asarray(r2, dtype=np.float64), 0.0)
        p = r2 + 1e-14
        if spec.heat_sigma > 0:
            p = sp_ndimage.gaussian_filter(p, sigma=float(spec.heat_sigma), mode="nearest")
        if abs(float(spec.focus_power) - 1.0) > 1e-15:
            p = np.maximum(p, 1e-14) ** float(spec.focus_power)
        p = np.maximum(p, 1e-14)
        p = p / np.maximum(np.sum(p), 1e-12)

        cap = float(getattr(spec, "top5_cap", 1.0))
        if np.isfinite(cap) and cap < 0.999:
            thr = np.quantile(r2, 0.95)
            mask = r2 >= thr
            mass = float(np.sum(p[mask]))
            uniform = np.full_like(p, 1.0 / p.size)
            uniform_mass = float(np.sum(uniform[mask]))
            if mass > cap and mass > uniform_mass + 1e-12:
                lam = (mass - cap) / max(mass - uniform_mass, 1e-12)
                lam = float(np.clip(lam, 0.0, 1.0))
                p = (1.0 - lam) * p + lam * uniform
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
        p_grid = self._density_from_residual_map(r2_grid, spec)
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
            "grad_coords": data["grad_coords"].astype(np.float64),
            "grad_values": data["grad_values"].astype(np.float64),
        }

    def _batch_to_jax(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: jax.device_put(jnp.asarray(v, dtype=jnp.float64), self.jax_device) for k, v in batch.items()}

    # -----------------------
    # HF JTJ/CG step
    # -----------------------
    def _hf_current_damping_floor(self) -> float:
        rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
        floor = float(self.cfg.hf_damping_min)
        if rel <= 5.0e-3:
            floor = min(floor, 1.0e-6)
        elif rel <= 2.0e-2:
            floor = min(floor, 1.0e-5)
        return floor

    def _hf_clip_damping(self, damping: float) -> float:
        return max(float(damping), self._hf_current_damping_floor())

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
                batch = self._build_training_batch(sampling_state, self.current_stage)
                self.cached_batch = {"stage_name": stage_name, "sampling_state": sampling_state, "batch": batch}
                block_pos = 1
            else:
                block_pos = ((it - 1) % refresh_every) + 1

            self.last_refresh_info = {"refreshed": int(refresh_due), "block_pos": int(block_pos), "block_len": int(refresh_every)}
            batch = self.cached_batch["batch"]
            prev_params = self.params
            prev_rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
            self.params = self.theta_step(self.params, self.geom_state, batch, self.current_stage)

            metrics = self.fixed_metrics(self.params, self.geom_state, batch, self.current_stage)
            smp = self.cached_batch["sampling_state"]
            self.last_sampling_info = {
                "status": "applied:heat_diffusion_residual_sampling",
                "top5": float(smp["top5_mass"]),
                "uniform_mix": float(self.current_stage["spec"].uniform_mix),
                "heat_sigma": float(self.current_stage["spec"].heat_sigma),
                "focus_power": float(self.current_stage["spec"].focus_power),
            }

            should_eval = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.rel_l2_eval_every == 0)
            rel_l2_epoch = float("nan")
            snapshot = None
            if should_eval:
                rel_l2_epoch, snapshot = self.gt.eval_rel_l2(self.params, self.geom_state, self.predict_batched, self.cfg.test_batch_size)

                rolled_back = False
                if bool(self.cfg.enable_rel_rollback) and np.isfinite(prev_rel) and it > 5:
                    bad_vs_prev = rel_l2_epoch > float(self.cfg.rollback_rel_growth) * max(prev_rel, 1e-12)
                    bad_vs_best = (np.isfinite(self.best_relL2) and
                                   rel_l2_epoch > float(self.cfg.rollback_best_growth) * max(self.best_relL2, 1e-12))
                    if bad_vs_prev and bad_vs_best:
                        self.params = prev_params
                        self.hf_damping = self._hf_clip_damping(self.hf_damping * float(self.cfg.rollback_damping_up))
                        self.cached_batch = None
                        rel_l2_epoch = prev_rel
                        snapshot = None
                        metrics = self.fixed_metrics(self.params, self.geom_state, batch, self.current_stage)
                        self.last_theta_step_info["status"] = self.last_theta_step_info.get("status", "theta") + "|rel_rollback"
                        rolled_back = True

                self.latest_relL2 = float(rel_l2_epoch)
                if (not rolled_back) and rel_l2_epoch < self.best_relL2:
                    delta = (best_save_rel - rel_l2_epoch) / max(best_save_rel, 1e-12) if np.isfinite(best_save_rel) else float("inf")
                    self.best_relL2 = rel_l2_epoch
                    self.best_relL2_iter = it
                    self.best_snapshot = snapshot
                    if snapshot is not None and ((not np.isfinite(best_save_rel)) or delta >= self.cfg.save_plot_every_best_delta or it <= 3):
                        self.save_best_snapshot_plot(out_dir / f"result_ffusion_{PDE_KIND}", snapshot, rel_l2_epoch, it, metrics["loss_total"])
                        best_save_rel = rel_l2_epoch

            self.latest_monitor_f = metrics["mse_f_train"]

            if (it % self.cfg.print_every == 0) or it == 1 or it == self.cfg.iters:
                elapsed = time.time() - t0
                eta_txt = estimate_eta(t0, it, self.cfg.iters)
                th = self.last_theta_step_info
                rel_extra = ""
                if snapshot is not None and PDE_KIND == "BB":
                    rel_extra = f" relU={snapshot['rel_u']:.3e} relV={snapshot['rel_v']:.3e}"
                log(
                    f"[ITER {it:6d}/{self.cfg.iters}] "
                    f"loss_total={metrics['loss_total']:.3e}  data_loss={metrics['data_loss']:.3e}  grad_loss={metrics['grad_loss']:.3e}  "
                    f"mse_f_train={metrics['mse_f_train']:.3e}  tail_loss={metrics['tail_loss']:.3e}  max_abs_r={metrics['max_abs_r']:.3e}  "
                    f"relL2={rel_l2_epoch:.3e}{rel_extra}  "
                    f"theta={th.get('status','na')}  sample={self.last_sampling_info['status']}  stage={stage_name}  "
                    f"refresh={self.last_refresh_info['refreshed']} block={self.last_refresh_info['block_pos']}/{self.last_refresh_info['block_len']}  "
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
    p.add_argument("--gt_KG", type=str, default="")
    p.add_argument("--log_file", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--n_balls", type=int, default="")
    p.add_argument("--layers", type=int, default="")
    p.add_argument("--width", type=int, default="")
    p.add_argument("--act", type=str, default="", choices=["tanh", "relu", "silu"])
    p.add_argument("--freqs", type=str, default="")

    p.add_argument("--iters", type=int, default="")
    p.add_argument("--hf_damping_init", type=float, default="")
    p.add_argument("--hf_damping_up", type=float, default="")
    p.add_argument("--hf_damping_down", type=float, default="")
    p.add_argument("--hf_damping_min", type=float, default="")

    p.add_argument("--rel_l2_eval_every", type=int, default="")
    p.add_argument("--monitor_eval_every", type=int, default="")
    p.add_argument("--print_every", type=int, default="")
    p.add_argument("--test_batch_size", type=int, default="")
    p.add_argument("--residual_eval_batch_size", type=int, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_handle = setup_logging(args.log_file)
    try:
        if int(args.n_balls) != FORCED_N_BALLS:
            raise ValueError(f"{PDE_LABEL} dedicated solver requires --n_balls {FORCED_N_BALLS}. Do not change it.")

        gt_path = getattr(args, "gt_KG")
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
            test_batch_size=args.test_batch_size,
            residual_eval_batch_size=args.residual_eval_batch_size,
        )
        solver = PINNFFusionSolver(gt, cfg, rng)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_geom_params = solver.geom_np["centers"].shape[0] * (solver.d_in + 1 + solver.d_in * solver.d_in)
        total_params = solver.theta_size + total_geom_params

        log(
            f"[CONFIG] pde={PDE_KIND} n_balls={cfg.n_balls} layers={cfg.layers} width={cfg.width} "
            f"iters={cfg.iters} theta_optimizer=hf_jtj_gpu gt={gt_path} "
            f"staged_grid={[(s.grid_t, s.grid_x) for s in solver.stage_specs.values()]} "
            f"freqs={cfg.freqs} diffusion_sampler=uniform_heavy_top5_capped hard_bc_ansatz=on"
        )
        log(f"[MODEL] approx_total_params={total_params} theta_params={solver.theta_size} geom_params~={total_geom_params}")
        log(f"[JAX] forced_device={solver.jax_device} default_backend={jax.default_backend()} bounds={solver.bounds}")

        solver.train(out_dir)
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
