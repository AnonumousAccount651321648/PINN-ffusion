#!/usr/bin/env python3
from __future__ import annotations



import argparse
import builtins
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat

from scipy import fft as sp_fft
from scipy import ndimage as sp_ndimage
from scipy import optimize as sp_opt
from scipy.interpolate import RegularGridInterpolator

from PDE import make_pde as torch_make_pde

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad as jax_grad
from jax import vmap as jax_vmap
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg


Array = jnp.ndarray
NpArray = np.ndarray
BoundsType = Tuple[Tuple[float, float], ...]


base = sys.modules[__name__]

DEFAULT_GT_BURGERS = ""


def require_jax_gpu() -> Any:
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if not gpus:
        raise RuntimeError(
            "This script requires a JAX GPU backend for theta-side J^T J HF / Gauss-Newton updates, but no GPU device was found."
        )
    return gpus[0]


# ============================================================
# basic utils
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
    if done <= 0 or total <= 0 or done > total:
        return "--:--"
    elapsed = time.time() - start_time
    rate = elapsed / done
    remain = rate * (total - done)
    return format_seconds(remain)


def normalize_bounds(bounds: BoundsType) -> BoundsType:
    out = []
    for b in bounds:
        lo, hi = float(b[0]), float(b[1])
        if not lo < hi:
            raise ValueError(f"Invalid bound {b}: require lower < upper")
        out.append((lo, hi))
    return tuple(out)


def rel_l2_np(pred: NpArray, truth: NpArray) -> float:
    num = float(np.sqrt(np.mean((pred - truth) ** 2) + 1e-18))
    den = float(np.sqrt(np.mean(truth ** 2) + 1e-18))
    return num / den


def meshgrid_ij_np(a: NpArray, b: NpArray):
    return np.meshgrid(a, b, indexing="ij")


def latin_hypercube_np(rng: np.random.Generator, n: int, d: int) -> NpArray:
    u = rng.random((n, d))
    cut = np.linspace(0.0, 1.0, n + 1)
    lower = cut[:-1, None]
    upper = cut[1:, None]
    pts = lower + (upper - lower) * u
    out = np.empty_like(pts)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = pts[perm, j]
    return np.clip(out, 0.0, 1.0)


def sample_box(rng: np.random.Generator, n: int, bounds: BoundsType, use_lhs: bool) -> NpArray:
    d = len(bounds)
    u = latin_hypercube_np(rng, n, d) if use_lhs else rng.random((n, d))
    mins = np.array([b[0] for b in bounds], dtype=np.float64)
    maxs = np.array([b[1] for b in bounds], dtype=np.float64)
    return mins + (maxs - mins) * u


def np_bucketize_interp1d(query: NpArray, grid: NpArray, values: NpArray) -> NpArray:
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    g = np.asarray(grid, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    q = np.clip(q, g[0], g[-1])
    if g.size == 1:
        return np.full((q.size, 1), v[0], dtype=np.float64)
    idx = np.searchsorted(g, q, side="right")
    idx = np.clip(idx, 1, g.size - 1)
    x0 = g[idx - 1]
    x1 = g[idx]
    y0 = v[idx - 1]
    y1 = v[idx]
    w = (q - x0) / (x1 - x0 + 1e-12)
    out = y0 + w * (y1 - y0)
    return out.reshape(-1, 1)


def np_interp2d_bilinear(q0: NpArray, q1: NpArray, grid0: NpArray, grid1: NpArray, values: NpArray) -> NpArray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    g0 = np.asarray(grid0, dtype=np.float64).reshape(-1)
    g1 = np.asarray(grid1, dtype=np.float64).reshape(-1)
    V = np.asarray(values, dtype=np.float64)

    q0 = np.clip(q0, g0[0], g0[-1])
    q1 = np.clip(q1, g1[0], g1[-1])

    if g0.size == 1 and g1.size == 1:
        return np.full((q0.size, 1), V[0, 0], dtype=np.float64)
    if g0.size == 1:
        return np_bucketize_interp1d(q1, g1, V[0, :])
    if g1.size == 1:
        return np_bucketize_interp1d(q0, g0, V[:, 0])

    i0 = np.searchsorted(g0, q0, side="right")
    i1 = np.searchsorted(g1, q1, side="right")
    i0 = np.clip(i0, 1, g0.size - 1)
    i1 = np.clip(i1, 1, g1.size - 1)

    g00 = g0[i0 - 1]
    g01 = g0[i0]
    g10 = g1[i1 - 1]
    g11 = g1[i1]
    w0 = (q0 - g00) / (g01 - g00 + 1e-12)
    w1 = (q1 - g10) / (g11 - g10 + 1e-12)

    v00 = V[i0 - 1, i1 - 1]
    v01 = V[i0 - 1, i1]
    v10 = V[i0, i1 - 1]
    v11 = V[i0, i1]

    out = (1.0 - w0) * (1.0 - w1) * v00 + (1.0 - w0) * w1 * v01 + w0 * (1.0 - w1) * v10 + w0 * w1 * v11
    return out.reshape(-1, 1)


def jax_interp2d_bilinear(q0: Array, q1: Array, grid0: Array, grid1: Array, values: Array) -> Array:
    q0 = jnp.reshape(q0, (-1,))
    q1 = jnp.reshape(q1, (-1,))
    q0 = jnp.clip(q0, grid0[0], grid0[-1])
    q1 = jnp.clip(q1, grid1[0], grid1[-1])
    i0 = jnp.searchsorted(grid0, q0, side="right")
    i1 = jnp.searchsorted(grid1, q1, side="right")
    i0 = jnp.clip(i0, 1, grid0.shape[0] - 1)
    i1 = jnp.clip(i1, 1, grid1.shape[0] - 1)
    g00 = grid0[i0 - 1]
    g01 = grid0[i0]
    g10 = grid1[i1 - 1]
    g11 = grid1[i1]
    w0 = (q0 - g00) / (g01 - g00 + 1e-12)
    w1 = (q1 - g10) / (g11 - g10 + 1e-12)
    v00 = values[i0 - 1, i1 - 1]
    v01 = values[i0 - 1, i1]
    v10 = values[i0, i1 - 1]
    v11 = values[i0, i1]
    out = (1.0 - w0) * (1.0 - w1) * v00 + (1.0 - w0) * w1 * v01 + w0 * (1.0 - w1) * v10 + w0 * w1 * v11
    return out.reshape(-1, 1)



# ============================================================
# PDE.py adapter / GT helpers
# ============================================================
def _sync_torch_seed_from_rng(rng: np.random.Generator) -> None:
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)


def _to_numpy(x: torch.Tensor) -> NpArray:
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def load_navier_mat(path: str) -> Dict[str, Any]:
    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]

    def pick(*names):
        for n in names:
            if n in mat:
                return mat[n]
        return None

    X_star = pick("X_star", "XSTAR", "X")
    t = pick("t", "t_star", "T")
    U_star = pick("U_star", "USTAR", "U")
    p_star = pick("p_star", "p", "P_star", "P")
    w = pick("w", "omega", "vorticity", "w_star", "W")

    if X_star is not None and U_star is not None:
        X_star = np.asarray(X_star, dtype=np.float64)
        if X_star.ndim == 2 and X_star.shape[1] == 2:
            pass
        elif X_star.ndim == 2 and X_star.shape[0] == 2:
            X_star = X_star.T
        else:
            X_star = X_star.reshape(-1, 2)
        if t is None:
            t = np.array([0.0], dtype=np.float64)
        else:
            t = np.asarray(t, dtype=np.float64).squeeze()
        U_star = np.asarray(U_star, dtype=np.float64)
        if U_star.ndim == 2 and U_star.shape[1] == 2:
            U_star = U_star[:, :, None]
        elif U_star.ndim != 3:
            N = X_star.shape[0]
            U_star = U_star.reshape(N, 2, -1)
        if p_star is not None:
            p_star = np.asarray(p_star, dtype=np.float64)
            if p_star.ndim == 1:
                p_star = p_star[:, None]
            elif p_star.ndim > 2:
                p_star = p_star.reshape(X_star.shape[0], -1)
        return {
            "kind": "uvp",
            "X_star": X_star,
            "t": t,
            "U_star": U_star,
            "p_star": p_star,
            "keys": keys,
        }

    if w is not None:
        w_arr = np.asarray(w, dtype=np.float64)
        x = pick("x", "X")
        y = pick("y", "Y")
        if x is not None and y is not None:
            x1 = np.asarray(x, dtype=np.float64).reshape(-1)
            y1 = np.asarray(y, dtype=np.float64).reshape(-1)
            w1 = np.asarray(w_arr, dtype=np.float64).reshape(-1)
            if x1.size == y1.size == w1.size:
                pts = np.stack([x1, y1], axis=1)
                return {"kind": "vorticity_points", "X_star": pts, "w": w1, "keys": keys}
            if w_arr.ndim == 2:
                Nx, Ny = x1.size, y1.size
                if w_arr.shape == (Ny, Nx):
                    return {"kind": "vorticity_grid", "x": x1, "y": y1, "w": w_arr.T, "keys": keys}
                if w_arr.shape == (Nx, Ny):
                    return {"kind": "vorticity_grid", "x": x1, "y": y1, "w": w_arr, "keys": keys}
        if X_star is not None:
            Xs = np.asarray(X_star, dtype=np.float64)
            if Xs.ndim == 2 and Xs.shape[1] == 2:
                pts = Xs
            elif Xs.ndim == 2 and Xs.shape[0] == 2:
                pts = Xs.T
            else:
                pts = Xs.reshape(-1, 2)
            w1 = np.asarray(w_arr, dtype=np.float64).reshape(-1)
            if pts.shape[0] == w1.size:
                return {"kind": "vorticity_points", "X_star": pts, "w": w1, "keys": keys}

    raise KeyError(f"navier mat format not recognized. keys={keys}")


class TorchPDEAdapter:
    def __init__(self, pde_name: str, gt_burgers: str, gt_helmholtz: str, gt_navier: str):
        self.raw_name = pde_name.lower().strip()
        self.device = torch.device('cpu')
        self.dtype = torch.float64
        self.gt_navier_path = gt_navier
        self.navier_gt = None

        if self.raw_name in ('burgers', 'burgers1d', 'burgers_1d'):
            self.torch_pde = torch_make_pde(
                'burgers_1d',
                gt_mat_path=gt_burgers,
                use_gt_bounds=True,
                device=self.device,
                dtype=self.dtype,
                use_lhs=True,
            )
            self.residual_dim = 1
            self.nu = float(self.torch_pde.nu)
        elif self.raw_name in ('helmholtz', 'helmholtz2d', 'helmholtz_2d'):
            self.torch_pde = torch_make_pde(
                'helmholtz_2d',
                gt_mat_path=gt_helmholtz,
                use_gt_bounds=True,
                device=self.device,
                dtype=self.dtype,
                use_lhs=True,
            )
            self.residual_dim = 1
            self.reaction_coeff = float(self.torch_pde.reaction_coeff)
            self.x_j = jnp.asarray(_to_numpy(self.torch_pde._gt_x))
            self.y_j = jnp.asarray(_to_numpy(self.torch_pde._gt_y))
            self.f_j = jnp.asarray(_to_numpy(self.torch_pde._gt_f))
        elif self.raw_name in ('navier_stokes', 'navier-stokes', 'ns', 'ns2d', 'navier_stokes_2d'):
            self.torch_pde = torch_make_pde(
                'navier_stokes_2d',
                Re=100.0,
                cylinder_radius=0.25,
                cylinder_center=(0.0, 0.0),
                t_bounds=(0.0, 16.0),
                x_bounds=(-2.5, 7.5),
                y_bounds=(-2.5, 2.5),
                device=self.device,
                dtype=self.dtype,
                use_lhs=True,
            )
            self.residual_dim = 3
            self.nu = float(self.torch_pde.nu)
            if gt_navier and Path(gt_navier).exists():
                try:
                    self.navier_gt = load_navier_mat(gt_navier)
                except Exception:
                    self.navier_gt = None
        else:
            raise ValueError(f'Unknown PDE name: {pde_name}')

        self.name = self.torch_pde.name
        self.coord_kind = self.torch_pde.coord_kind
        self.d_in = self.torch_pde.d_in
        self.d_out = self.torch_pde.d_out
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in self.torch_pde.bounds)

    def sample_data(self, rng: np.random.Generator, n_ic: int, n_bc: int):
        _sync_torch_seed_from_rng(rng)
        out = self.torch_pde.sample_data(n_ic, n_bc)
        if isinstance(out, tuple) and len(out) == 3:
            coords_t, vals_t, mask_t = out
            return _to_numpy(coords_t), _to_numpy(vals_t), _to_numpy(mask_t)
        coords_t, vals_t = out
        mask = np.ones_like(_to_numpy(vals_t), dtype=np.float64)
        return _to_numpy(coords_t), _to_numpy(vals_t), mask

    def sample_collocation(self, rng: np.random.Generator, n_f: int) -> NpArray:
        _sync_torch_seed_from_rng(rng)
        return _to_numpy(self.torch_pde.sample_collocation(n_f))

    def exact_solution(self, coords: NpArray) -> Optional[NpArray]:
        if not hasattr(self.torch_pde, 'exact_solution'):
            return None
        vals = self.torch_pde.exact_solution(torch.tensor(coords, device=self.device, dtype=self.dtype))
        if vals is None:
            return None
        return _to_numpy(vals)

    def filter_out_cylinder(self, coords: NpArray) -> NpArray:
        if self.name != 'navier_stokes_2d':
            return np.ones((coords.shape[0],), dtype=bool)
        x = coords[:, 1]
        y = coords[:, 2]
        cx, cy = float(self.torch_pde.cx), float(self.torch_pde.cy)
        r = float(self.torch_pde.r)
        return ((x - cx) ** 2 + (y - cy) ** 2) >= (r * r)

    def eval_rel_l2(self, params: Any, geom: Dict[str, Array], predict_fn, batch_size: int) -> Tuple[float, Dict[str, Any]]:
        if self.name == 'burgers_1d':
            t = _to_numpy(self.torch_pde._gt_t)
            x = _to_numpy(self.torch_pde._gt_x)
            u_true = _to_numpy(self.torch_pde._gt_usol)
            T, X = meshgrid_ij_np(t, x)
            coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
            pred = predict_fn(params, geom, coords, batch_size)
            u_pred = pred.reshape(t.size, x.size, -1)[..., 0]
            rel = rel_l2_np(u_pred, u_true)
            return rel, {
                'grid_coords': coords,
                'horizontal': t,
                'vertical': x,
                'pred_grid': u_pred,
                'true_grid': u_true,
                'coord_kind': 'tx',
                'h_label': 't',
                'v_label': 'x',
                'aspect': 'auto',
            }
        if self.name == 'helmholtz_2d':
            x = _to_numpy(self.torch_pde._gt_x)
            y = _to_numpy(self.torch_pde._gt_y)
            u_true = _to_numpy(self.torch_pde._gt_u)
            X, Y = meshgrid_ij_np(x, y)
            coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
            pred = predict_fn(params, geom, coords, batch_size)
            u_pred = pred.reshape(x.size, y.size, -1)[..., 0]
            rel = rel_l2_np(u_pred, u_true)
            return rel, {
                'grid_coords': coords,
                'horizontal': x,
                'vertical': y,
                'pred_grid': u_pred,
                'true_grid': u_true,
                'coord_kind': 'xy',
                'h_label': 'x',
                'v_label': 'y',
                'aspect': 'equal',
            }
        if self.name == 'navier_stokes_2d' and self.navier_gt is not None and self.navier_gt.get('kind') == 'uvp':
            gt = self.navier_gt
            X = np.asarray(gt['X_star'], dtype=np.float64)
            t = np.asarray(gt['t'], dtype=np.float64).reshape(-1)
            U = np.asarray(gt['U_star'], dtype=np.float64)
            P = None if gt.get('p_star', None) is None else np.asarray(gt['p_star'], dtype=np.float64)
            ti = 0
            N = X.shape[0]
            coords = np.concatenate([np.full((N, 1), float(t[ti]), dtype=np.float64), X[:, 0:1], X[:, 1:2]], axis=1)
            pred = predict_fn(params, geom, coords, batch_size)
            truth = np.concatenate([
                U[:, 0, ti:ti+1],
                U[:, 1, ti:ti+1],
                np.zeros((N, 1), dtype=np.float64) if P is None else P[:, ti:ti+1],
            ], axis=1)
            errs = [rel_l2_np(pred[:, 0:1], truth[:, 0:1]), rel_l2_np(pred[:, 1:2], truth[:, 1:2])]
            if P is not None:
                p_true = truth[:, 2:3]
                p_true_n = p_true - p_true.mean()
                p_pred_n = pred[:, 2:3] - pred[:, 2:3].mean()
                errs.append(rel_l2_np(p_pred_n, p_true_n))
            rel = float(np.mean(errs))
            x_unique = np.unique(X[:, 0])
            y_unique = np.unique(X[:, 1])
            Nx, Ny = x_unique.size, y_unique.size
            grid_coords = np.concatenate([
                np.full((Nx * Ny, 1), float(t[ti]), dtype=np.float64),
                np.stack(np.meshgrid(x_unique, y_unique, indexing='ij'), axis=-1).reshape(-1, 2),
            ], axis=1)
            pred_grid = predict_fn(params, geom, grid_coords, batch_size).reshape(Nx, Ny, 3)
            truth_grid = truth.reshape(Nx, Ny, 3)
            return rel, {
                'grid_coords': grid_coords,
                'horizontal': x_unique,
                'vertical': y_unique,
                'pred_grid': pred_grid,
                'true_grid': truth_grid,
                'coord_kind': 'txy',
                'h_label': 'x',
                'v_label': 'y',
                'aspect': 'equal',
                't_plot': float(t[ti]),
            }
        raise RuntimeError('relL2 evaluation GT is unavailable for this PDE configuration.')


def make_pde_spec(name: str, gt_burgers: str, gt_helmholtz: str, gt_navier: str) -> TorchPDEAdapter:
    return TorchPDEAdapter(name, gt_burgers, gt_helmholtz, gt_navier)

# ============================================================
# model and geometry
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


def make_expert_params(rng: np.random.Generator, d_in: int, d_out: int, hidden_layers: int, hidden_width: int) -> Tuple[Tuple[Array, Array], ...]:
    layers = []
    dims = [d_in] + [hidden_width] * hidden_layers + [d_out]
    for din, dout in zip(dims[:-1], dims[1:]):
        W = jnp.asarray(xavier_init(rng, din, dout))
        b = jnp.zeros((dout,), dtype=jnp.float64)
        layers.append((W, b))
    return tuple(layers)


def expert_apply(params: Tuple[Tuple[Array, Array], ...], x: Array, act_name: str) -> Array:
    h = x
    L = len(params)
    for i, (W, b) in enumerate(params):
        h = h @ W.T + b
        if i < L - 1:
            h = activation(act_name, h)
    return h


def sigmoid_inv(y: NpArray) -> NpArray:
    y = np.clip(y, 1e-9, 1.0 - 1e-9)
    return np.log(y / (1.0 - y))


def softplus_inv(y: NpArray) -> NpArray:
    y = np.maximum(y, 1e-12)
    return np.log(np.expm1(y))


def actual_geometry(phi_raw: Dict[str, Array], mins: Array, maxs: Array, min_radius: float) -> Dict[str, Array]:
    centers = mins[None, :] + jax.nn.sigmoid(phi_raw["center_raw"]) * (maxs - mins)[None, :]
    radii = min_radius + jax.nn.softplus(phi_raw["radius_raw"])
    return {"centers": centers, "radii": radii.reshape(-1)}


def phi_basis(x: Array, centers: Array, radii: Array, eps: float = 1e-12) -> Array:
    diff = x[:, None, :] - centers[None, :, :]
    dist2 = jnp.sum(diff * diff, axis=2)
    s = 1.0 - dist2 / (radii[None, :] * radii[None, :] + eps)
    return jnp.maximum(s, 0.0) ** 2


def lambdas_from_phi(ph: Array, eps: float = 1e-12) -> Array:
    return ph / (jnp.sum(ph, axis=1, keepdims=True) + eps)


def local_coords(x: Array, centers: Array, radii: Array, eps: float = 1e-12) -> Array:
    return (x[:, None, :] - centers[None, :, :]) / (radii[None, :, None] + eps)


def model_forward(params: Sequence[Tuple[Tuple[Array, Array], ...]], geom: Dict[str, Array], x: Array, act_name: str) -> Array:
    ph = phi_basis(x, geom["centers"], geom["radii"])
    lam = lambdas_from_phi(ph)
    z = local_coords(x, geom["centers"], geom["radii"])
    outs = [expert_apply(params[j], z[:, j, :], act_name) for j in range(len(params))]
    Y = jnp.stack(outs, axis=1)
    return jnp.sum(lam[:, :, None] * Y, axis=1)


def component_normalizers(geom: Dict[str, Array], integration_points: Array, admissible_volume: float, eps: float = 1e-12) -> Array:
    ph = phi_basis(integration_points, geom["centers"], geom["radii"], eps=eps)
    Z = admissible_volume * jnp.mean(ph, axis=0)
    return jnp.maximum(Z, eps)


def component_weights(geom: Dict[str, Array], integration_points: Array, admissible_volume: float, eps: float = 1e-12) -> Array:
    return 1.0 / component_normalizers(geom, integration_points, admissible_volume, eps=eps)


def continuous_density(geom: Dict[str, Array], x: Array, integration_points: Array, admissible_volume: float, eps: float = 1e-12) -> Array:
    ph = phi_basis(x, geom["centers"], geom["radii"], eps=eps)
    w = component_weights(geom, integration_points, admissible_volume, eps=eps)
    p = jnp.mean(ph * w[None, :], axis=1)
    return jnp.maximum(p, eps).reshape(-1, 1)


# ============================================================
# optimizer
# ============================================================
@dataclass
class AdamState:
    m: Any
    v: Any
    t: int


def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(m=zeros, v=zeros, t=0)


def adam_update(params, grads, state: AdamState, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    t = state.t + 1
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state.v, grads)
    mhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2 ** t), v)
    new_params = jax.tree_util.tree_map(lambda p, m1, v1: p - lr * m1 / (jnp.sqrt(v1) + eps), params, mhat, vhat)
    return new_params, AdamState(m=m, v=v, t=t)


# ============================================================
# LM helpers
# ============================================================
@dataclass
class TrainConfig:
    n_ic: int = ""
    n_bc: int = ""
    n_r_pool: int = ""
    batch_f: int = ""
    batch_u: int = ""
    iters: int = ""
    n_balls: int = ""
    layers: int = ""
    width: int = ""
    act: str = ""
    init_radius_scale: float = ""
    dd_min_radius_scale: float = ""
    dd_init_kmeans_points: int = ""
    dd_kmeans_iters: int = ""
    theta_optimizer: str = ""
    use_lm: bool = ""
    lm_damping_init: float = ""
    lm_damping_up: float = ""
    lm_damping_down: float = ""
    lm_eta_min: float = ""
    lm_step_norm_cap: float = ""
    lm_max_trials: int = ""
    lbfgs_maxiter: int = ""
    lbfgs_maxfun: int = ""
    lbfgs_maxls: int = ""
    lbfgs_pgtol: float = ""
    lbfgs_factr: float = ""
    phi_steps: int = ""
    beta: float = ""
    lr_phi: float = ""
    phi_lr_gamma: float = ""
    phi_lr_min: float = ""
    normalizer_points: int = ""
    rel_l2_eval_every: int = ""
    print_every: int = ""
    test_batch_size: int = ""
    paper_repro: bool = ""


@dataclass
class ResidualBatch:
    coords: NpArray
    proposal_density: NpArray


class ExpertMeta:
    def __init__(self, params):
        flat, unravel = ravel_pytree(params)
        self.size = int(flat.size)
        self.unravel = unravel


# ============================================================
# solver
# ============================================================
class PINNBallsSolver:
    def __init__(self, pde: TorchPDEAdapter, cfg: TrainConfig, rng: np.random.Generator):
        self.pde = pde
        self.cfg = cfg
        self.rng = rng
        self.bounds = normalize_bounds(pde.bounds)
        self.d_in = pde.d_in
        self.d_out = pde.d_out
        self.residual_dim = pde.residual_dim
        self.act_name = cfg.act
        self.mins_np = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.maxs_np = np.array([b[1] for b in self.bounds], dtype=np.float64)
        self.diag = float(np.linalg.norm(self.maxs_np - self.mins_np))
        self.min_radius = float(cfg.dd_min_radius_scale) * self.diag
        self.mins = jnp.asarray(self.mins_np)
        self.maxs = jnp.asarray(self.maxs_np)

        self.data_coords_full, self.data_vals_full, self.data_mask_full = self.pde.sample_data(self.rng, cfg.n_ic, cfg.n_bc)
        self.norm_points = jnp.asarray(self.pde.sample_collocation(self.rng, cfg.normalizer_points))
        self.dd_cover_points = self._initial_domain_cover_points()
        self.admissible_volume = self._estimate_admissible_volume(8192)

        expert_in = self.d_in
        self.params = tuple(
            make_expert_params(self.rng, expert_in, self.d_out, cfg.layers, cfg.width)
            for _ in range(cfg.n_balls)
        )
        self.expert_metas = [ExpertMeta(p) for p in self.params]
        self.expert_offsets = np.cumsum([0] + [m.size for m in self.expert_metas[:-1]]).astype(np.int64)
        self.theta_size = int(sum(m.size for m in self.expert_metas))

        self.phi_params = self._init_phi_params_from_kmeans()
        self.phi_opt_state = adam_init(self.phi_params)
        self.hf_damping = float(cfg.hf_damping_init)
        self.lm_damping = float(cfg.lm_damping_init)
        self.theta_adam_state = base.adam_init(self.params)

        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.best_params = self.params
        self.best_phi_params = self.phi_params
        self.best_eval_snapshot: Optional[Dict[str, Any]] = None
        self.latest_relL2: float = float("inf")
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_phi_step_info: Dict[str, Any] = {}

    # -------------------------
    # geometry init / volume
    # -------------------------
    def _estimate_admissible_volume(self, n_mc: int) -> float:
        pts = sample_box(self.rng, n_mc, self.bounds, use_lhs=False)
        if self.pde.name == 'navier_stokes_2d':
            mask = self.pde.filter_out_cylinder(pts)
            frac = float(mask.mean())
        else:
            frac = 1.0
        vol = float(np.prod(self.maxs_np - self.mins_np)) * frac
        return vol

    def _init_phi_params_from_kmeans(self) -> Dict[str, Array]:
        pts = self.pde.sample_collocation(self.rng, max(self.cfg.dd_init_kmeans_points, 16 * self.cfg.n_balls))
        if pts.shape[0] > self.cfg.dd_init_kmeans_points:
            sel = self.rng.choice(pts.shape[0], size=self.cfg.dd_init_kmeans_points, replace=False)
            pts = pts[sel]

        # Paper-faithful interpretation for initialization:
        # - centers c_j are obtained by K-means over the collocation points
        # - radii s_j start from the cluster standard deviation
        # - then they are enlarged only as much as needed to cover the assigned training samples
        #
        # No runtime growth loop, no overlap rollback, and no post-init repair heuristics are used.
        cover_pts = pts
        if self.data_coords_full is not None and self.data_coords_full.shape[0] > 0:
            cover_pts = np.concatenate([cover_pts, self.data_coords_full], axis=0)

        N = pts.shape[0]
        M = min(self.cfg.n_balls, N)
        idx = self.rng.permutation(N)[:M]
        centers = pts[idx].copy()

        for _ in range(max(1, int(self.cfg.dd_kmeans_iters))):
            dist2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assign = np.argmin(dist2, axis=1)
            new_centers = []
            for j in range(M):
                mask = assign == j
                if np.any(mask):
                    new_centers.append(pts[mask].mean(axis=0))
                else:
                    ridx = self.rng.integers(0, N)
                    new_centers.append(pts[ridx])
            centers = np.stack(new_centers, axis=0)

        dist2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dist2, axis=1)

        cover_dist2 = np.sum((cover_pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        cover_assign = np.argmin(cover_dist2, axis=1)

        min_r = max(self.min_radius, 1e-6)
        radii = []

        for j in range(M):
            cluster = pts[assign == j]
            if cluster.shape[0] == 0:
                r_std = min_r
            else:
                d = np.sqrt(np.sum((cluster - centers[j]) ** 2, axis=1) + 1e-12)
                r_std = max(float(d.std(ddof=0)), min_r)

            assigned_cover = cover_pts[cover_assign == j]
            if assigned_cover.shape[0] == 0:
                r_cover = min_r
            else:
                d_cover = np.sqrt(np.sum((assigned_cover - centers[j]) ** 2, axis=1) + 1e-12)
                r_cover = max(float(np.max(d_cover)), min_r)

            # minimum radius that still covers the assigned training samples
            rj = max(r_std, r_cover)
            radii.append(rj)

        radii = np.asarray(radii, dtype=np.float64).reshape(-1, 1)

        if M < self.cfg.n_balls:
            extra = self.cfg.n_balls - M
            centers_extra = pts[self.rng.integers(0, N, size=extra)]
            radii_extra = np.full((extra, 1), min_r, dtype=np.float64)
            centers = np.concatenate([centers, centers_extra], axis=0)
            radii = np.concatenate([radii, radii_extra], axis=0)

        center_unit = (centers - self.mins_np[None, :]) / (self.maxs_np - self.mins_np)[None, :]
        center_raw = sigmoid_inv(center_unit)
        radius_raw = softplus_inv(np.maximum(radii - self.min_radius, 1e-9))
        return {
            "center_raw": jnp.asarray(center_raw),
            "radius_raw": jnp.asarray(radius_raw),
        }

    def geom(self, phi_params: Optional[Dict[str, Array]] = None) -> Dict[str, Array]:
        if phi_params is None:
            phi_params = self.phi_params
        return actual_geometry(phi_params, self.mins, self.maxs, self.min_radius)

    # -------------------------
    # batched prediction
    # -------------------------
    def predict_batched(self, params, geom: Dict[str, Array], coords: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords.shape[0], batch_size):
            ed = min(st + batch_size, coords.shape[0])
            xb = jnp.asarray(coords[st:ed])
            yb = model_forward(params, geom, xb, self.act_name)
            outs.append(np.asarray(yb))
        return np.concatenate(outs, axis=0)

    def cover_sum_batched(self, geom: Dict[str, Array], coords: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords.shape[0], batch_size):
            ed = min(st + batch_size, coords.shape[0])
            xb = jnp.asarray(coords[st:ed])
            ph = phi_basis(xb, geom["centers"], geom["radii"])
            outs.append(np.asarray(jnp.sum(ph, axis=1)))
        return np.concatenate(outs, axis=0)

    # -------------------------
    # residuals and losses
    # -------------------------
    def residual_tensor(self, params, geom: Dict[str, Array], f_coords: Array) -> Array:
        pde_name = self.pde.name
        if pde_name == "burgers_1d":
            nu = float(self.pde.nu)
            def scalar_u(coord):
                return model_forward(params, geom, coord[None, :], self.act_name)[0, 0]
            du = jax_vmap(jax_grad(scalar_u))(f_coords)
            hess = jax_vmap(jax.hessian(scalar_u))(f_coords)
            u = model_forward(params, geom, f_coords, self.act_name)[:, 0]
            u_t = du[:, 0]
            u_x = du[:, 1]
            u_xx = hess[:, 1, 1]
            return (u_t + u * u_x - nu * u_xx).reshape(-1, 1)
        if pde_name == "helmholtz_2d":
            reaction = float(self.pde.reaction_coeff)
            xgrid = self.pde.x_j
            ygrid = self.pde.y_j
            fgrid = self.pde.f_j
            def scalar_u(coord):
                return model_forward(params, geom, coord[None, :], self.act_name)[0, 0]
            hess = jax_vmap(jax.hessian(scalar_u))(f_coords)
            u = model_forward(params, geom, f_coords, self.act_name)[:, 0]
            f_val = jax_interp2d_bilinear(f_coords[:, 0], f_coords[:, 1], xgrid, ygrid, fgrid)[:, 0]
            return (hess[:, 0, 0] + hess[:, 1, 1] + reaction * u - f_val).reshape(-1, 1)
        if pde_name == "navier_stokes_2d":
            nu = float(self.pde.nu)
            def uvp_single(coord):
                return model_forward(params, geom, coord[None, :], self.act_name)[0, :3]
            def ns_res_single(coord):
                out = uvp_single(coord)
                jac = jax.jacfwd(uvp_single)(coord)
                hess_u = jax.hessian(lambda z: uvp_single(z)[0])(coord)
                hess_v = jax.hessian(lambda z: uvp_single(z)[1])(coord)
                u = out[0]
                v = out[1]
                u_t = jac[0, 0]
                u_x = jac[0, 1]
                u_y = jac[0, 2]
                v_t = jac[1, 0]
                v_x = jac[1, 1]
                v_y = jac[1, 2]
                p_x = jac[2, 1]
                p_y = jac[2, 2]
                u_xx = hess_u[1, 1]
                u_yy = hess_u[2, 2]
                v_xx = hess_v[1, 1]
                v_yy = hess_v[2, 2]
                f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) + p_x
                f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + p_y
                f_div = u_x + v_y
                return jnp.stack([f_u, f_v, f_div], axis=0)
            return jax_vmap(ns_res_single)(f_coords)
        raise ValueError(f"Unsupported PDE: {pde_name}")

    def data_loss_and_residuals(self, params, geom: Dict[str, Array], data_coords: Array, data_vals: Array, data_mask: Array):
        if data_coords.shape[0] == 0:
            pred = jnp.zeros((0, self.d_out), dtype=jnp.float64)
            diff = pred
            mse_u = jnp.array(0.0, dtype=jnp.float64)
        else:
            pred = model_forward(params, geom, data_coords, self.act_name)
            diff = (pred - data_vals) * data_mask
            denom = jnp.maximum(jnp.sum(data_mask), 1.0)
            mse_u = jnp.sum(diff * diff) / denom
        return mse_u, diff

    def theta_objective(self, params, geom: Dict[str, Array], data_coords: Array, data_vals: Array, data_mask: Array,
                        f_coords: Array, proposal_density: Optional[Array] = None):
        mse_u, _ = self.data_loss_and_residuals(params, geom, data_coords, data_vals, data_mask)
        if f_coords.shape[0] == 0:
            mse_f = jnp.array(0.0, dtype=jnp.float64)
        else:
            r = self.residual_tensor(params, geom, f_coords)
            r2 = jnp.sum(r * r, axis=1)
            if proposal_density is None:
                mse_f = jnp.mean(r2)
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                mse_f = jnp.mean(r2 / (pbar * self.admissible_volume + 1e-12))
        return mse_u + mse_f

    def eq10_entropy_term(self, geom: Dict[str, Array], f_coords: Array) -> Array:
        p = continuous_density(geom, f_coords, self.norm_points, self.admissible_volume)
        return -float(self.cfg.beta) * jnp.mean(jnp.log(p))

    def fixed_batch_metrics(self, params, geom: Dict[str, Array], data_coords: Array, data_vals: Array, data_mask: Array, f_coords: Array, proposal_density: Optional[Array]) -> Dict[str, float]:
        mse_u, _ = self.data_loss_and_residuals(params, geom, data_coords, data_vals, data_mask)
        if f_coords.shape[0] == 0:
            mse_f_weighted = jnp.array(0.0, dtype=jnp.float64)
            tail_loss = jnp.array(0.0, dtype=jnp.float64)
        else:
            r = self.residual_tensor(params, geom, f_coords)
            r2 = jnp.sum(r * r, axis=1)
            if proposal_density is None:
                mse_f_weighted = jnp.mean(r2)
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                mse_f_weighted = jnp.mean(r2 / (pbar * self.admissible_volume + 1e-12))
            thr = jnp.quantile(r2, 0.99)
            tail_loss = jnp.mean(r2[r2 >= thr])
        theta_obj = mse_u + mse_f_weighted
        return {
            "theta_obj": float(theta_obj),
            "loss_total": float(theta_obj),
            "mse_u": float(mse_u),
            "mse_f_weighted": float(mse_f_weighted),
            "tail_loss": float(tail_loss),
        }

    def global_residual_vector(self, params, geom: Dict[str, Array], data_coords: Array, data_vals: Array, data_mask: Array,
                              f_coords: Array, proposal_density: Optional[Array] = None) -> Array:
        parts = []
        if data_coords.shape[0] > 0:
            pred = model_forward(params, geom, data_coords, self.act_name)
            diff = (pred - data_vals) * data_mask
            denom_u = jnp.sqrt(jnp.maximum(jnp.sum(data_mask), 1.0))
            parts.append(jnp.reshape(diff / denom_u, (-1,)))
        if f_coords.shape[0] > 0:
            r = self.residual_tensor(params, geom, f_coords)
            denom_f = jnp.sqrt(jnp.maximum(float(r.shape[0]), 1.0))
            if proposal_density is None:
                r_scaled = r / denom_f
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                iw_sqrt = jnp.sqrt(1.0 / (pbar * self.admissible_volume + 1e-12)).reshape(-1, 1)
                r_scaled = r * iw_sqrt / denom_f
            parts.append(jnp.reshape(r_scaled, (-1,)))
        if len(parts) == 0:
            return jnp.zeros((1,), dtype=jnp.float64)
        return jnp.concatenate(parts, axis=0)

    # -------------------------
    # batching and sampling
    # -------------------------
    def sample_data_batch(self) -> Tuple[NpArray, NpArray, NpArray]:
        N = self.data_coords_full.shape[0]
        if self.cfg.batch_u <= 0 or self.cfg.batch_u >= N:
            return self.data_coords_full, self.data_vals_full, self.data_mask_full
        idx = self.rng.permutation(N)[: self.cfg.batch_u]
        return self.data_coords_full[idx], self.data_vals_full[idx], self.data_mask_full[idx]

    def _sample_uniform_in_ball(self, center: NpArray, radius: float, n: int) -> NpArray:
        d = center.size
        dirs = self.rng.normal(size=(n, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        rr = self.rng.random((n, 1)) ** (1.0 / d)
        return center.reshape(1, -1) + radius * rr * dirs

    def _admissible_mask_np(self, x: NpArray) -> NpArray:
        mask = np.ones((x.shape[0],), dtype=bool)
        for d, (lo, hi) in enumerate(self.bounds):
            mask &= (x[:, d] >= lo) & (x[:, d] <= hi)
        if self.pde.name == 'navier_stokes_2d':
            mask &= self.pde.filter_out_cylinder(x)
        return mask

    def _sample_single_component(self, geom_np: Dict[str, NpArray], j: int, n: int) -> NpArray:
        if n <= 0:
            return np.empty((0, self.d_in), dtype=np.float64)
        center = geom_np["centers"][j]
        radius = float(geom_np["radii"][j])
        accepted = []
        remaining = int(n)
        tries = 0
        max_tries = 200
        while remaining > 0:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"Continuous p_phi sampling failed for component {j}.")
            m = max(256, 6 * remaining)
            cand = self._sample_uniform_in_ball(center, radius, m)
            mask_dom = self._admissible_mask_np(cand)
            phj = np.asarray(phi_basis(jnp.asarray(cand), jnp.asarray(geom_np["centers"]), jnp.asarray(geom_np["radii"])))[:, j]
            keep = mask_dom & (self.rng.random(m) < phj)
            pts = cand[keep]
            take = min(remaining, pts.shape[0])
            if take > 0:
                accepted.append(pts[:take])
                remaining -= take
        out = np.concatenate(accepted, axis=0)
        perm = self.rng.permutation(out.shape[0])
        return out[perm]

    def sample_residual_batch(self, geom: Dict[str, Array]) -> ResidualBatch:
        b = int(self.cfg.batch_f if self.cfg.batch_f > 0 else self.cfg.n_r_pool)
        comp_idx = self.rng.integers(0, self.cfg.n_balls, size=(b,))
        counts = np.bincount(comp_idx, minlength=self.cfg.n_balls)
        geom_np = {"centers": np.asarray(geom["centers"]), "radii": np.asarray(geom["radii"])}
        pts_list = []
        for j in range(self.cfg.n_balls):
            cj = int(counts[j])
            if cj > 0:
                pts_list.append(self._sample_single_component(geom_np, j, cj))
        coords = np.concatenate(pts_list, axis=0)
        coords = coords[self.rng.permutation(coords.shape[0])]
        pbar = np.asarray(continuous_density(geom, jnp.asarray(coords), self.norm_points, self.admissible_volume))
        return ResidualBatch(coords=coords, proposal_density=pbar)

    # -------------------------
    # evaluation and plots
    # -------------------------
    def save_best_snapshot_plot(self, out_dir: Path, snapshot: Dict[str, Any], geom: Dict[str, Array],
                                rel_l2_epoch: float, it: int, loss_total: float):
        coord_kind = snapshot["coord_kind"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if coord_kind in ("tx", "xy"):
            pred = snapshot["pred_grid"]
            true = snapshot["true_grid"]
            coords = snapshot["grid_coords"]

            cover = self.cover_sum_batched(geom, coords, self.cfg.test_batch_size)
            H = snapshot["horizontal"].size
            V = snapshot["vertical"].size
            cover_grid = cover.reshape(H, V)

            err = (pred - true) ** 2
            extent = [
                float(snapshot["horizontal"].min()),
                float(snapshot["horizontal"].max()),
                float(snapshot["vertical"].min()),
                float(snapshot["vertical"].max()),
            ]

            # 디버그: 실제 값 범위 확인
            pred_vmin = float(np.min(pred))
            pred_vmax = float(np.max(pred))
            true_vmin = float(np.min(true))
            true_vmax = float(np.max(true))

            log(
                f"[PLOT-DEBUG] pde={self.pde.name} iter={it} "
                f"pred_min={pred_vmin:.6e} pred_max={pred_vmax:.6e} "
                f"true_min={true_vmin:.6e} true_max={true_vmax:.6e}"
            )

            fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)

            # Pred는 pred 범위로
            im0 = axes[0].imshow(
                pred.T,
                origin="lower",
                extent=extent,
                aspect=snapshot["aspect"],
                vmin=pred_vmin,
                vmax=pred_vmax,
            )
            self._draw_dd_balls_overlay(axes[0], geom, extent)
            axes[0].set_title("Predicted Solution + DD balls")
            axes[0].set_xlabel(snapshot["h_label"])
            axes[0].set_ylabel(snapshot["v_label"])
            fig.colorbar(im0, ax=axes[0], fraction=0.046)

            # True는 true 범위로
            im1 = axes[1].imshow(
                true.T,
                origin="lower",
                extent=extent,
                aspect=snapshot["aspect"],
                vmin=true_vmin,
                vmax=true_vmax,
            )
            axes[1].set_title("True Solution")
            axes[1].set_xlabel(snapshot["h_label"])
            axes[1].set_ylabel(snapshot["v_label"])
            fig.colorbar(im1, ax=axes[1], fraction=0.046)

            im2 = axes[2].imshow(
                err.T,
                origin="lower",
                extent=extent,
                aspect=snapshot["aspect"],
            )
            axes[2].set_title(r"$|u^* - \hat{u}|^2$")
            axes[2].set_xlabel(snapshot["h_label"])
            axes[2].set_ylabel(snapshot["v_label"])
            fig.colorbar(im2, ax=axes[2], fraction=0.046)

            im3 = axes[3].imshow(
                cover_grid.T,
                origin="lower",
                extent=extent,
                aspect=snapshot["aspect"],
            )
            self._draw_dd_balls_overlay(axes[3], geom, extent)
            axes[3].set_title(r"cover_sum = $\sum_j \phi_j$")
            axes[3].set_xlabel(snapshot["h_label"])
            axes[3].set_ylabel(snapshot["v_label"])
            fig.colorbar(im3, ax=axes[3], fraction=0.046)

            fig.suptitle(
                f"BEST-relL2 | pde={self.pde.name} | iter={it} | "
                f"relL2={rel_l2_epoch:.3e} | loss_total={loss_total:.3e}"
            )
            out_path = out_dir / f"best_relL2_snapshot_{self.pde.name}_iter{it:06d}.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            log(f"[BEST-relL2-PLOT] saved -> {out_path}")

        elif coord_kind == "txy":
            pred = snapshot["pred_grid"]
            true = snapshot["true_grid"]
            x = snapshot["horizontal"]
            y = snapshot["vertical"]
            t_plot = snapshot["t_plot"]
            extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
            titles = ["u", "v", "p"]

            fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5), constrained_layout=True)
            for row in range(3):
                pred_vmin = float(np.min(pred[:, :, row]))
                pred_vmax = float(np.max(pred[:, :, row]))
                true_vmin = float(np.min(true[:, :, row]))
                true_vmax = float(np.max(true[:, :, row]))

                log(
                    f"[PLOT-DEBUG] pde={self.pde.name} iter={it} field={titles[row]} "
                    f"pred_min={pred_vmin:.6e} pred_max={pred_vmax:.6e} "
                    f"true_min={true_vmin:.6e} true_max={true_vmax:.6e}"
                )

                p_im = axes[row, 0].imshow(
                    pred[:, :, row].T,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    vmin=pred_vmin,
                    vmax=pred_vmax,
                )
                self._draw_dd_balls_overlay(axes[row, 0], geom, extent)
                axes[row, 0].set_title(f"Pred {titles[row]}")
                fig.colorbar(p_im, ax=axes[row, 0], fraction=0.046)

                t_im = axes[row, 1].imshow(
                    true[:, :, row].T,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    vmin=true_vmin,
                    vmax=true_vmax,
                )
                axes[row, 1].set_title(f"True {titles[row]}")
                fig.colorbar(t_im, ax=axes[row, 1], fraction=0.046)

                e_im = axes[row, 2].imshow(
                    np.abs(pred[:, :, row] - true[:, :, row]).T,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                )
                axes[row, 2].set_title(f"Abs error {titles[row]}")
                fig.colorbar(e_im, ax=axes[row, 2], fraction=0.046)

                for col in range(3):
                    axes[row, col].set_xlabel("x")
                    axes[row, col].set_ylabel("y")

            fig.suptitle(
                f"BEST-relL2 | pde={self.pde.name} | iter={it} | "
                f"relL2={rel_l2_epoch:.3e} | t={t_plot:.3f}"
            )
            out_path = out_dir / f"best_relL2_snapshot_{self.pde.name}_iter{it:06d}.png"
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            log(f"[BEST-relL2-PLOT] saved -> {out_path}")

    def _draw_dd_balls_overlay(self, ax, geom: Dict[str, Array], extent=None):
        centers = np.asarray(geom["centers"])
        radii = np.asarray(geom["radii"])
        if centers.shape[1] < 2:
            return
        ax.scatter(centers[:, 0], centers[:, 1], s=12, c="white", edgecolors="white", linewidths=0.4, zorder=4, clip_on=True)
        for j in range(centers.shape[0]):
            circ = plt.Circle((float(centers[j, 0]), float(centers[j, 1])), float(radii[j]), fill=False, color="white", linewidth=0.7, alpha=0.9, clip_on=True)
            circ.set_clip_path(ax.patch)
            ax.add_patch(circ)
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.margins(x=0.0, y=0.0)

    def save_final_solution_plot(self, out_dir: Path, params, geom: Dict[str, Array]):
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.pde.coord_kind == "tx":
            nt, nx = 51, 256
            t = np.linspace(self.bounds[0][0], self.bounds[0][1], nt)
            x = np.linspace(self.bounds[1][0], self.bounds[1][1], nx)
            T, X = meshgrid_ij_np(t, x)
            coords = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
            pred = self.predict_batched(params, geom, coords, self.cfg.test_batch_size).reshape(nt, nx, self.d_out)
            out_path = out_dir / f"{self.pde.name}.png"
            plt.figure(figsize=(7, 4.5))
            plt.imshow(pred[..., 0], aspect="auto", origin="lower", extent=[x.min(), x.max(), t.min(), t.max()])
            plt.xlabel("x")
            plt.ylabel("t")
            plt.title(f"PINN-Balls Solution: {self.pde.name}")
            plt.colorbar(label="u")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            log(f"Saved plot: {out_path}")
        elif self.pde.coord_kind == "xy":
            nx, ny = 256, 256
            x = np.linspace(self.bounds[0][0], self.bounds[0][1], nx)
            y = np.linspace(self.bounds[1][0], self.bounds[1][1], ny)
            X, Y = meshgrid_ij_np(x, y)
            coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
            pred = self.predict_batched(params, geom, coords, self.cfg.test_batch_size).reshape(nx, ny, self.d_out)
            out_path = out_dir / f"{self.pde.name}.png"
            plt.figure(figsize=(6.6, 5.5))
            plt.imshow(pred[..., 0].T, origin="lower", aspect="equal", extent=[x.min(), x.max(), y.min(), y.max()])
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"PINN-Balls Solution: {self.pde.name}")
            plt.colorbar(label="u")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            log(f"Saved plot: {out_path}")
        elif self.pde.coord_kind == "txy":
            nx, ny = 256, 128
            t_plot = 0.5 * (self.bounds[0][0] + self.bounds[0][1])
            x = np.linspace(self.bounds[1][0], self.bounds[1][1], nx)
            y = np.linspace(self.bounds[2][0], self.bounds[2][1], ny)
            X, Y = meshgrid_ij_np(x, y)
            T = np.full_like(X, t_plot)
            coords = np.stack([T.reshape(-1), X.reshape(-1), Y.reshape(-1)], axis=1)
            pred = self.predict_batched(params, geom, coords, self.cfg.test_batch_size).reshape(nx, ny, self.d_out)
            out_path = out_dir / f"{self.pde.name}.png"
            fig = plt.figure(figsize=(11, 3.6))
            names = ["u", "v", "p"]
            for k in range(3):
                ax = fig.add_subplot(1, 3, k + 1)
                im = ax.imshow(pred[..., k].T, origin="lower", aspect="auto", extent=[x.min(), x.max(), y.min(), y.max()])
                ax.set_title(f"{names[k]} @ t={t_plot:g}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(im, ax=ax, fraction=0.046)
            plt.suptitle(f"PINN-Balls Solution: {self.pde.name}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close(fig)
            log(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------
# Anisotropic geometry helpers
# ---------------------------------------------------------------------
def mahalanobis_d2(x: Array, centers: Array, G_mats: Array) -> Array:
    diff = x[:, None, :] - centers[None, :, :]
    return jnp.einsum("nmd,mde,nme->nm", diff, G_mats, diff)


def anisotropic_phi_basis_geom(x: Array, geom: Dict[str, Array], eps: float = 1e-12) -> Array:
    d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
    s = 1.0 - d2 / (geom["radii"][None, :] ** 2 + eps)
    return jnp.maximum(s, 0.0) ** 2


def robust_lambdas_from_phi(ph: Array, d2: Array, eps: float = 1e-12) -> Array:
    s = jnp.sum(ph, axis=1, keepdims=True)
    lam_soft = ph / (s + eps)
    nearest = jnp.argmin(d2, axis=1)
    lam_hard = jax.nn.one_hot(nearest, ph.shape[1], dtype=ph.dtype)
    return jnp.where(s > eps, lam_soft, lam_hard)


def local_coords_anisotropic(x: Array, geom: Dict[str, Array], eps: float = 1e-12) -> Array:
    diff = x[:, None, :] - geom["centers"][None, :, :]
    L = jnp.linalg.cholesky(geom["G_mats"])
    z = jnp.einsum("mde,nme->nmd", L, diff)
    return z / (geom["radii"][None, :, None] + eps)


def model_forward_anisotropic(
    params: Sequence[Tuple[Tuple[Array, Array], ...]],
    geom: Dict[str, Array],
    x: Array,
    act_name: str,
) -> Array:
    ph = anisotropic_phi_basis_geom(x, geom)
    d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
    lam = robust_lambdas_from_phi(ph, d2)
    z = local_coords_anisotropic(x, geom)
    outs = [base.expert_apply(params[j], z[:, j, :], act_name) for j in range(len(params))]
    Y = jnp.stack(outs, axis=1)
    return jnp.sum(lam[:, :, None] * Y, axis=1)


# Re-route all inherited forward/residual code through the anisotropic DD.
base.model_forward = model_forward_anisotropic


@dataclass
class DiffusionGeometryConfig(base.TrainConfig):
    # theta optimizer selectable at runtime (HF = J^T J matrix-free Gauss-Newton on JAX GPU)
    theta_optimizer: str = ""

    # Hessian-free / Truncated Newton
    hf_damping_init: float = ""
    hf_damping_min: float = ""
    hf_damping_up: float = ""
    hf_damping_down: float = ""
    hf_cg_tol: float = ""
    hf_cg_maxiter: int = ""
    hf_max_trials: int = ""
    hf_line_search_c1: float = ""
    hf_line_search_tau: float = ""
    hf_line_search_maxiter: int = ""
    hf_step_norm_cap: float = ""

    # blockwise theta refinement to reduce moving-target oscillation
    theta_inner_steps: int = ""
    refresh_every: int = ""
    late_stage_refresh_rel_l2: float = ""
    late_stage_refresh_iter: int = ""

    # phi/sampling diagnostics
    phi_topk_quantile: float = ""
    phi_shock_band_halfwidth_scale: float = ""
    feedback_ridge_search_thresh: float = ""
    feedback_ridge_lock_thresh: float = ""
    feedback_shock_mass_low: float = ""
    feedback_shock_mass_high: float = ""
    feedback_shock_mass_lock_min: float = ""
    feedback_shock_mass_release: float = ""
    feedback_cover_halfwidth_lock_max: float = ""
    feedback_cover_halfwidth_release: float = ""
    feedback_ridge_width_mismatch_lock_max: float = ""
    feedback_ridge_width_mismatch_release: float = ""
    feedback_ema: float = ""
    feedback_shock_prior_mix_search: float = ""
    feedback_shock_prior_mix_overfocus: float = ""
    feedback_shock_prior_mix_micro_frozen: float = ""
    feedback_residual_focus_power_micro_frozen: float = ""

    # diffusion-based adaptive sampling
    diffusion_particles: int = ""
    diffusion_dt: float = ""
    diffusion_time_scale: float = ""
    diffusion_grid_n1: int = ""
    diffusion_grid_n2: int = ""
    diffusion_grid_n3: int = ""
    diffusion_grid_jitter: float = ""
    diffusion_steps: int = ""
    residual_focus_power: float = ""
    residual_focus_eps: float = ""
    residual_focus_smoothing_sigma: float = ""

    # anisotropic Voronoi / ellipsoidal DD update
    dd_lloyd_steps: int = ""
    dd_warmup_iters: int = ""
    dd_residual_kappa: float = ""
    dd_cov_eps: float = ""
    dd_eig_min: float = ""
    dd_eig_max: float = ""
    dd_cond_max: float = ""
    dd_axis_min_scale: float = ""
    dd_axis_max_scale: float = ""
    dd_radius_smooth_tau: float = ""
    dd_radius_growth_cap_rel: float = ""
    dd_radius_margin: float = ""
    dd_radius_quantile: float = ""
    dd_weight_power: float = ""
    dd_center_smooth_tau: float = ""
    dd_metric_smooth_tau: float = ""
    dd_cover_margin: float = ""
    dd_cover_mix_frac: float = ""
    dd_cover_weight: float = ""
    dd_empty_keep_old: bool = ""
    dd_weak_repair_margin: float = ""
    dd_weak_repair_max_passes: int = ""
    dd_late_shrink_gain: float = ""
    dd_micro_shrink_gain: float = ""

    # staged DD freeze for late-stage theta refinement
    dd_freeze_enable: bool = ""
    dd_semi_freeze_rel_l2: float = ""
    dd_semi_freeze_patience: int = ""
    dd_full_freeze_rel_l2: float = ""
    dd_full_freeze_patience: int = ""
    dd_micro_freeze_ridge_dx: float = ""
    dd_micro_freeze_min_iter: int = ""
    dd_semi_radius_tau: float = ""
    dd_semi_radius_growth_cap_rel: float = ""
    dd_freeze_improve_tol: float = ""
    dd_freeze_cond_thresh: float = ""
    dd_micro_center_tau: float = ""
    dd_micro_metric_tau: float = ""
    dd_micro_radius_tau: float = ""
    dd_micro_radius_growth_cap_rel: float = ""
    dd_micro_radius_quantile: float = ""
    dd_micro_axis_max_scale: float = ""
    dd_micro_update_every: int = ""

    # late shock-line refinement (objective/sampling rather than controller tuning)
    late_shock_rel_l2: float = ""
    late_shock_plateau_patience: int = ""
    late_shock_freeze_ridge_dx: float = ""
    late_shock_freeze_shock_mass_min: float = ""
    late_shock_freeze_peakr_min: float = ""
    late_shock_theta_inner_steps: int = ""
    late_shock_refresh_every: int = ""
    late_shock_residual_frac: float = ""
    late_shock_residual_min: int = ""
    late_shock_residual_max: int = ""
    late_shock_anchor_points: int = ""
    late_shock_residual_weight: float = ""
    late_shock_anchor_weight: float = ""
    late_shock_band_halfwidth_scale: float = ""
    late_shock_time_start_frac: float = ""
    late_shock_freeze_dd: bool = ""


class PINNFFusionSolver(base.PINNBallsSolver):
    """
    PINN BALLS skeleton
    + diffusion-based adaptive sampling
    + anisotropic / Voronoi DD geometry update
    + theta optimizer = Hessian-free / Truncated Newton

    Removed on purpose from the active runtime path:
      - LM / L-BFGS theta branches
      - explicit phi Adam-ascent path
      - cached block / refresh cadence
      - dense-probe hold logic
    """

    def __init__(self, pde: TorchPDEAdapter, cfg: DiffusionGeometryConfig, rng: np.random.Generator):
        self.pde = pde
        self.cfg = cfg
        self.rng = rng
        self.bounds = base.normalize_bounds(pde.bounds)
        self.d_in = pde.d_in
        self.d_out = pde.d_out
        self.residual_dim = pde.residual_dim
        self.act_name = cfg.act
        self.mins_np = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.maxs_np = np.array([b[1] for b in self.bounds], dtype=np.float64)
        self.diag = float(np.linalg.norm(self.maxs_np - self.mins_np))
        self.min_radius = float(cfg.dd_min_radius_scale) * self.diag
        self.mins = jnp.asarray(self.mins_np)
        self.maxs = jnp.asarray(self.maxs_np)

        self.data_coords_full, self.data_vals_full, self.data_mask_full = self.pde.sample_data(self.rng, cfg.n_ic, cfg.n_bc)
        self.norm_points = jnp.asarray(self.pde.sample_collocation(self.rng, cfg.normalizer_points))
        self.admissible_volume = self._estimate_admissible_volume(8192)
        self.dd_cover_points = self._initial_domain_cover_points()

        expert_in = self.d_in
        self.params = tuple(
            base.make_expert_params(self.rng, expert_in, self.d_out, cfg.layers, cfg.width)
            for _ in range(cfg.n_balls)
        )
        self.expert_metas = [base.ExpertMeta(p) for p in self.params]
        self.expert_offsets = np.cumsum([0] + [m.size for m in self.expert_metas[:-1]]).astype(np.int64)
        self.theta_size = int(sum(m.size for m in self.expert_metas))

        self.geom_state = self._init_geometry_from_kmeans()
        self.jax_device = require_jax_gpu()
        self.norm_points = jax.device_put(self.norm_points, self.jax_device)
        self.params = jax.device_put(self.params, self.jax_device)
        self.geom_state = jax.device_put(self.geom_state, self.jax_device)

        self.grid_axes = self._make_sampling_axes()
        self.grid_shape = tuple(len(ax) for ax in self.grid_axes)
        self.grid_coords = self._grid_coords_from_axes(self.grid_axes)
        self.grid_valid_mask = self._admissible_mask_np(self.grid_coords).reshape(self.grid_shape)
        self.fourier_cache = self._prepare_fourier_cache() if self.d_in <= 2 else None

        self.p_uniform_grid = self._uniform_grid_density()
        self.p_grid = self.p_uniform_grid.copy()
        self.p_particles = self._sample_from_grid_density(self.p_grid, int(self.cfg.diffusion_particles))
        self.current_outer_iter = 0

        self.hf_damping = float(cfg.hf_damping_init)

        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.best_params = self.params
        self.best_geom_state = self.geom_state
        self.best_eval_snapshot: Optional[Dict[str, Any]] = None
        self.latest_relL2: float = float("inf")
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_sampling_info: Dict[str, Any] = {}
        self.last_dd_info: Dict[str, Any] = {}
        self.last_refresh_info: Dict[str, Any] = {"refreshed": 1, "block_pos": 1, "block_len": 1}
        self.dd_freeze_mode: str = "active"
        self.dd_no_improve_streak: int = 0
        self.dd_best_rel_l2_seen: float = float("inf")
        self.cached_block: Optional[Dict[str, Any]] = None
        self.feedback_ctrl: Dict[str, Any] = {}
        self.block_theta_step = 0
        self.last_micro_dd_update_iter: int = -10**9
        self.last_dd_update_iter: int = -10**9

        # Learning-based DD hardening: no manual stage schedule.
        # The geometry is progressively slowed down only when the observed
        # drift becomes small and the residual ridge / cover alignment locks in.
        self.dd_hardening_factor: float = 1.0
        self.dd_hardening_state: str = "adaptive"
        self.dd_lock_streak: int = 0
        self.dd_release_streak: int = 0
        self.dd_drift_ema: float = float("inf")
        self.dd_ridge_ema: float = float("inf")

        # Sampling state used by both theta and DD.
        self.p_explicit_tau_grid = self.p_uniform_grid.copy()
        self.p_particle_density = self.p_uniform_grid.copy()

        self.update_sampling_distribution(self.params, self.geom_state, 0)

    # ---------------------------------------------------------
    # geometry init / accessors
    # ---------------------------------------------------------
    def geom(self, _unused: Optional[Dict[str, Array]] = None) -> Dict[str, Array]:
        return self.geom_state

    def _identity_G(self) -> np.ndarray:
        return np.tile(np.eye(self.d_in, dtype=np.float64)[None, :, :], (self.cfg.n_balls, 1, 1))

    def _initial_domain_cover_points(self) -> np.ndarray:
        """
        Build a dense cover set over the full admissible domain for initialization.

        The goal here is not merely to cover the training / collocation samples,
        but to force the initial ellipses to span the whole computational domain
        before any adaptive sampling / DD specialization happens.
        """
        if self.d_in == 1:
            counts = [max(int(getattr(self.cfg, 'diffusion_grid_n1', 96)), 257)]
        elif self.d_in == 2:
            counts = [
                max(int(getattr(self.cfg, 'diffusion_grid_n1', 96)), 96),
                max(int(getattr(self.cfg, 'diffusion_grid_n2', 192)), 192),
            ]
        elif self.d_in == 3:
            counts = [
                max(int(getattr(self.cfg, 'diffusion_grid_n1', 24)), 24),
                max(int(getattr(self.cfg, 'diffusion_grid_n2', 48)), 48),
                max(int(getattr(self.cfg, 'diffusion_grid_n3', 24)), 24),
            ]
            max_total = 20000
            total = counts[0] * counts[1] * counts[2]
            if total > max_total:
                scale = (max_total / float(total)) ** (1.0 / 3.0)
                counts = [max(8, int(round(c * scale))) for c in counts]
        else:
            counts = [max(32, int(round(20000 ** (1.0 / max(self.d_in, 1))))) for _ in range(self.d_in)]

        axes = []
        for n, (lo, hi) in zip(counts, self.bounds):
            axes.append(np.linspace(lo, hi, max(int(n), 2), dtype=np.float64))
        mesh = np.meshgrid(*axes, indexing='ij')
        pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
        mask = self._admissible_mask_np(pts)
        pts = pts[mask]

        # If admissible filtering leaves nothing (should not happen), fall back
        # to a dense box sample so initialization does not crash.
        if pts.shape[0] == 0:
            pts = self._sample_uniform_admissible(4096)
        return pts

    def _mix_dd_points(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Runtime full-domain cover mixing is disabled. Domain-cover points are
        # used for initialization and monitoring only, not as a persistent DD anchor.
        pts = np.asarray(pts, dtype=np.float64)
        return pts, np.ones((pts.shape[0],), dtype=np.float64)

    def _cover_sum_np(self, pts: np.ndarray, centers: np.ndarray, radii: np.ndarray, G_mats: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        diff = pts[:, None, :] - centers[None, :, :]
        d2 = np.einsum("nmd,mde,nme->nm", diff, G_mats, diff)
        s = 1.0 - d2 / (radii[None, :] ** 2 + 1e-12)
        ph = np.maximum(s, 0.0) ** 2
        return np.sum(ph, axis=1)

    def _coverage_stats_np(self, centers: np.ndarray, radii: np.ndarray, G_mats: np.ndarray, pts: Optional[np.ndarray] = None) -> Dict[str, float]:
        if pts is None:
            pts = self.dd_cover_points
        if pts is None or pts.shape[0] == 0:
            return {"cover_sum_min": float("nan"), "uncovered_frac": float("nan"), "n_probe": 0}
        cover_sum = self._cover_sum_np(np.asarray(pts, dtype=np.float64), centers, radii, G_mats)
        uncovered = cover_sum <= 0.0
        return {
            "cover_sum_min": float(np.min(cover_sum)) if cover_sum.size > 0 else float("nan"),
            "uncovered_frac": float(np.mean(uncovered)) if cover_sum.size > 0 else float("nan"),
            "n_probe": int(pts.shape[0]),
        }

    def _weak_cover_repair(self, centers: np.ndarray, radii: np.ndarray, G_mats: np.ndarray, pts: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        if pts is None:
            pts = self.dd_cover_points
        if pts is None or pts.shape[0] == 0:
            return np.asarray(radii, dtype=np.float64).copy(), {"repair_passes": 0, "repair_expansions": 0, "repair_exact": 0}
        pts = np.asarray(pts, dtype=np.float64)
        out = np.asarray(radii, dtype=np.float64).copy()
        margin = float(getattr(self.cfg, "dd_weak_repair_margin", 1.01))

        cover_sum = self._cover_sum_np(pts, centers, out, G_mats)
        uncovered = cover_sum <= 0.0
        if not np.any(uncovered):
            return out, {"repair_passes": 0, "repair_expansions": 0, "repair_exact": 0}

        # One-shot exact minimal repair: assign each uncovered probe to its nearest
        # current ball and enlarge only that ball just enough to cover the farthest
        # uncovered probe assigned to it.
        assign, d2 = self._assignment(pts, {"centers": centers, "radii": out, "G_mats": G_mats})
        total_expansions = 0
        for j in range(self.cfg.n_balls):
            mask = uncovered & (assign == j)
            if np.any(mask):
                req = margin * float(np.max(np.sqrt(np.maximum(d2[mask, j], 0.0))))
                if req > out[j] + 1e-15:
                    out[j] = max(req, self.min_radius)
                    total_expansions += 1

        return out, {"repair_passes": 1, "repair_expansions": int(total_expansions), "repair_exact": 1}

    def _init_geometry_from_kmeans(self) -> Dict[str, Array]:
        pts = self.pde.sample_collocation(self.rng, max(self.cfg.dd_init_kmeans_points, 16 * self.cfg.n_balls))
        if pts.shape[0] > self.cfg.dd_init_kmeans_points:
            sel = self.rng.choice(pts.shape[0], size=self.cfg.dd_init_kmeans_points, replace=False)
            pts = pts[sel]

        # Initial ellipses must cover the whole admissible domain, not only the
        # currently sampled training/collocation set.  We therefore build the
        # initial radius from a dense full-domain cover set.
        cover_pts = self._initial_domain_cover_points()
        if self.data_coords_full is not None and self.data_coords_full.shape[0] > 0:
            cover_pts = np.concatenate([cover_pts, self.data_coords_full], axis=0)

        N = pts.shape[0]
        M = min(self.cfg.n_balls, N)
        idx = self.rng.permutation(N)[:M]
        centers = pts[idx].copy()

        for _ in range(max(1, int(self.cfg.dd_kmeans_iters))):
            dist2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            assign = np.argmin(dist2, axis=1)
            new_centers = []
            for j in range(M):
                mask = assign == j
                if np.any(mask):
                    new_centers.append(pts[mask].mean(axis=0))
                else:
                    ridx = self.rng.integers(0, N)
                    new_centers.append(pts[ridx])
            centers = np.stack(new_centers, axis=0)

        cover_dist2 = np.sum((cover_pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        cover_assign = np.argmin(cover_dist2, axis=1)
        radii = np.full((M,), self.min_radius, dtype=np.float64)
        for j in range(M):
            mask = cover_assign == j
            if np.any(mask):
                d = np.sqrt(np.sum((cover_pts[mask] - centers[j]) ** 2, axis=1) + 1e-12)
                radii[j] = max(float(np.max(d)), self.min_radius)

        if M < self.cfg.n_balls:
            extra = self.cfg.n_balls - M
            centers_extra = pts[self.rng.integers(0, N, size=extra)]
            radii_extra = np.full((extra,), self.min_radius, dtype=np.float64)
            centers = np.concatenate([centers, centers_extra], axis=0)
            radii = np.concatenate([radii, radii_extra], axis=0)

        G_mats = self._identity_G()
        radii = self._clamp_radii_by_axes(radii, G_mats)
        return {
            "centers": jnp.asarray(centers),
            "radii": jnp.asarray(radii),
            "G_mats": jnp.asarray(G_mats),
        }

    # ---------------------------------------------------------
    # diffusion-based adaptive sampling
    # ---------------------------------------------------------
    def _make_sampling_axes(self) -> Tuple[np.ndarray, ...]:
        if self.d_in == 1:
            counts = [int(self.cfg.diffusion_grid_n1)]
        elif self.d_in == 2:
            counts = [int(self.cfg.diffusion_grid_n1), int(self.cfg.diffusion_grid_n2)]
        elif self.d_in == 3:
            counts = [int(self.cfg.diffusion_grid_n1), int(self.cfg.diffusion_grid_n2), int(self.cfg.diffusion_grid_n3)]
        else:
            raise ValueError(f"Unsupported input dimension for diffusion grid: d_in={self.d_in}")
        axes = []
        for n, (lo, hi) in zip(counts, self.bounds):
            axes.append(np.linspace(lo, hi, max(int(n), 2), dtype=np.float64))
        return tuple(axes)

    def _grid_coords_from_axes(self, axes: Tuple[np.ndarray, ...]) -> np.ndarray:
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], axis=1)

    def _uniform_grid_density(self) -> np.ndarray:
        p = np.zeros(self.grid_shape, dtype=np.float64)
        p[self.grid_valid_mask] = 1.0
        return p / np.maximum(p.sum(), 1e-12)

    def _grid_steps(self) -> np.ndarray:
        hs = []
        for ax, (lo, hi) in zip(self.grid_axes, self.bounds):
            if len(ax) <= 1:
                hs.append(float(hi - lo))
            else:
                hs.append(float((hi - lo) / max(len(ax) - 1, 1)))
        return np.asarray(hs, dtype=np.float64)

    def _normalize_grid_density(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float64).copy()
        q[~self.grid_valid_mask] = 0.0
        q = np.maximum(q, 0.0)
        s = float(q.sum())
        if (not np.isfinite(s)) or s <= 0.0:
            return self._uniform_grid_density()
        return q / s

    def _grid_cell_volume(self) -> float:
        return float(np.prod(self._grid_steps()))

    def _proposal_density_from_grid(self, pts: np.ndarray, p_grid: Optional[np.ndarray] = None) -> np.ndarray:
        if pts.shape[0] == 0:
            return np.empty((0, 1), dtype=np.float64)
        if p_grid is None:
            p_grid = self.p_grid
        interp = RegularGridInterpolator(self.grid_axes, self._normalize_grid_density(p_grid), bounds_error=False, fill_value=0.0)
        cell_mass = np.asarray(interp(pts), dtype=np.float64).reshape(-1)
        q = cell_mass / max(self._grid_cell_volume(), 1e-12)
        return np.maximum(q, 1e-12).reshape(-1, 1)

    def _sample_uniform_admissible(self, n: int) -> np.ndarray:
        need = int(n)
        chunks: List[np.ndarray] = []
        while need > 0:
            m = max(256, int(1.25 * need))
            cand = base.sample_box(self.rng, m, self.bounds, use_lhs=True)
            mask = self._admissible_mask_np(cand)
            cand = cand[mask]
            if cand.shape[0] == 0:
                continue
            take = min(need, cand.shape[0])
            chunks.append(cand[:take])
            need -= take
        return np.concatenate(chunks, axis=0)

    def _sample_from_grid_density(self, p_grid: np.ndarray, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, self.d_in), dtype=np.float64)
        p = self._normalize_grid_density(p_grid).reshape(-1)
        idx = self.rng.choice(p.size, size=n, replace=True, p=p)
        pts = self.grid_coords[idx].copy()
        jitter_scale = float(getattr(self.cfg, "diffusion_grid_jitter", 0.0))
        if jitter_scale > 0.0:
            hs = self._grid_steps()
            noise = self.rng.uniform(low=-0.5, high=0.5, size=pts.shape) * hs.reshape(1, -1) * jitter_scale
            pts = pts + noise
        return self._project_particles_to_domain(pts, None)

    def _histogram_edges_from_axes(self) -> Tuple[np.ndarray, ...]:
        edges = []
        for ax, (lo, hi) in zip(self.grid_axes, self.bounds):
            ax = np.asarray(ax, dtype=np.float64)
            if ax.size <= 1:
                h = max(float(hi - lo), 1e-12)
                edges.append(np.asarray([lo - 0.5 * h, hi + 0.5 * h], dtype=np.float64))
            else:
                mids = 0.5 * (ax[:-1] + ax[1:])
                first = ax[0] - 0.5 * (ax[1] - ax[0])
                last = ax[-1] + 0.5 * (ax[-1] - ax[-2])
                edges.append(np.concatenate([[first], mids, [last]], axis=0))
        return tuple(edges)

    def _density_from_particles(self, pts: np.ndarray, smoothing_sigma: float = 0.75) -> np.ndarray:
        if pts is None or pts.shape[0] == 0:
            return self._uniform_grid_density()
        pts = self._project_particles_to_domain(np.asarray(pts, dtype=np.float64), None)
        hist, _ = np.histogramdd(pts, bins=self._histogram_edges_from_axes())
        if smoothing_sigma > 0.0:
            hist = sp_ndimage.gaussian_filter(hist, sigma=float(smoothing_sigma), mode="nearest")
        hist[~self.grid_valid_mask] = 0.0
        return self._normalize_grid_density(hist)

    def _rel_gate(self) -> float:
        return float(min(getattr(self, "best_relL2", float("inf")), getattr(self, "latest_relL2", float("inf"))))

    def _particle_density_smoothing_sigma(self) -> float:
        rel_gate = self._rel_gate()
        if rel_gate <= 2e-2:
            return 0.25
        if rel_gate <= 3e-2:
            return 0.30
        if rel_gate <= 5e-2:
            return 0.40
        if rel_gate <= 1e-1:
            return 0.55
        return 0.75

    def _explicit_state_blend(self) -> float:
        rel_gate = self._rel_gate()
        if rel_gate <= 2e-2:
            return 0.02
        if rel_gate <= 3e-2:
            return 0.03
        if rel_gate <= 5e-2:
            return 0.06
        if rel_gate <= 1e-1:
            return 0.10
        return 0.20

    def _update_dd_hardening(self) -> None:
        dd = getattr(self, "last_dd_info", {})
        samp = getattr(self, "last_sampling_info", {})
        drift = float(max(dd.get("center_drift", float("nan")), dd.get("radius_drift", float("nan")), dd.get("metric_drift", float("nan"))))
        ridge = float(dd.get("ridge_dx", float("nan")))
        cover_halfwidth = float(samp.get("cover_halfwidth", float("nan")))
        ridge_width_mismatch = float(samp.get("ridge_width_mismatch", float("nan")))
        rel_gate = self._rel_gate()

        if np.isfinite(drift):
            if np.isfinite(self.dd_drift_ema):
                self.dd_drift_ema = 0.8 * self.dd_drift_ema + 0.2 * drift
            else:
                self.dd_drift_ema = drift
        if np.isfinite(ridge):
            if np.isfinite(self.dd_ridge_ema):
                self.dd_ridge_ema = 0.8 * self.dd_ridge_ema + 0.2 * ridge
            else:
                self.dd_ridge_ema = ridge

        ridge_lock = float(getattr(self.cfg, "feedback_ridge_lock_thresh", 0.04))
        ridge_search = float(getattr(self.cfg, "feedback_ridge_search_thresh", 0.10))
        cover_lock_max = float(getattr(self.cfg, "feedback_cover_halfwidth_lock_max", 0.03))
        cover_release = float(getattr(self.cfg, "feedback_cover_halfwidth_release", 0.05))
        width_lock_max = float(getattr(self.cfg, "feedback_ridge_width_mismatch_lock_max", 0.02))
        width_release = float(getattr(self.cfg, "feedback_ridge_width_mismatch_release", 0.04))

        if rel_gate <= 3e-2:
            ridge_hard_lock = max(ridge_lock, 0.08)
            ridge_release = max(ridge_search, 0.12)
            drift_lock = 0.015
            drift_release = 0.025
            lock_needed = 3
        elif rel_gate <= 5e-2:
            ridge_hard_lock = max(ridge_lock, 0.10)
            ridge_release = max(ridge_search, 0.14)
            drift_lock = 0.020
            drift_release = 0.030
            lock_needed = 3
        elif rel_gate <= 1e-1:
            ridge_hard_lock = max(ridge_lock, 0.16)
            ridge_release = max(ridge_search, 0.22)
            drift_lock = 0.030
            drift_release = 0.040
            lock_needed = 2
        else:
            ridge_hard_lock = max(ridge_lock, 0.10)
            ridge_release = ridge_search
            drift_lock = 0.03
            drift_release = 0.04
            lock_needed = 3

        stable_lock = (
            np.isfinite(ridge)
            and ridge <= ridge_hard_lock
            and np.isfinite(drift)
            and drift <= drift_lock
            and np.isfinite(cover_halfwidth)
            and cover_halfwidth <= cover_lock_max
            and np.isfinite(ridge_width_mismatch)
            and ridge_width_mismatch <= width_lock_max
        )
        unstable = (
            (np.isfinite(ridge) and ridge > ridge_release)
            or (np.isfinite(drift) and drift > drift_release)
            or (np.isfinite(cover_halfwidth) and cover_halfwidth > cover_release)
            or (np.isfinite(ridge_width_mismatch) and ridge_width_mismatch > width_release)
        )

        if stable_lock:
            self.dd_lock_streak += 1
        else:
            self.dd_lock_streak = 0

        if unstable:
            self.dd_release_streak += 1
        else:
            self.dd_release_streak = 0

        if self.dd_lock_streak >= lock_needed:
            if rel_gate <= 3e-2:
                shrink = 0.75
            elif rel_gate <= 5e-2:
                shrink = 0.80
            elif rel_gate <= 1e-1:
                shrink = 0.86
            else:
                shrink = 0.90
            self.dd_hardening_factor = max(0.12, shrink * self.dd_hardening_factor)
            self.dd_lock_streak = 0
        elif self.dd_release_streak >= 2:
            grow = 1.06 if rel_gate <= 5e-2 else 1.10
            self.dd_hardening_factor = min(1.0, grow * self.dd_hardening_factor)
            self.dd_release_streak = 0

        # Only cap hardening after the ridge center error is already reasonably controlled.
        if rel_gate <= 3e-2:
            if np.isfinite(ridge) and ridge <= 0.10:
                self.dd_hardening_factor = min(self.dd_hardening_factor, 0.35)
        elif rel_gate <= 5e-2:
            if np.isfinite(ridge) and ridge <= 0.12:
                self.dd_hardening_factor = min(self.dd_hardening_factor, 0.60)
        elif rel_gate <= 1e-1:
            if np.isfinite(ridge) and ridge <= 0.18:
                self.dd_hardening_factor = min(self.dd_hardening_factor, 0.80)

        if self.dd_hardening_factor <= 0.18:
            self.dd_hardening_state = "locked"
        elif self.dd_hardening_factor < 0.80:
            self.dd_hardening_state = "hardening"
        else:
            self.dd_hardening_state = "adaptive"

        if isinstance(self.last_dd_info, dict):
            self.last_dd_info["hardening_factor"] = float(self.dd_hardening_factor)
            self.last_dd_info["hardening_state"] = str(self.dd_hardening_state)
            self.last_dd_info["drift_ema"] = float(self.dd_drift_ema) if np.isfinite(self.dd_drift_ema) else float("nan")
            self.last_dd_info["ridge_ema"] = float(self.dd_ridge_ema) if np.isfinite(self.dd_ridge_ema) else float("nan")

    def _residual_energy_batched(self, params, geom: Dict[str, Array], coords_np: np.ndarray, batch_size: int) -> np.ndarray:
        outs = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jnp.asarray(coords_np[st:ed])
            r = self.residual_tensor(params, geom, xb)
            r2 = jnp.sum(r * r, axis=1)
            outs.append(np.asarray(r2))
        return np.concatenate(outs, axis=0)

    def _build_residual_focus_density(self, params, geom: Dict[str, Array], power_override: Optional[float] = None) -> np.ndarray:
        r2 = self._residual_energy_batched(params, geom, self.grid_coords, self.cfg.test_batch_size).reshape(self.grid_shape)
        r2[~self.grid_valid_mask] = 0.0
        self.last_residual_energy_grid = np.asarray(r2, dtype=np.float64)
        p0 = np.maximum(r2, 0.0) + float(self.cfg.residual_focus_eps)
        sigma = float(getattr(self.cfg, "residual_focus_smoothing_sigma", 0.0))
        if sigma > 0.0:
            p0 = sp_ndimage.gaussian_filter(p0, sigma=sigma, mode="nearest")
            p0[~self.grid_valid_mask] = 0.0
        power = float(self.cfg.residual_focus_power if power_override is None else power_override)
        if abs(power - 1.0) > 1e-15:
            p0 = p0 ** power
        p0 = self._normalize_grid_density(p0)
        self.last_residual_focus_density = np.asarray(p0, dtype=np.float64)
        return p0

    def _prepare_fourier_cache(self) -> Optional[Dict[str, Any]]:
        if self.d_in > 2:
            return None
        cache: Dict[str, Any] = {"axes": self.grid_axes, "ops": []}
        for d, ax in enumerate(self.grid_axes):
            N = len(ax)
            lo, hi = self.bounds[d]
            L = max(float(hi - lo), 1e-12)
            if N <= 1:
                C = np.ones((1, 1), dtype=np.float64)
                Cinv = np.ones((1, 1), dtype=np.float64)
                S = np.zeros((1, 1), dtype=np.float64)
                lam = np.zeros((1,), dtype=np.float64)
            else:
                xnorm = (ax - lo) / L
                n = np.arange(N, dtype=np.float64)
                C = np.cos(np.pi * xnorm[:, None] * n[None, :])
                Cinv = np.linalg.inv(C)
                S = -(np.pi / L) * np.sin(np.pi * xnorm[:, None] * n[None, :]) * n[None, :]
                lam = (np.pi * n / L) ** 2
            cache["ops"].append({"C": C, "Cinv": Cinv, "S": S, "lam": lam})
        return cache

    def _fourier_coeff_from_density(self, p0: np.ndarray):
        if self.d_in == 1:
            op = self.fourier_cache["ops"][0]
            return op["Cinv"] @ p0.reshape(-1)
        if self.d_in == 2:
            op0 = self.fourier_cache["ops"][0]
            op1 = self.fourier_cache["ops"][1]
            return op0["Cinv"] @ p0 @ op1["Cinv"].T
        raise ValueError("Fourier coefficients only supported for d_in <= 2")

    def _fourier_density_and_score_at_time(self, coeff0, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        tau = float(max(tau, 0.0))
        if self.d_in == 1:
            op = self.fourier_cache["ops"][0]
            coeff_t = coeff0 * np.exp(-tau * op["lam"])
            p = op["C"] @ coeff_t
            dp = op["S"] @ coeff_t
            p = np.maximum(p, 1e-14)
            p = p / np.maximum(p.sum(), 1e-12)
            score = (dp / p).reshape(-1, 1)
            return p.reshape(self.grid_shape), score.reshape(self.grid_shape + (1,))
        if self.d_in == 2:
            op0 = self.fourier_cache["ops"][0]
            op1 = self.fourier_cache["ops"][1]
            lam = op0["lam"][:, None] + op1["lam"][None, :]
            coeff_t = coeff0 * np.exp(-tau * lam)
            p = op0["C"] @ coeff_t @ op1["C"].T
            dp0 = op0["S"] @ coeff_t @ op1["C"].T
            dp1 = op0["C"] @ coeff_t @ op1["S"].T
            p = np.maximum(p, 1e-14)
            p = p / np.maximum(p.sum(), 1e-12)
            score = np.stack([dp0 / p, dp1 / p], axis=-1)
            return p.reshape(self.grid_shape), score.reshape(self.grid_shape + (2,))
        raise ValueError("Fourier density only supported for d_in <= 2")

    def _grid_heat_density_and_score_at_time(self, p0: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        hs = self._grid_steps()
        sigma_phys = math.sqrt(max(2.0 * float(tau), 1e-12))
        sigma_pix = [max(sigma_phys / max(float(h), 1e-12), 1e-12) for h in hs]
        p = sp_ndimage.gaussian_filter(p0, sigma=sigma_pix, mode="nearest")
        p = self._normalize_grid_density(p)
        logp = np.log(np.maximum(p, 1e-14))
        grads = np.gradient(logp, *hs, edge_order=2)
        score = np.stack(grads, axis=-1)
        return p, score

    def _interp_vector_field(self, vec_grid: np.ndarray, pts: np.ndarray) -> np.ndarray:
        vals = np.zeros((pts.shape[0], self.d_in), dtype=np.float64)
        for d in range(self.d_in):
            interp = RegularGridInterpolator(self.grid_axes, vec_grid[..., d], bounds_error=False, fill_value=None)
            vals[:, d] = interp(pts)
        return vals

    def _project_particles_to_domain(self, pts: np.ndarray, p_ref_grid: Optional[np.ndarray] = None) -> np.ndarray:
        out = np.asarray(pts, dtype=np.float64).copy()
        for d, (lo, hi) in enumerate(self.bounds):
            out[:, d] = np.clip(out[:, d], lo, hi)
        mask = self._admissible_mask_np(out)
        if not np.all(mask):
            if p_ref_grid is None:
                refill = self._sample_uniform_admissible(int((~mask).sum()))
            else:
                refill = self._sample_from_grid_density(p_ref_grid, int((~mask).sum()))
            out[~mask] = refill
        return out

    def _burgers_xonly_coeff_from_density(self, p0: np.ndarray) -> np.ndarray:
        # Burgers grid is (t, x). Apply the Neumann heat smoothing only along x,
        # independently for each fixed t-row.
        op_x = self.fourier_cache["ops"][1]
        return p0 @ op_x["Cinv"].T

    def _burgers_xonly_density_and_score_at_time(self, coeff0: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        tau = float(max(tau, 0.0))
        op_x = self.fourier_cache["ops"][1]
        coeff_t = coeff0 * np.exp(-tau * op_x["lam"][None, :])
        p = coeff_t @ op_x["C"].T
        dp_x = coeff_t @ op_x["S"].T
        p = np.maximum(p, 1e-14)
        p = p / np.maximum(p.sum(), 1e-12)
        score = np.zeros(p.shape + (2,), dtype=np.float64)
        score[..., 1] = dp_x / p
        return p.reshape(self.grid_shape), score.reshape(self.grid_shape + (2,))

    def _run_ddim_reverse(self, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        n_steps = max(int(getattr(self.cfg, "diffusion_steps", 1)), 1)
        total_time = float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt)
        taus = np.linspace(total_time, 0.0, n_steps + 1, dtype=np.float64)

        explicit_tau_grid = p0
        score_type = "grid_heat"

        if self.pde.name == "burgers_1d" and self.d_in == 2:
            coeff0 = self._burgers_xonly_coeff_from_density(p0)
            explicit_tau_grid, _ = self._burgers_xonly_density_and_score_at_time(coeff0, taus[0])
            score_type = "analytic_fourier_xonly"
            x = self._sample_from_grid_density(explicit_tau_grid, int(self.cfg.diffusion_particles))
            for m in range(n_steps):
                tau_cur = taus[m]
                tau_nxt = taus[m + 1]
                dtau = float(tau_cur - tau_nxt)
                p_tau, score_grid = self._burgers_xonly_density_and_score_at_time(coeff0, tau_cur)
                score = self._interp_vector_field(score_grid, x)
                x = x - dtau * score
                x = self._project_particles_to_domain(x, p_tau)
            particle_sigma = self._particle_density_smoothing_sigma()
            particle_density = self._density_from_particles(x, smoothing_sigma=particle_sigma)
            blend_explicit = self._explicit_state_blend()
            # In the late stage, rely more on the particle state and less on the
            # explicit heat density so that the sampler can collapse onto a thin shock layer.
            p_state = self._normalize_grid_density((1.0 - blend_explicit) * particle_density + blend_explicit * explicit_tau_grid)
            x_state = self._sample_from_grid_density(p_state, int(self.cfg.diffusion_particles))
            return p_state, x_state, {
                "explicit_tau_grid": explicit_tau_grid,
                "particle_density": particle_density,
                "state_blend_explicit": float(blend_explicit),
                "score_type": score_type,
            }

        if self.d_in <= 2:
            coeff0 = self._fourier_coeff_from_density(p0)
            explicit_tau_grid, _ = self._fourier_density_and_score_at_time(coeff0, taus[0])
            score_type = "analytic_fourier"
            x = self._sample_from_grid_density(explicit_tau_grid, int(self.cfg.diffusion_particles))
            for m in range(n_steps):
                tau_cur = taus[m]
                tau_nxt = taus[m + 1]
                dtau = float(tau_cur - tau_nxt)
                p_tau, score_grid = self._fourier_density_and_score_at_time(coeff0, tau_cur)
                score = self._interp_vector_field(score_grid, x)
                x = x - dtau * score
                x = self._project_particles_to_domain(x, p_tau)
            particle_density = self._density_from_particles(x, smoothing_sigma=self._particle_density_smoothing_sigma())
            p_state = self._normalize_grid_density(particle_density)
            x_state = self._sample_from_grid_density(p_state, int(self.cfg.diffusion_particles))
            return p_state, x_state, {
                "explicit_tau_grid": explicit_tau_grid,
                "particle_density": particle_density,
                "state_blend_explicit": 0.0,
                "score_type": score_type,
            }

        explicit_tau_grid, _ = self._grid_heat_density_and_score_at_time(p0, taus[0])
        score_type = "grid_heat"
        x = self._sample_from_grid_density(explicit_tau_grid, int(self.cfg.diffusion_particles))
        for m in range(n_steps):
            tau_cur = taus[m]
            tau_nxt = taus[m + 1]
            dtau = float(tau_cur - tau_nxt)
            p_tau, score_grid = self._grid_heat_density_and_score_at_time(p0, tau_cur)
            score = self._interp_vector_field(score_grid, x)
            x = x - dtau * score
            x = self._project_particles_to_domain(x, p_tau)
        particle_density = self._density_from_particles(x, smoothing_sigma=self._particle_density_smoothing_sigma())
        p_state = self._normalize_grid_density(particle_density)
        x_state = self._sample_from_grid_density(p_state, int(self.cfg.diffusion_particles))
        return p_state, x_state, {
            "explicit_tau_grid": explicit_tau_grid,
            "particle_density": particle_density,
            "state_blend_explicit": 0.0,
            "score_type": score_type,
        }

    def _shock_center_and_halfwidth(self) -> Tuple[float, float]:
        x_lo, x_hi = self.bounds[1]
        x_center = 0.0 if (x_lo <= 0.0 <= x_hi) else 0.5 * (x_lo + x_hi)

        rel_gate = float(min(getattr(self, "best_relL2", float("inf")), getattr(self, "latest_relL2", float("inf"))))
        if rel_gate > 2e-1:
            scale = float(getattr(self.cfg, "phi_shock_band_halfwidth_scale", 0.05))
        elif rel_gate > 1e-1:
            scale = 0.03
        elif rel_gate > 5e-2:
            scale = 0.02
        else:
            scale = 0.01

        halfwidth = scale * float(x_hi - x_lo)
        return x_center, max(halfwidth, 1e-6)

    def _sampling_attention_metrics(self, p_grid: np.ndarray, particles: np.ndarray) -> Dict[str, float]:
        metrics = {
            "topk_overlap": float("nan"),
            "shock_mass": float("nan"),
            "particle_x0_frac": float("nan"),
        }
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return metrics
        r2 = getattr(self, "last_residual_energy_grid", None)
        if r2 is None:
            return metrics
        valid_vals = r2[self.grid_valid_mask]
        if valid_vals.size > 0:
            q = float(getattr(self.cfg, "phi_topk_quantile", 0.95))
            thr = float(np.quantile(valid_vals, q))
            hotspot = (r2 >= thr) & self.grid_valid_mask
            metrics["topk_overlap"] = float(np.sum(p_grid[hotspot]))
        x_center, halfwidth = self._shock_center_and_halfwidth()
        x_axis = np.asarray(self.grid_axes[1], dtype=np.float64)
        xmask = np.abs(x_axis - x_center) <= halfwidth
        if np.any(xmask):
            metrics["shock_mass"] = float(np.sum(p_grid[:, xmask]))
        if particles is not None and particles.shape[0] > 0:
            metrics["particle_x0_frac"] = float(np.mean(np.abs(particles[:, 1] - x_center) <= halfwidth))
        return metrics

    def _shock_width_metrics(self, p_grid: np.ndarray) -> Dict[str, float]:
        out = {
            "cover_halfwidth": float("nan"),
            "ridge_width_mismatch": float("nan"),
            "shock_peak_ratio": float("nan"),
        }
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return out
        r2 = getattr(self, "last_residual_energy_grid", None)
        if r2 is None:
            return out

        x_axis = np.asarray(self.grid_axes[1], dtype=np.float64)
        if x_axis.size <= 1:
            return out
        dx = float(np.mean(np.diff(x_axis)))
        cover = np.asarray(p_grid, dtype=np.float64)
        cover_row = cover / np.maximum(np.sum(cover, axis=1, keepdims=True), 1e-12)
        r2_row = np.asarray(r2, dtype=np.float64)
        r2_row = r2_row / np.maximum(np.sum(r2_row, axis=1, keepdims=True), 1e-12)

        def row_halfwidth(row: np.ndarray) -> float:
            peak = float(np.max(row))
            if peak <= 0.0:
                return 0.0
            mask = row >= 0.5 * peak
            return float(np.sum(mask) * dx * 0.5)

        w_cover = np.array([row_halfwidth(cover_row[i]) for i in range(cover_row.shape[0])], dtype=np.float64)
        w_ridge = np.array([row_halfwidth(r2_row[i]) for i in range(r2_row.shape[0])], dtype=np.float64)

        row_w = np.max(r2, axis=1)
        if np.all(row_w <= 0.0):
            row_w = np.full_like(row_w, 1.0 / max(row_w.size, 1), dtype=np.float64)
        else:
            row_w = row_w / np.maximum(np.sum(row_w), 1e-12)

        out["cover_halfwidth"] = float(np.sum(row_w * w_cover))
        out["ridge_width_mismatch"] = float(np.sum(row_w * np.abs(w_cover - w_ridge)))

        x_center, halfwidth = self._shock_center_and_halfwidth()
        xmask = np.abs(x_axis - x_center) <= halfwidth
        if np.any(xmask):
            row_mass = np.sum(cover[:, xmask], axis=1)
            out["shock_peak_ratio"] = float(np.max(row_mass) / np.maximum(np.mean(np.sum(cover, axis=1)), 1e-12))
        return out

    def _build_shock_band_prior_grid(self) -> np.ndarray:
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return self._uniform_grid_density()
        x_center, halfwidth = self._shock_center_and_halfwidth()
        x_axis = np.asarray(self.grid_axes[1], dtype=np.float64)
        xmask = np.abs(x_axis - x_center) <= halfwidth
        prior = np.zeros(self.grid_shape, dtype=np.float64)
        if np.any(xmask):
            prior[:, xmask] = 1.0
        prior[~self.grid_valid_mask] = 0.0
        return self._normalize_grid_density(prior)

    def _build_late_shock_mass_prior_grid(self) -> np.ndarray:
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return self._uniform_grid_density()
        x_center, _ = self._shock_center_and_halfwidth()
        hw = self._late_shock_band_halfwidth()
        t_lo, t_hi = self._late_shock_time_window()
        t_axis = np.asarray(self.grid_axes[0], dtype=np.float64)
        x_axis = np.asarray(self.grid_axes[1], dtype=np.float64)
        tmask = (t_axis >= t_lo) & (t_axis <= t_hi)
        xmask = np.abs(x_axis - x_center) <= hw
        prior = np.zeros(self.grid_shape, dtype=np.float64)
        if not np.any(tmask) or not np.any(xmask):
            return self._build_shock_band_prior_grid()
        if getattr(self, "last_residual_energy_grid", None) is not None:
            r2 = np.asarray(self.last_residual_energy_grid, dtype=np.float64)
            row_w = np.max(r2, axis=1)
            row_w = np.where(np.isfinite(row_w), np.maximum(row_w, 0.0), 0.0)
            if np.any(row_w > 0.0):
                row_w = row_w / np.maximum(np.sum(row_w), 1e-12)
            else:
                row_w = np.full((self.grid_shape[0],), 1.0 / max(self.grid_shape[0], 1), dtype=np.float64)
        else:
            row_w = np.full((self.grid_shape[0],), 1.0 / max(self.grid_shape[0], 1), dtype=np.float64)
        prior[np.ix_(tmask, xmask)] = row_w[tmask, None]
        prior[~self.grid_valid_mask] = 0.0
        return self._normalize_grid_density(prior)

    def _enforce_shock_prior_mass(self, density: np.ndarray, prior: np.ndarray, target_mass: float) -> np.ndarray:
        density = self._normalize_grid_density(density)
        prior = self._normalize_grid_density(prior)
        if target_mass <= 0.0:
            return density
        support = prior > 0.0
        if not np.any(support):
            return density
        cur = float(np.sum(density[support]))
        tgt = float(np.clip(target_mass, 0.0, 0.92))
        if cur >= tgt - 1e-6:
            return density
        alpha = (tgt - cur) / max(1.0 - cur, 1e-12)
        alpha = float(np.clip(alpha, 0.0, 0.97))
        return self._normalize_grid_density((1.0 - alpha) * density + alpha * prior)

    def _shock_recovery_active(self) -> bool:
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return False
        rel_gate = self._rel_gate()
        if not np.isfinite(rel_gate):
            return False
        samp = getattr(self, "last_sampling_info", {})
        dd = getattr(self, "last_dd_info", {})
        shock_mass = float(samp.get("shock_mass", float("nan")))
        px0 = float(samp.get("particle_x0_frac", float("nan")))
        peakr = float(samp.get("shock_peak_ratio", float("nan")))
        ridge = float(dd.get("ridge_dx", float("nan")))
        poor = (
            (not np.isfinite(shock_mass)) or shock_mass < 0.14
            or (not np.isfinite(px0)) or px0 < 0.12
            or (not np.isfinite(peakr)) or peakr < 0.45
            or (np.isfinite(ridge) and ridge > 0.22)
        )
        return bool((rel_gate <= 1.2e-1 and poor) or (rel_gate <= float(getattr(self.cfg, "late_shock_rel_l2", 2.8e-2))))

    def _late_shock_mass_control_policy(self) -> Dict[str, float]:
        samp = getattr(self, "last_sampling_info", {})
        dd = getattr(self, "last_dd_info", {})
        shock_mass = float(samp.get("shock_mass", float("nan")))
        px0 = float(samp.get("particle_x0_frac", float("nan")))
        peakr = float(samp.get("shock_peak_ratio", float("nan")))
        ridge = float(dd.get("ridge_dx", float("nan")))
        rel_gate = self._rel_gate()

        pre_mix = 0.18
        post_mix = 0.14
        particle_replace = 0.06
        target_pre = 0.10
        target_post = 0.12

        if (not np.isfinite(shock_mass)) or shock_mass < 0.05 or (np.isfinite(px0) and px0 < 0.05) or (np.isfinite(peakr) and peakr < 0.20):
            pre_mix = 0.62 if rel_gate <= 5.0e-2 else 0.52
            post_mix = 0.55 if rel_gate <= 5.0e-2 else 0.44
            particle_replace = 0.18 if rel_gate <= 5.0e-2 else 0.14
            target_pre = 0.18 if rel_gate <= 5.0e-2 else 0.14
            target_post = 0.24 if rel_gate <= 5.0e-2 else 0.18
        elif shock_mass < 0.10 or (np.isfinite(px0) and px0 < 0.09) or (np.isfinite(peakr) and peakr < 0.35) or (np.isfinite(ridge) and ridge > 0.25):
            pre_mix = 0.52 if rel_gate <= 5.0e-2 else 0.42
            post_mix = 0.42 if rel_gate <= 5.0e-2 else 0.34
            particle_replace = 0.14 if rel_gate <= 5.0e-2 else 0.10
            target_pre = 0.16 if rel_gate <= 5.0e-2 else 0.12
            target_post = 0.20 if rel_gate <= 5.0e-2 else 0.16
        elif shock_mass < 0.16 or (np.isfinite(px0) and px0 < 0.14) or (np.isfinite(peakr) and peakr < 0.50) or (np.isfinite(ridge) and ridge > 0.16):
            pre_mix = 0.34
            post_mix = 0.26
            particle_replace = 0.08
            target_pre = 0.12
            target_post = 0.16

        return {
            "pre_mix": float(pre_mix),
            "post_mix": float(post_mix),
            "particle_replace": float(np.clip(particle_replace, 0.0, 0.35)),
            "target_pre": float(np.clip(target_pre, 0.0, 0.80)),
            "target_post": float(np.clip(target_post, 0.0, 0.85)),
        }

    def _late_shock_aux_loss_active(self) -> bool:
        if not self._late_shock_objective_active():
            return False
        samp = getattr(self, "last_sampling_info", {})
        dd = getattr(self, "last_dd_info", {})
        rel_gate = self._rel_gate()
        shock_mass = float(samp.get("shock_mass", float("nan")))
        px0 = float(samp.get("particle_x0_frac", float("nan")))
        peakr = float(samp.get("shock_peak_ratio", float("nan")))
        ridge = float(dd.get("ridge_dx", float("nan")))
        poor = (
            (not np.isfinite(shock_mass)) or shock_mass < 0.14
            or (not np.isfinite(px0)) or px0 < 0.12
            or (not np.isfinite(peakr)) or peakr < 0.45
            or (np.isfinite(ridge) and ridge > 0.18)
        )
        return bool(poor or (np.isfinite(rel_gate) and rel_gate <= float(getattr(self.cfg, "late_shock_rel_l2", 2.8e-2))))

    def _late_shock_objective_active(self) -> bool:
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return False
        rel_gate = self._rel_gate()
        if not np.isfinite(rel_gate):
            return False
        if self._shock_recovery_active():
            return True
        if rel_gate > float(getattr(self.cfg, "late_shock_rel_l2", 2.8e-2)):
            return False
        return True

    def _late_shock_freeze_dd_active(self) -> bool:
        if not bool(getattr(self.cfg, "late_shock_freeze_dd", True)):
            return False
        if not self._late_shock_objective_active():
            return False
        if self._rel_gate() > 1.5e-2:
            return False
        dd = getattr(self, "last_dd_info", {})
        samp = getattr(self, "last_sampling_info", {})
        ridge = float(dd.get("ridge_dx", float("nan")))
        shock_mass = float(samp.get("shock_mass", float("nan")))
        peakr = float(samp.get("shock_peak_ratio", float("nan")))
        ridge_gate = float(getattr(self.cfg, "late_shock_freeze_ridge_dx", 8.0e-2))
        shock_gate = float(getattr(self.cfg, "late_shock_freeze_shock_mass_min", 0.18))
        peakr_gate = float(getattr(self.cfg, "late_shock_freeze_peakr_min", 0.65))
        return (
            np.isfinite(ridge) and ridge <= ridge_gate
            and np.isfinite(shock_mass) and shock_mass >= shock_gate
            and np.isfinite(peakr) and peakr >= peakr_gate
        )

    def _late_shock_time_window(self) -> Tuple[float, float]:
        t_lo, t_hi = self.bounds[0]
        frac = float(np.clip(getattr(self.cfg, "late_shock_time_start_frac", 0.30), 0.0, 0.95))
        t_mid = t_lo + frac * (t_hi - t_lo)
        return t_mid, t_hi

    def _late_shock_band_halfwidth(self) -> float:
        _, base_hw = self._shock_center_and_halfwidth()
        x_lo, x_hi = self.bounds[1]
        span = float(x_hi - x_lo)
        rel_gate = self._rel_gate()
        cfg_hw = float(getattr(self.cfg, "late_shock_band_halfwidth_scale", 0.004)) * span
        if rel_gate <= 1e-2:
            hw = min(base_hw, 0.5 * cfg_hw)
        elif rel_gate <= 2e-2:
            hw = min(base_hw, 0.75 * cfg_hw)
        else:
            hw = min(base_hw, cfg_hw)
        return max(hw, 1e-6)

    def _late_shock_batch_sizes(self) -> Tuple[int, int]:
        b = int(self.cfg.batch_f if self.cfg.batch_f > 0 else self.cfg.n_r_pool)
        frac = float(np.clip(getattr(self.cfg, "late_shock_residual_frac", 0.015), 0.0, 0.95))
        base = int(round(b * frac))
        samp = getattr(self, "last_sampling_info", {})
        dd = getattr(self, "last_dd_info", {})
        shock_mass = float(samp.get("shock_mass", float("nan")))
        px0 = float(samp.get("particle_x0_frac", float("nan")))
        peakr = float(samp.get("shock_peak_ratio", float("nan")))
        ridge = float(dd.get("ridge_dx", float("nan")))
        poor = (
            (not np.isfinite(shock_mass)) or shock_mass < 0.10
            or (not np.isfinite(px0)) or px0 < 0.10
            or (not np.isfinite(peakr)) or peakr < 0.40
            or (np.isfinite(ridge) and ridge > 0.20)
        )
        shock_n = max(int(getattr(self.cfg, "late_shock_residual_min", 96)), base)
        shock_n = min(int(getattr(self.cfg, "late_shock_residual_max", 192)), shock_n)
        if poor:
            shock_n = max(shock_n, 160)
        anchor_n = int(max(getattr(self.cfg, "late_shock_anchor_points", 32), 0))
        if poor:
            anchor_n = max(anchor_n, 32)
        else:
            anchor_n = min(anchor_n, 16)
        return shock_n, anchor_n

    def _sample_late_shock_coords(self, n: int, line_frac: float = 0.75) -> np.ndarray:
        if n <= 0 or self.pde.name != "burgers_1d" or self.d_in != 2:
            return np.empty((0, self.d_in), dtype=np.float64)
        t_lo, t_hi = self.bounds[0]
        x_center, _ = self._shock_center_and_halfwidth()
        hw = self._late_shock_band_halfwidth()
        t_focus_lo, t_focus_hi = self._late_shock_time_window()
        n_line = int(round(line_frac * n))
        n_band = max(n - n_line, 0)
        n_focus = int(round(0.82 * n))
        n_full = n - n_focus
        t_focus = self.rng.uniform(t_focus_lo, t_focus_hi, size=(n_focus,)) if n_focus > 0 else np.empty((0,), dtype=np.float64)
        t_full = self.rng.uniform(t_lo, t_hi, size=(n_full,)) if n_full > 0 else np.empty((0,), dtype=np.float64)
        t = np.concatenate([t_focus, t_full], axis=0)
        self.rng.shuffle(t)
        x_line = np.full((n_line,), x_center, dtype=np.float64)
        if n_band > 0:
            band_sigma = max(0.45 * hw, 1e-6)
            x_band = x_center + self.rng.normal(loc=0.0, scale=band_sigma, size=(n_band,))
            x_band = np.clip(x_band, x_center - 1.25 * hw, x_center + 1.25 * hw)
            x = np.concatenate([x_line, x_band], axis=0)
        else:
            x = x_line
        self.rng.shuffle(x)
        return np.stack([t[:n], x[:n]], axis=1).astype(np.float64)

    def _sample_late_shock_anchor_batch(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n <= 0 or self.pde.name != "burgers_1d" or self.d_in != 2:
            return np.empty((0, self.d_in), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64)
        coords = self._sample_late_shock_coords(n, line_frac=0.85)
        x_center, _ = self._shock_center_and_halfwidth()
        coords[:, 1] = x_center
        vals = self.pde.exact_solution(coords)
        if vals is None:
            return np.empty((0, self.d_in), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64)
        mask = np.ones_like(vals, dtype=np.float64)
        return coords, np.asarray(vals, dtype=np.float64), mask

    def _sample_late_shock_residual_coords(self) -> np.ndarray:
        n, _ = self._late_shock_batch_sizes()
        return self._sample_late_shock_coords(n, line_frac=0.60)

    def _ctrl(self, key: str, default: Optional[float] = None) -> float:
        if key in self.feedback_ctrl:
            return float(self.feedback_ctrl[key])
        if default is None:
            raise KeyError(key)
        return float(default)

    def _feedback_targets(self, ridge_dx: float, shock_mass: float) -> Dict[str, float]:
        freeze_mode = str(getattr(self, "dd_freeze_mode", "active"))
        samp_info = getattr(self, "last_sampling_info", {})
        cover_halfwidth = float(samp_info.get("cover_halfwidth", float("nan")))
        ridge_width_mismatch = float(samp_info.get("ridge_width_mismatch", float("nan")))
        peakr = float(samp_info.get("shock_peak_ratio", float("nan")))
        rel_gate = self._rel_gate()

        base = {
            "dd_weight_power": float(self.cfg.dd_weight_power),
            "dd_residual_kappa": float(self.cfg.dd_residual_kappa),
            "dd_radius_quantile": float(self.cfg.dd_radius_quantile),
            "dd_axis_max_scale": float(self.cfg.dd_axis_max_scale),
            "dd_center_smooth_tau": float(self.cfg.dd_center_smooth_tau),
            "dd_metric_smooth_tau": float(self.cfg.dd_metric_smooth_tau),
            "dd_radius_growth_cap_rel": float(self.cfg.dd_radius_growth_cap_rel),
            "residual_focus_power": float(self.cfg.residual_focus_power),
            "shock_prior_mix": 0.06,
        }
        low_shock = max(float(self.cfg.feedback_shock_mass_low), 0.12)
        high_shock = max(float(self.cfg.feedback_shock_mass_high), 0.26)
        ridge_search = float(self.cfg.feedback_ridge_search_thresh)
        lock_width = float(getattr(self.cfg, "feedback_ridge_width_mismatch_lock_max", 0.02))
        lock_cover = float(getattr(self.cfg, "feedback_cover_halfwidth_lock_max", 0.03))

        if rel_gate <= 3e-2:
            ridge_lock_target = 0.08
            ridge_search_target = 0.12
        elif rel_gate <= 5e-2:
            ridge_lock_target = 0.10
            ridge_search_target = 0.14
        elif rel_gate <= 1e-1:
            ridge_lock_target = 0.18
            ridge_search_target = 0.24
        else:
            ridge_lock_target = max(ridge_search, 0.08)
            ridge_search_target = max(ridge_search, 0.10)

        width_good = (
            (not np.isfinite(cover_halfwidth) or cover_halfwidth <= lock_cover)
            and (not np.isfinite(ridge_width_mismatch) or ridge_width_mismatch <= lock_width)
        )
        width_bad = (
            (np.isfinite(cover_halfwidth) and cover_halfwidth > 0.05)
            or (np.isfinite(ridge_width_mismatch) and ridge_width_mismatch > 0.04)
        )
        late_stage = rel_gate <= 5e-2
        micro_stage = rel_gate <= 3e-2
        shock_weak = (
            (np.isfinite(peakr) and peakr < 0.50)
            or (np.isfinite(shock_mass) and shock_mass < 0.10)
        )

        if freeze_mode in ("micro_frozen", "frozen"):
            mode = "micro_lock" if freeze_mode == "micro_frozen" else "frozen_lock"
            if width_good and np.isfinite(ridge_dx) and ridge_dx <= 0.06 and not shock_weak:
                micro_mix = 0.03 if micro_stage else 0.04
            elif width_good and not shock_weak:
                micro_mix = 0.05
            else:
                micro_mix = max(float(getattr(self.cfg, "feedback_shock_prior_mix_micro_frozen", 0.08)), 0.10 if late_stage else 0.08)
            base.update({
                "dd_weight_power": max(base["dd_weight_power"], 0.70),
                "dd_residual_kappa": max(base["dd_residual_kappa"], 0.55),
                "dd_radius_quantile": min(base["dd_radius_quantile"], float(getattr(self.cfg, "dd_micro_radius_quantile", 0.45))),
                "dd_axis_max_scale": min(base["dd_axis_max_scale"], float(getattr(self.cfg, "dd_micro_axis_max_scale", 0.08))),
                "dd_center_smooth_tau": min(base["dd_center_smooth_tau"], float(getattr(self.cfg, "dd_micro_center_tau", 0.002))),
                "dd_metric_smooth_tau": min(base["dd_metric_smooth_tau"], float(getattr(self.cfg, "dd_micro_metric_tau", 0.001))),
                "dd_radius_growth_cap_rel": min(base["dd_radius_growth_cap_rel"], float(getattr(self.cfg, "dd_micro_radius_growth_cap_rel", 0.0))),
                "residual_focus_power": max(base["residual_focus_power"], float(getattr(self.cfg, "feedback_residual_focus_power_micro_frozen", 1.05))),
                "shock_prior_mix": micro_mix,
            })
        elif (
            (not np.isfinite(ridge_dx))
            or ridge_dx > ridge_search_target
            or width_bad
            or (rel_gate > 0.10 and np.isfinite(shock_mass) and shock_mass < low_shock)
        ):
            mode = "search"
            late_search_mix = 0.18 if (micro_stage and shock_weak) else (0.16 if shock_weak else 0.10)
            base.update({
                "dd_weight_power": max(base["dd_weight_power"], 0.72),
                "dd_residual_kappa": max(base["dd_residual_kappa"], 0.60),
                "dd_radius_quantile": min(base["dd_radius_quantile"], 0.60),
                "dd_axis_max_scale": min(base["dd_axis_max_scale"], 0.12),
                "dd_center_smooth_tau": min(base["dd_center_smooth_tau"], 0.03),
                "dd_metric_smooth_tau": min(base["dd_metric_smooth_tau"], 0.02),
                "dd_radius_growth_cap_rel": max(base["dd_radius_growth_cap_rel"], 0.008),
                "residual_focus_power": max(base["residual_focus_power"], 1.15),
                "shock_prior_mix": late_search_mix if late_stage else float(self.cfg.feedback_shock_prior_mix_search),
            })
        elif (not late_stage) and shock_mass > high_shock and np.isfinite(ridge_dx) and ridge_dx > 0.08:
            mode = "overfocus"
            base.update({
                "dd_weight_power": max(base["dd_weight_power"], 0.68),
                "dd_residual_kappa": max(base["dd_residual_kappa"], 0.55),
                "dd_radius_quantile": min(base["dd_radius_quantile"], 0.52),
                "dd_axis_max_scale": min(base["dd_axis_max_scale"], 0.10),
                "dd_center_smooth_tau": min(base["dd_center_smooth_tau"], 0.025),
                "dd_metric_smooth_tau": min(base["dd_metric_smooth_tau"], 0.018),
                "dd_radius_growth_cap_rel": min(base["dd_radius_growth_cap_rel"], 0.005),
                "residual_focus_power": min(max(base["residual_focus_power"] - 0.05, 0.95), 1.10),
                "shock_prior_mix": 0.08 if late_stage else float(self.cfg.feedback_shock_prior_mix_overfocus),
            })
        elif np.isfinite(ridge_dx) and ridge_dx <= ridge_lock_target and width_good:
            mode = "lock"
            base.update({
                "dd_weight_power": max(base["dd_weight_power"], 0.66),
                "dd_residual_kappa": max(base["dd_residual_kappa"], 0.52),
                "dd_radius_quantile": min(base["dd_radius_quantile"], 0.46 if late_stage else 0.48),
                "dd_axis_max_scale": min(base["dd_axis_max_scale"], 0.08 if late_stage else 0.09),
                "dd_center_smooth_tau": min(base["dd_center_smooth_tau"], 0.002 if late_stage else 0.015),
                "dd_metric_smooth_tau": min(base["dd_metric_smooth_tau"], 0.001 if late_stage else 0.010),
                "dd_radius_growth_cap_rel": 0.0 if late_stage else min(base["dd_radius_growth_cap_rel"], 0.002),
                "residual_focus_power": max(base["residual_focus_power"], 1.08),
                "shock_prior_mix": 0.10 if micro_stage else (0.12 if late_stage else 0.08),
            })
        else:
            mode = "cruise"
            base.update({
                "dd_weight_power": max(base["dd_weight_power"], 0.64),
                "dd_residual_kappa": max(base["dd_residual_kappa"], 0.50),
                "dd_radius_quantile": min(base["dd_radius_quantile"], 0.48 if late_stage else 0.52),
                "dd_axis_max_scale": min(base["dd_axis_max_scale"], 0.08 if late_stage else 0.10),
                "dd_center_smooth_tau": min(base["dd_center_smooth_tau"], 0.002 if late_stage else 0.018),
                "dd_metric_smooth_tau": min(base["dd_metric_smooth_tau"], 0.001 if late_stage else 0.012),
                "dd_radius_growth_cap_rel": 0.0 if late_stage else min(max(base["dd_radius_growth_cap_rel"], 0.003), 0.006),
                "residual_focus_power": max(base["residual_focus_power"], 1.05),
                "shock_prior_mix": 0.10 if micro_stage else (0.12 if late_stage else 0.06),
            })

        if late_stage:
            shock_ready = np.isfinite(shock_mass) and shock_mass >= 0.16 and np.isfinite(peakr) and peakr >= 0.70
            if shock_ready and np.isfinite(ridge_dx) and ridge_dx <= 0.10:
                base["dd_center_smooth_tau"] = min(base["dd_center_smooth_tau"], 0.002)
                base["dd_metric_smooth_tau"] = min(base["dd_metric_smooth_tau"], 0.001)
                base["dd_radius_growth_cap_rel"] = 0.0
                base["dd_axis_max_scale"] = min(base["dd_axis_max_scale"], 0.08)
                base["dd_radius_quantile"] = min(base["dd_radius_quantile"], 0.48)
            else:
                base["dd_center_smooth_tau"] = max(base["dd_center_smooth_tau"], 0.010)
                base["dd_metric_smooth_tau"] = max(base["dd_metric_smooth_tau"], 0.006)
                base["dd_radius_growth_cap_rel"] = max(base["dd_radius_growth_cap_rel"], 0.003)
                base["dd_axis_max_scale"] = max(base["dd_axis_max_scale"], 0.10)
                base["dd_radius_quantile"] = max(base["dd_radius_quantile"], 0.52)
            if (np.isfinite(shock_mass) and shock_mass < 0.08) or (np.isfinite(peakr) and peakr < 0.35):
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.34)
            elif (np.isfinite(shock_mass) and shock_mass < 0.14) or (np.isfinite(ridge_dx) and ridge_dx > 0.10) or (np.isfinite(peakr) and peakr < 0.55):
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.24)
            else:
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.16)
        if micro_stage:
            shock_ready = np.isfinite(shock_mass) and shock_mass >= 0.18 and np.isfinite(peakr) and peakr >= 0.80
            if shock_ready and np.isfinite(ridge_dx) and ridge_dx <= 0.08:
                base["dd_center_smooth_tau"] = min(base["dd_center_smooth_tau"], 0.001)
                base["dd_metric_smooth_tau"] = min(base["dd_metric_smooth_tau"], 0.0005)
                base["dd_axis_max_scale"] = min(base["dd_axis_max_scale"], 0.07)
                base["dd_radius_quantile"] = min(base["dd_radius_quantile"], 0.42)
            else:
                base["dd_center_smooth_tau"] = max(base["dd_center_smooth_tau"], 0.006)
                base["dd_metric_smooth_tau"] = max(base["dd_metric_smooth_tau"], 0.003)
                base["dd_axis_max_scale"] = max(base["dd_axis_max_scale"], 0.09)
                base["dd_radius_quantile"] = max(base["dd_radius_quantile"], 0.50)
                base["dd_radius_growth_cap_rel"] = max(base["dd_radius_growth_cap_rel"], 0.002)
            if (np.isfinite(shock_mass) and shock_mass < 0.08) or (np.isfinite(peakr) and peakr < 0.35):
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.38)
            elif (np.isfinite(shock_mass) and shock_mass < 0.14) or (np.isfinite(ridge_dx) and ridge_dx > 0.09) or (np.isfinite(peakr) and peakr < 0.55):
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.28)
            else:
                base["shock_prior_mix"] = max(base["shock_prior_mix"], 0.18)

        base["mode"] = mode
        return base

    def _update_feedback_controller(self) -> None:
        ridge_dx = float(self.last_dd_info.get("ridge_dx", float("nan")))
        shock_mass = float(self.last_sampling_info.get("shock_mass", float("nan")))
        targets = self._feedback_targets(ridge_dx, shock_mass)
        ema = float(np.clip(self.cfg.feedback_ema, 0.0, 1.0))
        if self.feedback_ctrl.get("mode", "bootstrap") == "bootstrap":
            ema = 1.0
        merged = dict(self.feedback_ctrl)
        merged["mode"] = str(targets["mode"])
        for key, val in targets.items():
            if key == "mode":
                continue
            old = float(self.feedback_ctrl.get(key, val))
            merged[key] = (1.0 - ema) * old + ema * float(val)
        self.feedback_ctrl = merged

    def update_sampling_distribution(self, params, geom: Dict[str, Array], outer_iter: int) -> None:
        focus_power = float(self._ctrl("residual_focus_power", self.cfg.residual_focus_power))
        p0 = self._build_residual_focus_density(params, geom, power_override=focus_power)

        late_shock_active = self._late_shock_objective_active()
        shock_prior_mix = float(np.clip(self._ctrl("shock_prior_mix", 0.0), 0.0, 0.90))
        shock_prior = None
        late_policy = None
        if self.pde.name == "burgers_1d" and self.d_in == 2:
            if late_shock_active:
                shock_prior = self._build_late_shock_mass_prior_grid()
                late_policy = self._late_shock_mass_control_policy()
                shock_prior_mix = max(shock_prior_mix, float(late_policy.get("pre_mix", 0.0)))
            elif shock_prior_mix > 0.0:
                shock_prior = self._build_shock_band_prior_grid()
            if shock_prior is not None and shock_prior_mix > 0.0:
                p0 = self._normalize_grid_density((1.0 - shock_prior_mix) * p0 + shock_prior_mix * shock_prior)
            if late_shock_active and shock_prior is not None and late_policy is not None:
                p0 = self._enforce_shock_prior_mass(p0, shock_prior, float(late_policy.get("target_pre", 0.0)))

        p_grid, particles, sampler_aux = self._run_ddim_reverse(p0)
        if late_shock_active and self.pde.name == "burgers_1d" and self.d_in == 2:
            if shock_prior is None:
                shock_prior = self._build_late_shock_mass_prior_grid()
            if late_policy is None:
                late_policy = self._late_shock_mass_control_policy()
            post_mix = float(np.clip(late_policy.get("post_mix", 0.0), 0.0, 0.75))
            if post_mix > 0.0:
                p_grid = self._normalize_grid_density((1.0 - post_mix) * p_grid + post_mix * shock_prior)
            p_grid = self._enforce_shock_prior_mass(p_grid, shock_prior, float(late_policy.get("target_post", 0.0)))
            if particles is not None and particles.shape[0] > 0:
                replace_frac = float(np.clip(late_policy.get("particle_replace", 0.0), 0.0, 0.35))
                n_rep = int(round(replace_frac * particles.shape[0]))
                if n_rep > 0:
                    idx = self.rng.choice(particles.shape[0], size=(n_rep,), replace=False)
                    particles = np.asarray(particles, dtype=np.float64).copy()
                    particles[idx] = self._sample_late_shock_coords(n_rep, line_frac=0.92)
        self.p_grid = self._normalize_grid_density(p_grid)
        self.p_particles = particles
        self.p_explicit_tau_grid = self._normalize_grid_density(sampler_aux.get("explicit_tau_grid", self.p_grid))
        self.p_particle_density = self._normalize_grid_density(sampler_aux.get("particle_density", self.p_grid))
        if late_shock_active and self.pde.name == "burgers_1d" and self.d_in == 2:
            self.p_explicit_tau_grid = self._enforce_shock_prior_mass(self.p_explicit_tau_grid, shock_prior, float(late_policy.get("target_post", 0.0)))
            self.p_particle_density = self._normalize_grid_density(self._density_from_particles(self.p_particles, smoothing_sigma=self._particle_density_smoothing_sigma()))
            self.p_particle_density = self._enforce_shock_prior_mass(self.p_particle_density, shock_prior, max(0.5 * float(late_policy.get("target_post", 0.0)), 0.0))

        attn = self._sampling_attention_metrics(self.p_grid, self.p_particles)
        width_metrics = self._shock_width_metrics(self.p_grid)
        self.last_sampling_info = {
            "status": "applied:ddim_shared_state_sampling",
            "score_type": str(sampler_aux.get("score_type", "unknown")),
            "time": float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt),
            "steps": int(getattr(self.cfg, "diffusion_steps", 1)),
            "r2_mean": float(np.mean(p0)),
            "r2_max": float(np.max(p0)),
            "grid_shape": tuple(int(s) for s in self.grid_shape),
            "topk_overlap": float(attn.get("topk_overlap", float("nan"))),
            "shock_mass": float(attn.get("shock_mass", float("nan"))),
            "particle_x0_frac": float(attn.get("particle_x0_frac", float("nan"))),
            "cover_halfwidth": float(width_metrics.get("cover_halfwidth", float("nan"))),
            "ridge_width_mismatch": float(width_metrics.get("ridge_width_mismatch", float("nan"))),
            "shock_peak_ratio": float(width_metrics.get("shock_peak_ratio", float("nan"))),
            "focus_power": float(focus_power),
            "shock_prior_mix": float(shock_prior_mix),
            "state_blend_explicit": float(sampler_aux.get("state_blend_explicit", 0.0)),
            "grid_particle_l1": float(np.sum(np.abs(self.p_grid - self.p_particle_density))),
            "grid_explicit_l1": float(np.sum(np.abs(self.p_grid - self.p_explicit_tau_grid))),
        }

    # ---------------------------------------------------------
    # anisotropic DD geometry update
    # ---------------------------------------------------------
    def _assignment(self, pts: np.ndarray, geom_np: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        diff = pts[:, None, :] - geom_np["centers"][None, :, :]
        d2 = np.einsum("nmd,mde,nme->nm", diff, geom_np["G_mats"], diff)
        assign = np.argmin(d2, axis=1)
        return assign, d2

    def _project_spd(self, A: np.ndarray) -> np.ndarray:
        A = 0.5 * (A + A.T)
        vals, vecs = np.linalg.eigh(A)
        vals = np.clip(vals, float(self.cfg.dd_eig_min), float(self.cfg.dd_eig_max))
        cond_max = max(float(getattr(self.cfg, "dd_cond_max", 1e2)), 1.0)
        lam_max = float(np.max(vals))
        lam_min_allowed = max(float(self.cfg.dd_eig_min), lam_max / cond_max)
        vals = np.clip(vals, lam_min_allowed, float(self.cfg.dd_eig_max))
        return (vecs * vals.reshape(1, -1)) @ vecs.T

    def _axis_bounds_abs(self, axis_max_scale: Optional[float] = None) -> Tuple[float, float]:
        a_min = max(float(getattr(self.cfg, "dd_axis_min_scale", 0.04)) * self.diag, 1e-8)
        eff_axis_max = float(getattr(self.cfg, "dd_axis_max_scale", 0.30) if axis_max_scale is None else axis_max_scale)
        a_max = max(eff_axis_max * self.diag, a_min)
        return a_min, a_max

    def _effective_axis_stats(self, radii: np.ndarray, G_mats: np.ndarray) -> Tuple[float, float]:
        amin = float("nan")
        amax = float("nan")
        vals_all = np.linalg.eigvalsh(0.5 * (G_mats + np.transpose(G_mats, (0, 2, 1))))
        vals_all = np.clip(vals_all, 1e-12, None)
        axes = radii[:, None] / np.sqrt(vals_all)
        amin = float(np.min(axes))
        amax = float(np.max(axes))
        return amin, amax

    def _clamp_radii_by_axes(self, radii: np.ndarray, G_mats: np.ndarray, axis_max_scale: Optional[float] = None) -> np.ndarray:
        out = np.asarray(radii, dtype=np.float64).copy()
        a_min, a_max = self._axis_bounds_abs(axis_max_scale=axis_max_scale)
        vals_all = np.linalg.eigvalsh(0.5 * (G_mats + np.transpose(G_mats, (0, 2, 1))))
        vals_all = np.clip(vals_all, 1e-12, None)
        for j in range(out.shape[0]):
            lam_min = float(np.min(vals_all[j]))
            lam_max = float(np.max(vals_all[j]))
            r_low = max(self.min_radius, a_min * math.sqrt(lam_max))
            r_high = a_max * math.sqrt(lam_min)
            if r_high < r_low:
                out[j] = r_low
            else:
                out[j] = float(np.clip(out[j], r_low, r_high))
        return out

    def _residual_grad_at_center(self, params, geom_trial: Dict[str, Array], center_np: np.ndarray) -> np.ndarray:
        x = jnp.asarray(center_np.reshape(1, -1))

        def scalar_rho(coord: Array) -> Array:
            r = self.residual_tensor(params, geom_trial, coord[None, :])[0]
            return jnp.sum(r * r)

        g = jax_grad(scalar_rho)(x[0])
        return np.asarray(g, dtype=np.float64)

    def _maybe_advance_dd_freeze(self, improved: bool, rel_l2_epoch: float) -> None:
        if not bool(getattr(self.cfg, "dd_freeze_enable", False)):
            self.dd_freeze_mode = "active"
            self.dd_no_improve_streak = 0
            return
        if not np.isfinite(rel_l2_epoch):
            return

        tol = float(getattr(self.cfg, "dd_freeze_improve_tol", 1e-4))
        if improved or (rel_l2_epoch + tol < self.dd_best_rel_l2_seen):
            self.dd_best_rel_l2_seen = float(rel_l2_epoch)
            self.dd_no_improve_streak = 0
        else:
            self.dd_no_improve_streak += 1

        cond_now = float(self.last_dd_info.get("cond_max_eff", float("inf"))) if isinstance(self.last_dd_info, dict) else float("inf")
        ridge_now = float(self.last_dd_info.get("ridge_dx", float("inf"))) if isinstance(self.last_dd_info, dict) else float("inf")
        cond_thresh = float(getattr(self.cfg, "dd_freeze_cond_thresh", 6.0))
        current_iter = int(getattr(self, "current_outer_iter", 0))

        if self.dd_freeze_mode == "active":
            if (
                rel_l2_epoch <= float(getattr(self.cfg, "dd_semi_freeze_rel_l2", 0.08))
                and self.dd_no_improve_streak >= int(getattr(self.cfg, "dd_semi_freeze_patience", 12))
                and current_iter >= max(100, int(0.25 * float(getattr(self.cfg, "dd_warmup_iters", 800))))
            ):
                self.dd_freeze_mode = "semi_frozen"
                self.dd_no_improve_streak = 0
                return

        if self.dd_freeze_mode == "semi_frozen":
            if (
                rel_l2_epoch <= float(getattr(self.cfg, "dd_full_freeze_rel_l2", 0.04))
                and ridge_now <= float(getattr(self.cfg, "dd_micro_freeze_ridge_dx", 0.08))
                and self.dd_no_improve_streak >= int(getattr(self.cfg, "dd_full_freeze_patience", 20))
                and current_iter >= int(getattr(self.cfg, "dd_micro_freeze_min_iter", getattr(self.cfg, "dd_warmup_iters", 800)))
                and (not np.isfinite(cond_now) or cond_now <= cond_thresh)
            ):
                self.dd_freeze_mode = "micro_frozen"
                self.dd_no_improve_streak = 0
                self.last_micro_dd_update_iter = -10**9
                return

        if self.dd_freeze_mode == "micro_frozen":
            return

        if self.dd_freeze_mode == "frozen":
            return

    def _residual_cover_ridge_distance(self, geom_np: Dict[str, np.ndarray]) -> float:
        if self.pde.name != "burgers_1d" or self.d_in != 2:
            return float("nan")
        r2 = getattr(self, "last_residual_energy_grid", None)
        if r2 is None:
            return float("nan")
        cover_sum = self._cover_sum_np(self.grid_coords, geom_np["centers"], geom_np["radii"], geom_np["G_mats"]).reshape(self.grid_shape)
        x_axis = np.asarray(self.grid_axes[1], dtype=np.float64)
        ridx_r = np.argmax(r2, axis=1)
        ridx_c = np.argmax(cover_sum, axis=1)
        dx = np.abs(x_axis[ridx_r] - x_axis[ridx_c])
        row_w = np.max(r2, axis=1)
        if np.all(row_w <= 0.0):
            return float(np.mean(dx))
        row_w = row_w / np.maximum(np.sum(row_w), 1e-12)
        return float(np.sum(row_w * dx))

    def dd_geometry_update(self, params, outer_iter: int) -> None:
        pts_adapt = self.p_particles
        mode = str(getattr(self, "dd_hardening_state", "adaptive"))
        freeze_mode = str(getattr(self, "dd_freeze_mode", "active"))
        if pts_adapt is None or pts_adapt.shape[0] == 0:
            self.last_dd_info = {"status": "skipped:no_particles", "mode": mode, "freeze_mode": freeze_mode}
            return

        centers = np.asarray(self.geom_state["centers"], dtype=np.float64)
        radii = np.asarray(self.geom_state["radii"], dtype=np.float64)
        G_mats = np.asarray(self.geom_state["G_mats"], dtype=np.float64)
        old_centers = centers.copy()
        old_radii = radii.copy()
        old_G_mats = G_mats.copy()

        rel_gate = self._rel_gate()
        samp_info_now = getattr(self, "last_sampling_info", {})
        cover_halfwidth_now = float(samp_info_now.get("cover_halfwidth", float("nan")))
        ridge_width_mismatch_now = float(samp_info_now.get("ridge_width_mismatch", float("nan")))
        shock_mass_now = float(samp_info_now.get("shock_mass", float("nan")))
        peakr_now = float(samp_info_now.get("shock_peak_ratio", float("nan")))
        dd_stride = 1
        if rel_gate <= 3e-2:
            if np.isfinite(shock_mass_now) and shock_mass_now >= 0.10 and np.isfinite(peakr_now) and peakr_now >= 0.45:
                dd_stride = 3
            else:
                dd_stride = 1
        elif rel_gate <= 5e-2:
            if np.isfinite(shock_mass_now) and shock_mass_now >= 0.11 and np.isfinite(peakr_now) and peakr_now >= 0.40:
                dd_stride = 2
            else:
                dd_stride = 1

        prev_ridge_dx = float(getattr(self, "last_dd_info", {}).get("ridge_dx", float("nan")))
        shock_force = ((np.isfinite(shock_mass_now) and shock_mass_now < 0.10) or (np.isfinite(peakr_now) and peakr_now < 0.40))
        if rel_gate <= 3e-2:
            force_dd_update = (
                (np.isfinite(prev_ridge_dx) and prev_ridge_dx > 0.08)
                or (np.isfinite(cover_halfwidth_now) and cover_halfwidth_now > 0.028)
                or (np.isfinite(ridge_width_mismatch_now) and ridge_width_mismatch_now > 0.016)
                or shock_force
            )
        elif rel_gate <= 5e-2:
            force_dd_update = (
                (np.isfinite(prev_ridge_dx) and prev_ridge_dx > 0.10)
                or (np.isfinite(cover_halfwidth_now) and cover_halfwidth_now > 0.032)
                or (np.isfinite(ridge_width_mismatch_now) and ridge_width_mismatch_now > 0.018)
                or shock_force
            )
        else:
            force_dd_update = (
                (np.isfinite(cover_halfwidth_now) and cover_halfwidth_now > 0.036)
                or (np.isfinite(ridge_width_mismatch_now) and ridge_width_mismatch_now > 0.020)
                or (np.isfinite(prev_ridge_dx) and prev_ridge_dx > 0.12)
                or shock_force
            )
        if outer_iter - int(getattr(self, "last_dd_update_iter", -10**9)) < dd_stride and not force_dd_update:
            prev_info = dict(getattr(self, "last_dd_info", {}))
            prev_info.update({
                "status": f"skipped:late_cadence/{dd_stride}",
                "mode": mode,
                "freeze_mode": freeze_mode,
                "radius_smooth_tau": 0.0,
                "center_smooth_tau": 0.0,
                "metric_smooth_tau": 0.0,
                "hardening_factor": float(self.dd_hardening_factor),
                "hardening_state": str(self.dd_hardening_state),
            })
            self.last_dd_info = prev_info
            return

        alpha = min(1.0, float(outer_iter) / max(float(self.cfg.dd_warmup_iters), 1.0))
        hard = float(np.clip(self.dd_hardening_factor, 0.12, 1.0))

        tau_c = float(np.clip(self._ctrl("dd_center_smooth_tau", self.cfg.dd_center_smooth_tau) * hard, 0.0, 1.0))
        tau_g = float(np.clip(self._ctrl("dd_metric_smooth_tau", self.cfg.dd_metric_smooth_tau) * hard, 0.0, 1.0))
        tau_r = float(np.clip(self.cfg.dd_radius_smooth_tau * (0.75 + 0.25 * hard), 0.0, 1.0))
        empty_keep_old = bool(getattr(self.cfg, "dd_empty_keep_old", True))

        weight_power_eff = float(self._ctrl("dd_weight_power", self.cfg.dd_weight_power))
        residual_kappa_eff = float(self._ctrl("dd_residual_kappa", self.cfg.dd_residual_kappa))
        radius_quantile_eff = float(self._ctrl("dd_radius_quantile", self.cfg.dd_radius_quantile))
        axis_max_scale_eff = float(self._ctrl("dd_axis_max_scale", self.cfg.dd_axis_max_scale))
        radius_growth_cap_eff = float(self._ctrl("dd_radius_growth_cap_rel", self.cfg.dd_radius_growth_cap_rel)) * (0.65 + 0.35 * hard)

        shock_ready_for_slowdown = (
            np.isfinite(shock_mass_now) and shock_mass_now >= 0.12
            and np.isfinite(peakr_now) and peakr_now >= 0.45
            and np.isfinite(prev_ridge_dx) and prev_ridge_dx <= 0.14
        )
        if rel_gate <= 5e-2:
            if shock_ready_for_slowdown:
                tau_c = min(tau_c, 0.002)
                tau_g = min(tau_g, 0.001)
                tau_r = min(tau_r, 1e-4)
                radius_growth_cap_eff = 0.0
                radius_quantile_eff = min(radius_quantile_eff, 0.48)
                axis_max_scale_eff = min(axis_max_scale_eff, 0.08)
            else:
                tau_c = max(tau_c, 0.010)
                tau_g = max(tau_g, 0.006)
                tau_r = max(tau_r, 5e-4)
                radius_growth_cap_eff = max(radius_growth_cap_eff, 0.002)
                radius_quantile_eff = max(radius_quantile_eff, 0.50)
                axis_max_scale_eff = max(axis_max_scale_eff, 0.09)
        if rel_gate <= 3e-2:
            if shock_ready_for_slowdown:
                tau_c = min(tau_c, 0.001)
                tau_g = min(tau_g, 5e-4)
                tau_r = min(tau_r, 5e-5)
                radius_growth_cap_eff = 0.0
                radius_quantile_eff = min(radius_quantile_eff, 0.42)
                axis_max_scale_eff = min(axis_max_scale_eff, 0.07)
            else:
                tau_c = max(tau_c, 0.006)
                tau_g = max(tau_g, 0.003)
                tau_r = max(tau_r, 3e-4)
                radius_growth_cap_eff = max(radius_growth_cap_eff, 0.001)
                radius_quantile_eff = max(radius_quantile_eff, 0.48)
                axis_max_scale_eff = max(axis_max_scale_eff, 0.08)
        if rel_gate <= 1e-1 and shock_ready_for_slowdown:
            tau_c = min(tau_c, 0.010)
            tau_g = min(tau_g, 0.006)
            tau_r = min(tau_r, 5e-4)

        if freeze_mode == "semi_frozen":
            tau_c = min(tau_c, 0.35 * float(self.cfg.dd_center_smooth_tau))
            tau_g = min(tau_g, 0.20 * float(self.cfg.dd_metric_smooth_tau))
            tau_r = min(tau_r, float(getattr(self.cfg, "dd_semi_radius_tau", self.cfg.dd_radius_smooth_tau)))
            radius_growth_cap_eff = min(radius_growth_cap_eff, float(getattr(self.cfg, "dd_semi_radius_growth_cap_rel", 0.005)))
            radius_quantile_eff = min(radius_quantile_eff, 0.60)
        elif freeze_mode == "micro_frozen":
            tau_c = min(tau_c, float(getattr(self.cfg, "dd_micro_center_tau", 0.001)))
            tau_g = min(tau_g, float(getattr(self.cfg, "dd_micro_metric_tau", 0.0006)))
            tau_r = min(tau_r, float(getattr(self.cfg, "dd_micro_radius_tau", 0.00005)))
            radius_growth_cap_eff = min(radius_growth_cap_eff, float(getattr(self.cfg, "dd_micro_radius_growth_cap_rel", 0.0)))
            radius_quantile_eff = min(radius_quantile_eff, float(getattr(self.cfg, "dd_micro_radius_quantile", 0.58)))
            axis_max_scale_eff = min(axis_max_scale_eff, float(getattr(self.cfg, "dd_micro_axis_max_scale", 0.16)))
        elif freeze_mode == "frozen":
            axis_min_eff, axis_max_eff = self._effective_axis_stats(radii, G_mats)
            cond_vals = np.linalg.eigvalsh(0.5 * (G_mats + np.transpose(G_mats, (0, 2, 1))))
            cond_vals = np.clip(cond_vals, 1e-12, None)
            cond_max_eff = float(np.max(np.max(cond_vals, axis=1) / np.maximum(np.min(cond_vals, axis=1), 1e-12)))
            ridge_dx = self._residual_cover_ridge_distance({"centers": centers, "radii": radii, "G_mats": G_mats})
            cover_sets = [self.dd_cover_points]
            if self.data_coords_full is not None and self.data_coords_full.shape[0] > 0:
                cover_sets.append(self.data_coords_full)
            cover_pts_stats = np.concatenate([p for p in cover_sets if p is not None and p.shape[0] > 0], axis=0)
            cover_stats = self._coverage_stats_np(centers, radii, G_mats, cover_pts_stats)
            self.last_dd_info = {
                "status": "skipped:frozen_dd",
                "mode": mode,
                "freeze_mode": freeze_mode,
                "alpha": float(alpha),
                "mean_radius": float(np.mean(radii)),
                "radius_quantile": float(radius_quantile_eff),
                "radius_margin": float(self.cfg.dd_radius_margin),
                "radius_smooth_tau": 0.0,
                "center_smooth_tau": 0.0,
                "metric_smooth_tau": 0.0,
                "weight_power_eff": float(weight_power_eff),
                "residual_kappa_eff": float(residual_kappa_eff),
                "cover_sum_min": float(cover_stats.get("cover_sum_min", float("nan"))),
                "uncovered_frac": float(cover_stats.get("uncovered_frac", float("nan"))),
                "repair_passes": 0,
                "repair_expansions": 0,
                "repair_exact": 0,
                "axis_min_eff": float(axis_min_eff),
                "axis_max_eff": float(axis_max_eff),
                "cond_max_eff": float(cond_max_eff),
                "geometry_drift": 0.0,
                "center_drift": 0.0,
                "radius_drift": 0.0,
                "metric_drift": 0.0,
                "ridge_dx": float(ridge_dx),
                "hardening_factor": float(self.dd_hardening_factor),
                "hardening_state": str(self.dd_hardening_state),
            }
            return

        proposal_density = self._proposal_density_from_grid(pts_adapt, self.p_grid).reshape(-1)
        r2 = self._residual_energy_batched(params, self.geom_state, pts_adapt, self.cfg.test_batch_size)
        importance_r2 = r2 / (proposal_density * self.admissible_volume + 1e-12)
        weights_mix = np.power(np.maximum(importance_r2, 1e-12), weight_power_eff)
        weights_mix = weights_mix / max(float(np.mean(weights_mix)), 1e-12)
        pts_mix = np.asarray(pts_adapt, dtype=np.float64)

        for _ in range(max(int(self.cfg.dd_lloyd_steps), 1)):
            geom_np = {"centers": centers, "radii": radii, "G_mats": G_mats}
            assign, _ = self._assignment(pts_mix, geom_np)

            center_prop = centers.copy()
            for j in range(self.cfg.n_balls):
                mask = assign == j
                if np.any(mask):
                    ww = weights_mix[mask]
                    center_prop[j] = (pts_mix[mask] * ww[:, None]).sum(axis=0) / np.maximum(ww.sum(), 1e-12)
                elif empty_keep_old:
                    center_prop[j] = centers[j]
                else:
                    ridx = int(self.rng.integers(0, pts_mix.shape[0]))
                    center_prop[j] = pts_mix[ridx]
            centers = (1.0 - tau_c) * centers + tau_c * center_prop

            G_prop = G_mats.copy()
            geom_tmp = {
                "centers": jnp.asarray(centers),
                "radii": jnp.asarray(radii),
                "G_mats": jnp.asarray(G_mats),
            }
            assign, _ = self._assignment(pts_mix, {"centers": centers, "radii": radii, "G_mats": G_mats})
            for j in range(self.cfg.n_balls):
                mask = assign == j
                if np.any(mask):
                    ww = weights_mix[mask]
                    zz = pts_mix[mask]
                    zbar = (zz * ww[:, None]).sum(axis=0) / np.maximum(ww.sum(), 1e-12)
                    centered = zz - zbar.reshape(1, -1)
                    Sigma = (centered * ww[:, None]).T @ centered / np.maximum(ww.sum(), 1e-12)
                else:
                    Sigma = np.eye(self.d_in, dtype=np.float64) * (self.min_radius ** 2)
                Gp = np.linalg.inv(0.5 * (Sigma + Sigma.T) + float(self.cfg.dd_cov_eps) * np.eye(self.d_in))
                Gp = self._project_spd(Gp)
                gvec = self._residual_grad_at_center(params, geom_tmp, centers[j])
                Gres = np.eye(self.d_in, dtype=np.float64) + residual_kappa_eff * np.outer(gvec, gvec)
                Gres = self._project_spd(Gres)
                target = self._project_spd((1.0 - alpha) * Gp + alpha * Gres)
                G_prop[j] = target
            G_mats = np.stack([self._project_spd((1.0 - tau_g) * G_mats[j] + tau_g * G_prop[j]) for j in range(self.cfg.n_balls)], axis=0)

        adaptive_cover_pts = [pts_adapt]
        if self.data_coords_full is not None and self.data_coords_full.shape[0] > 0:
            adaptive_cover_pts.append(self.data_coords_full)
        adaptive_cover_pts = np.concatenate(adaptive_cover_pts, axis=0)
        assign_cover, d2_cover = self._assignment(adaptive_cover_pts, {"centers": centers, "radii": radii, "G_mats": G_mats})
        q = float(np.clip(radius_quantile_eff, 0.0, 1.0))
        radius_prop = radii.copy()
        for j in range(self.cfg.n_balls):
            mask = assign_cover == j
            if np.any(mask):
                vals = np.sqrt(np.maximum(d2_cover[mask, j], 0.0))
                rj = float(np.max(vals) if q >= 1.0 - 1e-12 else np.quantile(vals, q))
                radius_prop[j] = max(float(self.cfg.dd_radius_margin) * rj, self.min_radius)
            else:
                radius_prop[j] = max(float(radii[j]), self.min_radius)

        rel_gate = float(min(getattr(self, "best_relL2", float("inf")), getattr(self, "latest_relL2", float("inf"))))
        shrink_gain = 1.0
        if rel_gate <= 0.15:
            shrink_gain = float(getattr(self.cfg, "dd_late_shrink_gain", 1.5))
        if rel_gate <= 0.08:
            shrink_gain = float(getattr(self.cfg, "dd_micro_shrink_gain", 2.0))
        if rel_gate <= 0.05:
            shrink_gain = max(shrink_gain, 3.0)
        if rel_gate <= 0.03:
            shrink_gain = max(shrink_gain, 4.0)

        new_radii = np.where(
            radius_prop < radii,
            np.maximum(radii - shrink_gain * tau_r * (radii - radius_prop), self.min_radius),
            np.maximum((1.0 - tau_r) * radii + tau_r * radius_prop, self.min_radius),
        )
        if radius_growth_cap_eff <= 0.0:
            new_radii = np.minimum(new_radii, radii)
        else:
            new_radii = np.minimum(new_radii, radii * (1.0 + radius_growth_cap_eff))
        new_radii = self._clamp_radii_by_axes(new_radii, G_mats, axis_max_scale=axis_max_scale_eff)

        cover_sets = [self.dd_cover_points]
        if self.data_coords_full is not None and self.data_coords_full.shape[0] > 0:
            cover_sets.append(self.data_coords_full)
        cover_pts_stats = np.concatenate([p for p in cover_sets if p is not None and p.shape[0] > 0], axis=0)
        cover_stats = self._coverage_stats_np(centers, new_radii, G_mats, cover_pts_stats)
        new_radii, repair_stats = self._weak_cover_repair(centers, new_radii, G_mats, cover_pts_stats)
        cover_stats = self._coverage_stats_np(centers, new_radii, G_mats, cover_pts_stats)

        axis_min_eff, axis_max_eff = self._effective_axis_stats(new_radii, G_mats)
        cond_vals = np.linalg.eigvalsh(0.5 * (G_mats + np.transpose(G_mats, (0, 2, 1))))
        cond_vals = np.clip(cond_vals, 1e-12, None)
        cond_max_eff = float(np.max(np.max(cond_vals, axis=1) / np.maximum(np.min(cond_vals, axis=1), 1e-12)))

        center_drift = float(np.max(np.linalg.norm(centers - old_centers, axis=1)) / max(self.diag, 1e-12))
        radius_drift = float(np.max(np.abs(new_radii - old_radii) / np.maximum(old_radii, 1e-12)))
        G_old_norm = np.linalg.norm(old_G_mats.reshape(old_G_mats.shape[0], -1), axis=1)
        G_new_norm = np.linalg.norm((G_mats - old_G_mats).reshape(G_mats.shape[0], -1), axis=1)
        metric_drift = float(np.max(G_new_norm / np.maximum(G_old_norm, 1e-12)))
        combined_drift = max(center_drift, radius_drift, metric_drift)

        geom_np_final = {"centers": centers, "radii": new_radii, "G_mats": G_mats}
        self.last_dd_update_iter = int(outer_iter)
        ridge_dx = self._residual_cover_ridge_distance(geom_np_final)
        self.geom_state = {
            "centers": jnp.asarray(centers),
            "radii": jnp.asarray(new_radii),
            "G_mats": jnp.asarray(G_mats),
        }
        if freeze_mode == "micro_frozen":
            self.last_micro_dd_update_iter = int(outer_iter)
        self.last_dd_info = {
            "status": "applied:voronoi_lloyd_shared_state",
            "mode": mode,
            "freeze_mode": freeze_mode,
            "alpha": float(alpha),
            "mean_radius": float(np.mean(new_radii)),
            "radius_quantile": float(q),
            "radius_margin": float(self.cfg.dd_radius_margin),
            "radius_smooth_tau": float(tau_r),
            "center_smooth_tau": float(tau_c),
            "metric_smooth_tau": float(tau_g),
            "weight_power_eff": float(weight_power_eff),
            "residual_kappa_eff": float(residual_kappa_eff),
            "cover_sum_min": float(cover_stats.get("cover_sum_min", float("nan"))),
            "uncovered_frac": float(cover_stats.get("uncovered_frac", float("nan"))),
            "repair_passes": int(repair_stats.get("repair_passes", 0)),
            "repair_expansions": int(repair_stats.get("repair_expansions", 0)),
            "repair_exact": int(repair_stats.get("repair_exact", 0)),
            "axis_min_eff": float(axis_min_eff),
            "axis_max_eff": float(axis_max_eff),
            "cond_max_eff": float(cond_max_eff),
            "geometry_drift": float(combined_drift),
            "center_drift": float(center_drift),
            "radius_drift": float(radius_drift),
            "metric_drift": float(metric_drift),
            "ridge_dx": float(ridge_dx),
            "hardening_factor": float(self.dd_hardening_factor),
            "hardening_state": str(self.dd_hardening_state),
        }

    # ---------------------------------------------------------
    # metrics / plotting
    # ---------------------------------------------------------
    def cover_sum_batched(self, geom: Dict[str, Array], coords: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords.shape[0], batch_size):
            ed = min(st + batch_size, coords.shape[0])
            xb = jnp.asarray(coords[st:ed])
            ph = anisotropic_phi_basis_geom(xb, geom)
            outs.append(np.asarray(jnp.sum(ph, axis=1)))
        return np.concatenate(outs, axis=0)

    def theta_objective(
        self,
        params,
        geom: Dict[str, Array],
        data_coords: Array,
        data_vals: Array,
        data_mask: Array,
        f_coords: Array,
        proposal_density: Optional[Array] = None,
        shock_coords: Optional[Array] = None,
        anchor_coords: Optional[Array] = None,
        anchor_vals: Optional[Array] = None,
        anchor_mask: Optional[Array] = None,
    ):
        mse_u, _ = self.data_loss_and_residuals(params, geom, data_coords, data_vals, data_mask)
        mse_f = jnp.array(0.0, dtype=jnp.float64)
        mse_f_shock = jnp.array(0.0, dtype=jnp.float64)
        mse_anchor = jnp.array(0.0, dtype=jnp.float64)
        if f_coords.shape[0] > 0:
            r = self.residual_tensor(params, geom, f_coords)
            r2 = jnp.sum(r * r, axis=1)
            if proposal_density is None:
                mse_f = jnp.mean(r2)
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                mse_f = jnp.mean(r2 / (pbar * self.admissible_volume + 1e-12))
        if shock_coords is not None and shock_coords.shape[0] > 0:
            r_shock = self.residual_tensor(params, geom, shock_coords)
            mse_f_shock = jnp.mean(jnp.sum(r_shock * r_shock, axis=1))
        if anchor_coords is not None and anchor_vals is not None and anchor_mask is not None and anchor_coords.shape[0] > 0:
            pred_anchor = model_forward(params, geom, anchor_coords, self.act_name)
            diff_anchor = (pred_anchor - anchor_vals) * anchor_mask
            denom_a = jnp.maximum(jnp.sum(anchor_mask), 1.0)
            mse_anchor = jnp.sum(diff_anchor * diff_anchor) / denom_a
        lam_shock = float(getattr(self.cfg, "late_shock_residual_weight", 6.0))
        lam_anchor = float(getattr(self.cfg, "late_shock_anchor_weight", 10.0))
        return mse_u + mse_f + lam_shock * mse_f_shock + lam_anchor * mse_anchor

    def global_residual_vector(
        self,
        params,
        geom: Dict[str, Array],
        data_coords: Array,
        data_vals: Array,
        data_mask: Array,
        f_coords: Array,
        proposal_density: Optional[Array] = None,
        shock_coords: Optional[Array] = None,
        anchor_coords: Optional[Array] = None,
        anchor_vals: Optional[Array] = None,
        anchor_mask: Optional[Array] = None,
    ) -> Array:
        parts = []
        if data_coords.shape[0] > 0:
            pred = model_forward(params, geom, data_coords, self.act_name)
            diff = (pred - data_vals) * data_mask
            denom_u = jnp.sqrt(jnp.maximum(jnp.sum(data_mask), 1.0))
            parts.append(jnp.reshape(diff / denom_u, (-1,)))
        if f_coords.shape[0] > 0:
            r = self.residual_tensor(params, geom, f_coords)
            denom_f = jnp.sqrt(jnp.maximum(float(r.shape[0]), 1.0))
            if proposal_density is None:
                r_scaled = r / denom_f
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                iw_sqrt = jnp.sqrt(1.0 / (pbar * self.admissible_volume + 1e-12)).reshape(-1, 1)
                r_scaled = r * iw_sqrt / denom_f
            parts.append(jnp.reshape(r_scaled, (-1,)))
        if shock_coords is not None and shock_coords.shape[0] > 0:
            r_shock = self.residual_tensor(params, geom, shock_coords)
            shock_scale = math.sqrt(max(float(getattr(self.cfg, "late_shock_residual_weight", 6.0)), 0.0)) / jnp.sqrt(jnp.maximum(float(r_shock.shape[0]), 1.0))
            parts.append(jnp.reshape(r_shock * shock_scale, (-1,)))
        if anchor_coords is not None and anchor_vals is not None and anchor_mask is not None and anchor_coords.shape[0] > 0:
            pred_anchor = model_forward(params, geom, anchor_coords, self.act_name)
            diff_anchor = (pred_anchor - anchor_vals) * anchor_mask
            denom_a = jnp.sqrt(jnp.maximum(jnp.sum(anchor_mask), 1.0))
            anchor_scale = math.sqrt(max(float(getattr(self.cfg, "late_shock_anchor_weight", 10.0)), 0.0))
            parts.append(jnp.reshape(diff_anchor * anchor_scale / denom_a, (-1,)))
        if len(parts) == 0:
            return jnp.zeros((1,), dtype=jnp.float64)
        return jnp.concatenate(parts, axis=0)

    def fixed_batch_metrics(self, params, geom: Dict[str, Array], data_coords: Array, data_vals: Array, data_mask: Array, f_coords: Array, proposal_density: Optional[Array], shock_coords: Optional[Array] = None, anchor_coords: Optional[Array] = None, anchor_vals: Optional[Array] = None, anchor_mask: Optional[Array] = None) -> Dict[str, float]:
        mse_u, _ = self.data_loss_and_residuals(params, geom, data_coords, data_vals, data_mask)
        mse_f_weighted = jnp.array(0.0, dtype=jnp.float64)
        tail_loss = jnp.array(0.0, dtype=jnp.float64)
        mse_f_shock = jnp.array(0.0, dtype=jnp.float64)
        mse_anchor = jnp.array(0.0, dtype=jnp.float64)
        if f_coords.shape[0] > 0:
            r = self.residual_tensor(params, geom, f_coords)
            r2 = jnp.sum(r * r, axis=1)
            if proposal_density is None:
                mse_f_weighted = jnp.mean(r2)
            else:
                pbar = jnp.maximum(proposal_density.reshape(-1), 1e-12)
                mse_f_weighted = jnp.mean(r2 / (pbar * self.admissible_volume + 1e-12))
            thr = jnp.quantile(r2, 0.99)
            tail_loss = jnp.mean(r2[r2 >= thr])
        if shock_coords is not None and shock_coords.shape[0] > 0:
            r_shock = self.residual_tensor(params, geom, shock_coords)
            mse_f_shock = jnp.mean(jnp.sum(r_shock * r_shock, axis=1))
        if anchor_coords is not None and anchor_vals is not None and anchor_mask is not None and anchor_coords.shape[0] > 0:
            pred_anchor = model_forward(params, geom, anchor_coords, self.act_name)
            diff_anchor = (pred_anchor - anchor_vals) * anchor_mask
            denom_a = jnp.maximum(jnp.sum(anchor_mask), 1.0)
            mse_anchor = jnp.sum(diff_anchor * diff_anchor) / denom_a
        lam_shock = float(getattr(self.cfg, "late_shock_residual_weight", 6.0))
        lam_anchor = float(getattr(self.cfg, "late_shock_anchor_weight", 10.0))
        theta_obj = mse_u + mse_f_weighted + lam_shock * mse_f_shock + lam_anchor * mse_anchor
        return {
            "theta_obj": float(theta_obj),
            "loss_total": float(theta_obj),
            "mse_u": float(mse_u),
            "mse_f_weighted": float(mse_f_weighted),
            "mse_f_shock": float(mse_f_shock),
            "mse_anchor": float(mse_anchor),
            "tail_loss": float(tail_loss),
        }

    def _draw_dd_balls_overlay(self, ax, geom: Dict[str, Array], extent=None):
        centers = np.asarray(geom["centers"])
        radii = np.asarray(geom["radii"])
        G_mats = np.asarray(geom["G_mats"])
        if centers.shape[1] < 2:
            return
        ax.scatter(centers[:, 0], centers[:, 1], s=12, c="white", edgecolors="white", linewidths=0.4, zorder=4, clip_on=True)
        for j in range(centers.shape[0]):
            G2 = G_mats[j][:2, :2]
            vals, vecs = np.linalg.eigh(0.5 * (G2 + G2.T))
            vals = np.clip(vals, 1e-12, None)
            width = 2.0 * float(radii[j]) / math.sqrt(float(vals[0]))
            height = 2.0 * float(radii[j]) / math.sqrt(float(vals[1]))
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            ell = matplotlib.patches.Ellipse(
                (float(centers[j, 0]), float(centers[j, 1])),
                width=width,
                height=height,
                angle=angle,
                fill=False,
                color="white",
                linewidth=0.7,
                alpha=0.9,
            )
            ell.set_clip_path(ax.patch)
            ax.add_patch(ell)
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.margins(x=0.0, y=0.0)

    # ---------------------------------------------------------
    # theta updates
    # ---------------------------------------------------------
    def _hf_clip_damping(self, damping: float) -> float:
        return max(float(damping), float(self.cfg.hf_damping_min))

    def _scale_hf_step_to_cap(self, step_np: NpArray) -> Tuple[NpArray, float, float, float]:
        raw_step_norm = float(np.linalg.norm(step_np))
        cap = float(self.cfg.hf_step_norm_cap)
        if cap > 0.0 and raw_step_norm > cap:
            scale = cap / max(raw_step_norm, 1e-12)
            step_np = step_np * scale
            step_norm = float(np.linalg.norm(step_np))
        else:
            scale = 1.0
            step_norm = raw_step_norm
        return step_np, raw_step_norm, step_norm, scale


    def _hf_cg_solve(self, linop_fn, b: Array, tol_abs: float, cg_tol: float, cg_maxiter: int) -> Tuple[Array, Dict[str, Any]]:
        x, _ = jax_cg(
            linop_fn,
            b,
            tol=float(cg_tol),
            atol=float(tol_abs),
            maxiter=max(int(cg_maxiter), 1),
        )
        r = linop_fn(x) - b
        res_norm = float(np.asarray(jnp.linalg.norm(r), dtype=np.float64))
        info = {
            "cg_iters": int(max(int(cg_maxiter), 1)),
            "cg_res_norm": float(res_norm),
            "negative_curvature": 0,
            "cg_tol": float(cg_tol),
        }
        return x, info

    def theta_step_hf(
        self,
        params,
        geom: Dict[str, Array],
        data_coords_np: NpArray,
        data_vals_np: NpArray,
        data_mask_np: NpArray,
        f_coords_np: NpArray,
        proposal_density_np: Optional[NpArray] = None,
        shock_coords_np: Optional[NpArray] = None,
        anchor_coords_np: Optional[NpArray] = None,
        anchor_vals_np: Optional[NpArray] = None,
        anchor_mask_np: Optional[NpArray] = None,
    ):
        data_coords = jax.device_put(jnp.asarray(data_coords_np), self.jax_device)
        data_vals = jax.device_put(jnp.asarray(data_vals_np), self.jax_device)
        data_mask = jax.device_put(jnp.asarray(data_mask_np), self.jax_device)
        f_coords = jax.device_put(jnp.asarray(f_coords_np), self.jax_device)
        proposal_density = None if proposal_density_np is None else jax.device_put(jnp.asarray(proposal_density_np), self.jax_device)
        shock_coords = None if shock_coords_np is None else jax.device_put(jnp.asarray(shock_coords_np), self.jax_device)
        anchor_coords = None if anchor_coords_np is None else jax.device_put(jnp.asarray(anchor_coords_np), self.jax_device)
        anchor_vals = None if anchor_vals_np is None else jax.device_put(jnp.asarray(anchor_vals_np), self.jax_device)
        anchor_mask = None if anchor_mask_np is None else jax.device_put(jnp.asarray(anchor_mask_np), self.jax_device)

        params = jax.device_put(params, self.jax_device)
        geom = jax.device_put(geom, self.jax_device)

        theta0, unravel = ravel_pytree(params)
        theta0 = jax.device_put(jnp.asarray(theta0), self.jax_device)
        theta0_np = np.asarray(theta0, dtype=np.float64)

        def residual_from_flat(theta_flat: Array) -> Array:
            params_trial = unravel(theta_flat)
            return self.global_residual_vector(
                params_trial, geom, data_coords, data_vals, data_mask, f_coords, proposal_density, shock_coords, anchor_coords, anchor_vals, anchor_mask
            )

        residual_only = jax.jit(residual_from_flat)
        value_only = jax.jit(lambda th: jnp.vdot(residual_from_flat(th), residual_from_flat(th)))

        e0 = residual_only(theta0)
        obj_old = float(np.asarray(jnp.vdot(e0, e0), dtype=np.float64))
        _, vjp_res = jax.vjp(residual_from_flat, theta0)
        jt_e = vjp_res(e0)[0]
        g0 = 2.0 * jt_e
        g0_np = np.asarray(g0, dtype=np.float64)
        jt_e_np = np.asarray(jt_e, dtype=np.float64)
        grad_norm = float(np.linalg.norm(g0_np))

        if (not np.isfinite(obj_old)) or (not np.all(np.isfinite(g0_np))):
            self.last_theta_step_info = {
                "status": "rollback:nonfinite_old_obj",
                "optimizer": "hf_jtj",
                "obj_old": float(obj_old),
                "obj_trial": float(obj_old),
                "step_norm": 0.0,
                "actual_reduction": 0.0,
                "rel_reduction": 0.0,
                "grad_norm": float(grad_norm),
                "cg_iters": 0,
                "cg_res_norm": float("nan"),
                "ls_iters": 0,
                "damping": float(self.hf_damping),
            }
            return params

        if grad_norm <= 1e-14:
            self.last_theta_step_info = {
                "status": "skipped:tiny_grad",
                "optimizer": "hf_jtj",
                "obj_old": float(obj_old),
                "obj_trial": float(obj_old),
                "step_norm": 0.0,
                "actual_reduction": 0.0,
                "rel_reduction": 0.0,
                "grad_norm": float(grad_norm),
                "cg_iters": 0,
                "cg_res_norm": 0.0,
                "ls_iters": 0,
                "damping": float(self.hf_damping),
            }
            return params

        e0_lin, jvp_res = jax.linearize(residual_from_flat, theta0)
        damping = self._hf_clip_damping(float(self.hf_damping))

        rel_gate = float(min(getattr(self, "best_relL2", float("inf")), getattr(self, "latest_relL2", float("inf"))))
        cg_tol = float(self.cfg.hf_cg_tol)
        cg_maxiter = int(self.cfg.hf_cg_maxiter)
        if rel_gate <= 0.15:
            cg_tol = min(cg_tol, 5e-4)
            cg_maxiter = max(cg_maxiter, 100)
        if rel_gate <= 0.05:
            cg_tol = min(cg_tol, 1e-4)
            cg_maxiter = max(cg_maxiter, 160)

        accepted = False
        params_trial = params
        obj_trial = obj_old
        step_norm = 0.0
        raw_step_norm = 0.0
        actual_reduction = 0.0
        rel_reduction = 0.0
        ls_iters_used = 0
        cg_info = {"cg_iters": 0, "cg_res_norm": float("nan"), "negative_curvature": 0}
        status = "rollback:not_improved"
        trials_used = 0

        for trial in range(1, max(int(self.cfg.hf_max_trials), 1) + 1):
            trials_used = trial
            damping_arr = jax.device_put(jnp.asarray(damping, dtype=theta0.dtype), self.jax_device)

            def linop_fn(v: Array) -> Array:
                jv = jvp_res(v)
                return vjp_res(jv)[0] + damping_arr * v

            tol_abs = max(float(cg_tol) * max(np.linalg.norm(jt_e_np), 1e-12), 1e-10)
            step, cg_info = self._hf_cg_solve(linop_fn, -jt_e, tol_abs=tol_abs, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
            step_np = np.asarray(step, dtype=np.float64)

            if not np.all(np.isfinite(step_np)):
                step_np = -g0_np
                cg_info["negative_curvature"] = 1

            directional_deriv = float(np.dot(g0_np, step_np))
            if (not np.isfinite(directional_deriv)) or directional_deriv >= 0.0:
                step_np = -g0_np
                directional_deriv = -float(np.dot(g0_np, g0_np))

            step_np, raw_step_norm, step_norm, _ = self._scale_hf_step_to_cap(step_np)
            step_dev = jax.device_put(jnp.asarray(step_np), self.jax_device)

            accepted_ls = False
            alpha = 1.0
            obj_try = obj_old
            for ls in range(1, max(int(self.cfg.hf_line_search_maxiter), 1) + 1):
                theta_try = theta0 + alpha * step_dev
                obj_try = float(np.asarray(value_only(theta_try), dtype=np.float64))
                if np.isfinite(obj_try) and obj_try <= obj_old + float(self.cfg.hf_line_search_c1) * alpha * directional_deriv:
                    accepted_ls = True
                    ls_iters_used = ls
                    theta_accept = theta_try
                    break
                alpha *= float(self.cfg.hf_line_search_tau)

            if accepted_ls:
                obj_trial = obj_try
                actual_reduction = float(obj_old - obj_trial)
                rel_reduction = actual_reduction / max(abs(obj_old), 1e-12)
                params_trial = unravel(theta_accept)
                params_trial = jax.device_put(params_trial, self.jax_device)
                damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_down))
                accepted = True
                status = "applied:hf_jtj" if actual_reduction >= 0.0 else "applied:hf_jtj_worse"
                break

            damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_up))
            status = "rollback:not_improved"

        self.hf_damping = float(damping)
        self.last_theta_step_info = {
            "status": status,
            "optimizer": "hf_jtj",
            "obj_old": float(obj_old),
            "obj_trial": float(obj_trial),
            "step_norm": float(step_norm if accepted else 0.0),
            "raw_step_norm": float(raw_step_norm),
            "actual_reduction": float(actual_reduction),
            "rel_reduction": float(rel_reduction),
            "grad_norm": float(grad_norm),
            "cg_iters": int(cg_info.get("cg_iters", 0)),
            "cg_res_norm": float(cg_info.get("cg_res_norm", float("nan"))),
            "neg_curv": int(cg_info.get("negative_curvature", 0)),
            "ls_iters": int(ls_iters_used),
            "damping": float(self.hf_damping),
            "trials": int(trials_used),
            "nit": int(cg_info.get("cg_iters", 0)),
            "funcalls": int(ls_iters_used + 1),
        }
        return params_trial if accepted else params
    def theta_step(
        self,
        params,
        geom: Dict[str, Array],
        data_coords_np: NpArray,
        data_vals_np: NpArray,
        data_mask_np: NpArray,
        f_coords_np: NpArray,
        proposal_density_np: Optional[NpArray] = None,
        shock_coords_np: Optional[NpArray] = None,
        anchor_coords_np: Optional[NpArray] = None,
        anchor_vals_np: Optional[NpArray] = None,
        anchor_mask_np: Optional[NpArray] = None,
        outer_iter: Optional[int] = None,
    ):
        return self.theta_step_hf(
            params, geom, data_coords_np, data_vals_np, data_mask_np, f_coords_np, proposal_density_np, shock_coords_np, anchor_coords_np, anchor_vals_np, anchor_mask_np
        )

    def sample_residual_batch(self, geom: Dict[str, Array]) -> ResidualBatch:
        b = int(self.cfg.batch_f if self.cfg.batch_f > 0 else self.cfg.n_r_pool)
        if self.p_grid is None:
            self.update_sampling_distribution(self.params, self.geom_state, int(getattr(self, "current_outer_iter", 0)))
        coords = self._sample_from_grid_density(self.p_grid, b)
        q_grid = self._proposal_density_from_grid(coords, self.p_grid).reshape(-1)
        self.last_theta_batch_info = {
            "uniform_mix_frac": 0.0,
            "adaptive_count": int(b),
            "uniform_count": 0,
            "warmup": 0,
            "shock_count": 0,
            "anchor_count": 0,
        }
        return ResidualBatch(coords=coords, proposal_density=q_grid.reshape(-1, 1))

    def _current_theta_schedule(self, outer_iter: int) -> Tuple[int, int]:
        theta_inner_steps = max(int(getattr(self.cfg, "theta_inner_steps", 1)), 1)
        rel_gate = self._rel_gate()

        if self._late_shock_objective_active():
            theta_inner_steps = max(theta_inner_steps, int(getattr(self.cfg, "late_shock_theta_inner_steps", 1)))
            return theta_inner_steps, 1

        if rel_gate <= 5e-2:
            block_len = 1
        elif rel_gate <= 0.15:
            block_len = 1
        elif rel_gate <= 0.30:
            block_len = 2
        else:
            block_len = max(int(getattr(self.cfg, "refresh_every", 1)), 1)

        if outer_iter >= int(getattr(self.cfg, "late_stage_refresh_iter", 250)) and rel_gate > 5e-2:
            block_len = min(block_len, 1 if rel_gate <= 0.20 else 2)

        return theta_inner_steps, block_len

    # ---------------------------------------------------------
    # train loop (PINN BALLS skeleton)
    # ---------------------------------------------------------
    def train(self, out_dir: Path) -> None:
        t0 = time.time()
        for it in range(1, self.cfg.iters + 1):
            iter_t0 = time.time()
            self.current_outer_iter = int(it)
            theta_inner_steps, block_len = self._current_theta_schedule(it)
            refresh_due = (self.cached_block is None) or ((it - 1) % block_len == 0)
            late_shock_active = self._late_shock_objective_active()
            late_shock_aux_active = self._late_shock_aux_loss_active()
            late_shock_freeze = self._late_shock_freeze_dd_active()

            if refresh_due:
                self.update_sampling_distribution(self.params, self.geom_state, it)
                if late_shock_freeze:
                    prev = dict(getattr(self, "last_dd_info", {}))
                    prev.update({
                        "status": "skipped:late_shock_freeze",
                        "freeze_mode": "late_shock_frozen",
                        "hardening_state": prev.get("hardening_state", "late_shock_frozen"),
                        "hardening_factor": float(getattr(self, "dd_hardening_factor", 0.35)),
                    })
                    self.last_dd_info = prev
                else:
                    self.dd_geometry_update(self.params, it)
                    self._update_dd_hardening()

                data_batch = self.sample_data_batch()
                residual_batch = self.sample_residual_batch(self.geom_state)
                if late_shock_aux_active:
                    shock_coords_np = self._sample_late_shock_residual_coords()
                    anchor_coords_np, anchor_vals_np, anchor_mask_np = self._sample_late_shock_anchor_batch(int(getattr(self.cfg, "late_shock_anchor_points", 64)))
                else:
                    shock_coords_np = np.empty((0, self.d_in), dtype=np.float64)
                    anchor_coords_np = np.empty((0, self.d_in), dtype=np.float64)
                    anchor_vals_np = np.empty((0, self.d_out), dtype=np.float64)
                    anchor_mask_np = np.empty((0, self.d_out), dtype=np.float64)
                self.cached_block = {
                    "geom": self.geom_state,
                    "data": data_batch,
                    "residual": residual_batch,
                    "shock_coords": shock_coords_np,
                    "anchor": (anchor_coords_np, anchor_vals_np, anchor_mask_np),
                }
                self.last_theta_batch_info["shock_count"] = int(shock_coords_np.shape[0])
                self.last_theta_batch_info["anchor_count"] = int(anchor_coords_np.shape[0])
                block_pos = 1
            else:
                block_pos = ((it - 1) % block_len) + 1

            self.last_refresh_info = {
                "refreshed": int(refresh_due),
                "block_pos": int(block_pos),
                "block_len": int(block_len),
                "theta_inner_steps": int(theta_inner_steps),
            }

            geom_train = self.cached_block["geom"]
            data_coords_np, data_vals_np, data_mask_np = self.cached_block["data"]
            sampled = self.cached_block["residual"]
            shock_coords_np = self.cached_block.get("shock_coords", np.empty((0, self.d_in), dtype=np.float64))
            anchor_coords_np, anchor_vals_np, anchor_mask_np = self.cached_block.get("anchor", (np.empty((0, self.d_in), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64), np.empty((0, self.d_out), dtype=np.float64)))

            for inner in range(theta_inner_steps):
                self.block_theta_step = inner + 1
                self.params = self.theta_step(
                    self.params,
                    geom_train,
                    data_coords_np,
                    data_vals_np,
                    data_mask_np,
                    sampled.coords,
                    sampled.proposal_density,
                    shock_coords_np=shock_coords_np,
                    anchor_coords_np=anchor_coords_np,
                    anchor_vals_np=anchor_vals_np,
                    anchor_mask_np=anchor_mask_np,
                    outer_iter=it,
                )

            geom_after = self.geom_state

            should_eval_rel = (
                (it == 1)
                or (it == self.cfg.iters)
                or (self.cfg.rel_l2_eval_every > 0 and it % self.cfg.rel_l2_eval_every == 0)
            )
            should_log = (
                (it % self.cfg.print_every == 0)
                or (it == 1)
                or (it == self.cfg.iters)
            )
            need_diagnostics = bool(should_eval_rel or should_log)

            post_update = None
            if need_diagnostics:
                data_coords = jnp.asarray(data_coords_np)
                data_vals = jnp.asarray(data_vals_np)
                data_mask = jnp.asarray(data_mask_np)
                f_coords = jnp.asarray(sampled.coords)
                proposal_density = jnp.asarray(sampled.proposal_density)
                shock_coords = jnp.asarray(shock_coords_np) if shock_coords_np.shape[0] > 0 else None
                anchor_coords = jnp.asarray(anchor_coords_np) if anchor_coords_np.shape[0] > 0 else None
                anchor_vals = jnp.asarray(anchor_vals_np) if anchor_vals_np.shape[0] > 0 else None
                anchor_mask = jnp.asarray(anchor_mask_np) if anchor_mask_np.shape[0] > 0 else None
                post_update = self.fixed_batch_metrics(
                    self.params, geom_after, data_coords, data_vals, data_mask, f_coords, proposal_density, shock_coords, anchor_coords, anchor_vals, anchor_mask
                )

            rel_l2_epoch = float("nan")
            snapshot = None
            improved = False
            if should_eval_rel:
                rel_l2_epoch, snapshot = self.pde.eval_rel_l2(
                    self.params, geom_after, self.predict_batched, self.cfg.test_batch_size
                )
                self.latest_relL2 = float(rel_l2_epoch)
                improved = bool(rel_l2_epoch < self.best_relL2)
                if improved:
                    self.best_relL2 = rel_l2_epoch
                    self.best_relL2_iter = it
                    self.best_params = self.params
                    self.best_geom_state = self.geom_state
                    self.best_eval_snapshot = snapshot
                    loss_total_for_plot = float("nan") if post_update is None else post_update["loss_total"]
                    self.save_best_snapshot_plot(
                        out_dir / "result_ffusion_bur", snapshot, geom_after, rel_l2_epoch, it, loss_total_for_plot
                    )
                self._maybe_advance_dd_freeze(improved, rel_l2_epoch)

            self._update_feedback_controller()

            if should_log:
                if post_update is None:
                    post_update = {
                        "theta_obj": float("nan"),
                        "loss_total": float("nan"),
                        "mse_u": float("nan"),
                        "mse_f_weighted": float("nan"),
                        "mse_f_shock": float("nan"),
                        "mse_anchor": float("nan"),
                        "tail_loss": float("nan"),
                    }

                theta_info = self.last_theta_step_info
                samp_info = self.last_sampling_info
                dd_info = self.last_dd_info
                elapsed = time.time() - t0
                eta_txt = base.estimate_eta(t0, it, self.cfg.iters)

                base.log(
                    f"[ITER {it:6d}/{self.cfg.iters}] "
                    f"loss_total={post_update['loss_total']:.3e}  "
                    f"theta_obj={post_update['theta_obj']:.3e}  "
                    f"mse_u={post_update['mse_u']:.3e}  mse_f_weighted={post_update['mse_f_weighted']:.3e}  mse_f_shock={post_update.get('mse_f_shock', float('nan')):.3e}  mse_anchor={post_update.get('mse_anchor', float('nan')):.3e}  tail_loss={post_update['tail_loss']:.3e}  "
                    f"relL2={rel_l2_epoch:.3e}  "
                    f"theta={theta_info.get('status', 'na')}  sample={samp_info.get('status', 'na')}[{samp_info.get('score_type', 'na')}]  dd={dd_info.get('status', 'na')}[{dd_info.get('hardening_state', 'na')}|{dd_info.get('freeze_mode', self.dd_freeze_mode)}]  "
                    f"theta_opt=hf_jtj_gpu  refresh={self.last_refresh_info.get('refreshed', 0)}  block={self.last_refresh_info.get('block_pos', 1)}/{self.last_refresh_info.get('block_len', 1)}  theta_inner={self.last_refresh_info.get('theta_inner_steps', 1)}  shock_n={self.last_theta_batch_info.get('shock_count', 0)}  anchor_n={self.last_theta_batch_info.get('anchor_count', 0)}  diff_steps={samp_info.get('steps', 0)}  "
                    f"top5={samp_info.get('topk_overlap', float('nan')):.3f}  shock_mass={samp_info.get('shock_mass', float('nan')):.3f}  px0={samp_info.get('particle_x0_frac', float('nan')):.3f}  c_hw={samp_info.get('cover_halfwidth', float('nan')):.3e}  wmis={samp_info.get('ridge_width_mismatch', float('nan')):.3e}  peakr={samp_info.get('shock_peak_ratio', float('nan')):.3f}  shock_mix={samp_info.get('shock_prior_mix', float('nan')):.3f}  "
                    f"grid_pocc_l1={samp_info.get('grid_particle_l1', float('nan')):.3e}  grid_ptau_l1={samp_info.get('grid_explicit_l1', float('nan')):.3e}  "
                    f"alpha={dd_info.get('alpha', float('nan')):.3e}  mean_r={dd_info.get('mean_radius', float('nan')):.3e}  hard={dd_info.get('hardening_factor', float('nan')):.3f}  "
                    f"q={dd_info.get('radius_quantile', float('nan')):.2f}  margin={dd_info.get('radius_margin', float('nan')):.2f}  tau_r={dd_info.get('radius_smooth_tau', float('nan')):.4f}  wp={dd_info.get('weight_power_eff', float('nan')):.3f}  kappa={dd_info.get('residual_kappa_eff', float('nan')):.3f}  "
                    f"tau_c={dd_info.get('center_smooth_tau', float('nan')):.4f}  tau_g={dd_info.get('metric_smooth_tau', float('nan')):.4f}  "
                    f"axis_min={dd_info.get('axis_min_eff', float('nan')):.3e}  axis_max={dd_info.get('axis_max_eff', float('nan')):.3e}  cond_max={dd_info.get('cond_max_eff', float('nan')):.3e}  ridge_dx={dd_info.get('ridge_dx', float('nan')):.3e}  uncover={dd_info.get('uncovered_frac', float('nan')):.3e}  drift={dd_info.get('geometry_drift', float('nan')):.3e}  "
                    f"iter_time={base.format_seconds(time.time() - iter_t0)}  total={base.format_seconds(elapsed)}  eta={eta_txt}"
                )
        base.log(f"[TRAIN-SUMMARY] best_relL2(iter={self.best_relL2_iter})={self.best_relL2:.3e}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pde", type=str, default="")
    p.add_argument("--seed", type=int, default="")
    p.add_argument("--paper_repro", action="store_true")
    p.add_argument("--nic", type=int, default="")
    p.add_argument("--nbc", type=int, default="")
    p.add_argument("--nf", type=int, default="")
    p.add_argument("--batch_f", type=int, default="")
    p.add_argument("--batch_u", type=int, default="")
    p.add_argument("--n_balls", type=int, default="")
    p.add_argument("--layers", type=int, default="")
    p.add_argument("--width", type=int, default="")
    p.add_argument("--act", type=str, default="", choices=["tanh", "silu", "relu"])
    p.add_argument("--init_radius_scale", type=float, default="")
    p.add_argument("--dd_min_radius_scale", type=float, default="")
    p.add_argument("--dd_init_kmeans_points", type=int, default="")
    p.add_argument("--dd_kmeans_iters", type=int, default="")
    p.add_argument("--iters", type=int, default="")

    # theta update: PINN BALLS style only
    p.add_argument("--theta_optimizer", type=str, default="", choices=["hf"])
    p.add_argument("--hf_damping_init", type=float, default="")
    p.add_argument("--hf_damping_min", type=float, default="")
    p.add_argument("--hf_damping_up", type=float, default="")
    p.add_argument("--hf_damping_down", type=float, default="")
    p.add_argument("--hf_cg_tol", type=float, default="")
    p.add_argument("--hf_cg_maxiter", type=int, default="")
    p.add_argument("--hf_max_trials", type=int, default="")
    p.add_argument("--hf_line_search_c1", type=float, default="")
    p.add_argument("--hf_line_search_tau", type=float, default="")
    p.add_argument("--hf_line_search_maxiter", type=int, default="")
    p.add_argument("--hf_step_norm_cap", type=float, default="")
    p.add_argument("--theta_inner_steps", type=int, default="")
    p.add_argument("--refresh_every", type=int, default="")
    p.add_argument("--late_stage_refresh_rel_l2", type=float, default="")
    p.add_argument("--late_stage_refresh_iter", type=int, default="")

    p.add_argument("--rel_l2_eval_every", type=int, default="")
    p.add_argument("--print_every", type=int, default="")
    p.add_argument("--test_batch_size", type=int, default="")
    p.add_argument("--gt_burgers", type=str, default="")
    p.add_argument("--gt_helmholtz", type=str, default="")
    p.add_argument("--gt_navier", type=str, default="")
    p.add_argument("--log_file", type=str, default="")

    # diffusion-based adaptive sampling
    p.add_argument("--diffusion_particles", type=int, default="")
    p.add_argument("--diffusion_dt", type=float, default="")
    p.add_argument("--diffusion_time_scale", type=float, default="")
    p.add_argument("--diffusion_grid_n1", type=int, default="")
    p.add_argument("--diffusion_grid_n2", type=int, default="")
    p.add_argument("--diffusion_grid_n3", type=int, default="")
    p.add_argument("--diffusion_grid_jitter", type=float, default="")
    p.add_argument("--diffusion_steps", type=int, default="")
    p.add_argument("--residual_focus_power", type=float, default="")
    p.add_argument("--residual_focus_smoothing_sigma", type=float, default="")
    p.add_argument("--phi_topk_quantile", type=float, default="")
    p.add_argument("--phi_shock_band_halfwidth_scale", type=float, default="")
    p.add_argument("--feedback_ridge_search_thresh", type=float, default="")
    p.add_argument("--feedback_ridge_lock_thresh", type=float, default="")
    p.add_argument("--feedback_shock_mass_low", type=float, default="")
    p.add_argument("--feedback_shock_mass_high", type=float, default="")
    p.add_argument("--feedback_shock_mass_lock_min", type=float, default="")
    p.add_argument("--feedback_shock_mass_release", type=float, default="")
    p.add_argument("--feedback_cover_halfwidth_lock_max", type=float, default="")
    p.add_argument("--feedback_cover_halfwidth_release", type=float, default="")
    p.add_argument("--feedback_ridge_width_mismatch_lock_max", type=float, default="")
    p.add_argument("--feedback_ridge_width_mismatch_release", type=float, default="")
    p.add_argument("--feedback_ema", type=float, default="")
    p.add_argument("--feedback_shock_prior_mix_search", type=float, default="")
    p.add_argument("--feedback_shock_prior_mix_overfocus", type=float, default="")
    p.add_argument("--feedback_shock_prior_mix_micro_frozen", type=float, default="")
    p.add_argument("--feedback_residual_focus_power_micro_frozen", type=float, default="")

    # anisotropic Voronoi DD
    p.add_argument("--dd_lloyd_steps", type=int, default="")
    p.add_argument("--dd_warmup_iters", type=int, default="")
    p.add_argument("--dd_residual_kappa", type=float, default="")
    p.add_argument("--dd_cov_eps", type=float, default="")
    p.add_argument("--dd_eig_min", type=float, default="")
    p.add_argument("--dd_eig_max", type=float, default="")
    p.add_argument("--dd_cond_max", type=float, default="")
    p.add_argument("--dd_axis_min_scale", type=float, default="")
    p.add_argument("--dd_axis_max_scale", type=float, default="")
    p.add_argument("--dd_radius_smooth_tau", type=float, default="")
    p.add_argument("--dd_radius_growth_cap_rel", type=float, default="")
    p.add_argument("--dd_radius_margin", type=float, default="")
    p.add_argument("--dd_radius_quantile", type=float, default="")
    p.add_argument("--dd_weight_power", type=float, default="")
    p.add_argument("--dd_center_smooth_tau", type=float, default="")
    p.add_argument("--dd_metric_smooth_tau", type=float, default="")
    p.add_argument("--dd_cover_margin", type=float, default="")
    p.add_argument("--dd_cover_mix_frac", type=float, default="")
    p.add_argument("--dd_cover_weight", type=float, default="")
    p.add_argument("--dd_empty_keep_old", action="store_true")
    p.add_argument("--no_dd_empty_keep_old", action="store_true")
    p.add_argument("--dd_freeze_enable", action="store_true")
    p.add_argument("--no_dd_freeze_enable", action="store_true")
    p.add_argument("--dd_semi_freeze_rel_l2", type=float, default="")
    p.add_argument("--dd_semi_freeze_patience", type=int, default="")
    p.add_argument("--dd_full_freeze_rel_l2", type=float, default="")
    p.add_argument("--dd_full_freeze_patience", type=int, default="")
    p.add_argument("--dd_micro_freeze_ridge_dx", type=float, default="")
    p.add_argument("--dd_micro_freeze_min_iter", type=int, default="")
    p.add_argument("--dd_semi_radius_tau", type=float, default="")
    p.add_argument("--dd_semi_radius_growth_cap_rel", type=float, default="")
    p.add_argument("--dd_freeze_improve_tol", type=float, default="")
    p.add_argument("--dd_freeze_cond_thresh", type=float, default="")
    p.add_argument("--dd_micro_center_tau", type=float, default="")
    p.add_argument("--dd_micro_metric_tau", type=float, default="")
    p.add_argument("--dd_micro_radius_tau", type=float, default="")
    p.add_argument("--dd_micro_radius_growth_cap_rel", type=float, default="")
    p.add_argument("--dd_micro_radius_quantile", type=float, default="")
    p.add_argument("--dd_micro_axis_max_scale", type=float, default="")
    p.add_argument("--dd_micro_update_every", type=int, default="")
    p.add_argument("--late_shock_rel_l2", type=float, default="")
    p.add_argument("--late_shock_plateau_patience", type=int, default="")
    p.add_argument("--late_shock_freeze_ridge_dx", type=float, default="")
    p.add_argument("--late_shock_freeze_shock_mass_min", type=float, default="")
    p.add_argument("--late_shock_freeze_peakr_min", type=float, default="")
    p.add_argument("--late_shock_theta_inner_steps", type=int, default="")
    p.add_argument("--late_shock_refresh_every", type=int, default="")
    p.add_argument("--late_shock_residual_frac", type=float, default="")
    p.add_argument("--late_shock_residual_min", type=int, default="")
    p.add_argument("--late_shock_residual_max", type=int, default="")
    p.add_argument("--late_shock_anchor_points", type=int, default="")
    p.add_argument("--late_shock_residual_weight", type=float, default="")
    p.add_argument("--late_shock_anchor_weight", type=float, default="")
    p.add_argument("--late_shock_band_halfwidth_scale", type=float, default="")
    p.add_argument("--late_shock_time_start_frac", type=float, default="")
    p.add_argument("--late_shock_freeze_dd", action="store_true")
    p.add_argument("--no_late_shock_freeze_dd", action="store_true")
    p.add_argument("--dd_weak_repair_margin", type=float, default="")
    p.add_argument("--dd_late_shrink_gain", type=float, default="")
    p.add_argument("--dd_micro_shrink_gain", type=float, default="")
    p.set_defaults(dd_empty_keep_old=True, dd_freeze_enable=False, late_shock_freeze_dd=True)
    return p.parse_args()


def apply_paper_repro_defaults(args: argparse.Namespace) -> argparse.Namespace:
    name = args.pde.lower().strip()
    if not args.paper_repro:
        return args
    if name in ("burgers", "burgers1d", "burgers_1d"):
        args.nic = 1000
        args.nbc = 2000
        args.nf = 10000
        args.batch_f = 10000
        args.batch_u = 3000
        args.layers = 8
        args.width = 12
        args.iters = 2000
        args.dd_min_radius_scale = 0.25
    elif name in ("helmholtz", "helmholtz2d", "helmholtz_2d"):
        args.nic = 0
        args.nbc = 3000
        args.nf = 10000
        args.batch_f = 10000
        args.batch_u = 3000
        args.layers = 3
        args.width = 10
        args.iters = 2000
        args.dd_min_radius_scale = 0.35
    elif name in ("navier_stokes", "navier-stokes", "ns", "ns2d", "navier_stokes_2d"):
        args.nic = 5000
        args.nbc = 15000
        args.nf = 500000
        args.batch_f = 10000
        args.batch_u = 5000
        args.layers = 3
        args.width = 10
        args.iters = 2000
        args.dd_min_radius_scale = 0.35
    return args


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args = apply_paper_repro_defaults(args)
    args.theta_optimizer = "hf"

    name = args.pde.lower().strip()
    if name in ("burgers", "burgers1d", "burgers_1d"):
        args.diffusion_grid_n1 = max(int(getattr(args, "diffusion_grid_n1", 96)), 96)
        args.diffusion_grid_n2 = max(int(getattr(args, "diffusion_grid_n2", 192)), 192)
    elif name in ("helmholtz", "helmholtz2d", "helmholtz_2d"):
        args.diffusion_grid_n1 = max(int(getattr(args, "diffusion_grid_n1", 128)), 128)
        args.diffusion_grid_n2 = max(int(getattr(args, "diffusion_grid_n2", 128)), 128)
    elif name in ("navier_stokes", "navier-stokes", "ns", "ns2d", "navier_stokes_2d"):
        args.diffusion_grid_n1 = max(int(getattr(args, "diffusion_grid_n1", 24)), 24)
        args.diffusion_grid_n2 = max(int(getattr(args, "diffusion_grid_n2", 64)), 64)
        args.diffusion_grid_n3 = max(int(getattr(args, "diffusion_grid_n3", 24)), 24)
    if getattr(args, "no_dd_empty_keep_old", False):
        args.dd_empty_keep_old = False
    elif getattr(args, "dd_empty_keep_old", False):
        args.dd_empty_keep_old = True
    else:
        args.dd_empty_keep_old = True
    if getattr(args, "no_late_shock_freeze_dd", False):
        args.late_shock_freeze_dd = False
    elif getattr(args, "late_shock_freeze_dd", False):
        args.late_shock_freeze_dd = True
    else:
        args.late_shock_freeze_dd = True
    return args


def main() -> None:
    args = parse_args()
    args = apply_defaults(args)
    log_handle = base.setup_logging(args.log_file)
    try:
        rng = base.set_seed(args.seed)
        pde = make_pde_spec(args.pde, args.gt_burgers, args.gt_helmholtz, args.gt_navier)
        cfg = DiffusionGeometryConfig(
            n_ic=args.nic,
            n_bc=args.nbc,
            n_r_pool=args.nf,
            batch_f=args.batch_f,
            batch_u=args.batch_u,
            iters=args.iters,
            n_balls=args.n_balls,
            layers=args.layers,
            width=args.width,
            act=args.act,
            init_radius_scale=args.init_radius_scale,
            dd_min_radius_scale=args.dd_min_radius_scale,
            dd_init_kmeans_points=args.dd_init_kmeans_points,
            dd_kmeans_iters=args.dd_kmeans_iters,
            theta_optimizer=args.theta_optimizer,
            hf_damping_init=args.hf_damping_init,
            hf_damping_min=args.hf_damping_min,
            hf_damping_up=args.hf_damping_up,
            hf_damping_down=args.hf_damping_down,
            hf_cg_tol=args.hf_cg_tol,
            hf_cg_maxiter=args.hf_cg_maxiter,
            hf_max_trials=args.hf_max_trials,
            hf_line_search_c1=args.hf_line_search_c1,
            hf_line_search_tau=args.hf_line_search_tau,
            hf_line_search_maxiter=args.hf_line_search_maxiter,
            hf_step_norm_cap=args.hf_step_norm_cap,
            theta_inner_steps=args.theta_inner_steps,
            refresh_every=args.refresh_every,
            late_stage_refresh_rel_l2=args.late_stage_refresh_rel_l2,
            late_stage_refresh_iter=args.late_stage_refresh_iter,
            rel_l2_eval_every=args.rel_l2_eval_every,
            print_every=args.print_every,
            test_batch_size=args.test_batch_size,
            paper_repro=bool(args.paper_repro),
            diffusion_particles=args.diffusion_particles,
            diffusion_dt=args.diffusion_dt,
            diffusion_time_scale=args.diffusion_time_scale,
            diffusion_grid_n1=args.diffusion_grid_n1,
            diffusion_grid_n2=args.diffusion_grid_n2,
            diffusion_grid_n3=args.diffusion_grid_n3,
            diffusion_grid_jitter=args.diffusion_grid_jitter,
            diffusion_steps=args.diffusion_steps,
            residual_focus_power=args.residual_focus_power,
            residual_focus_smoothing_sigma=args.residual_focus_smoothing_sigma,
            phi_topk_quantile=args.phi_topk_quantile,
            phi_shock_band_halfwidth_scale=args.phi_shock_band_halfwidth_scale,
            feedback_ridge_search_thresh=args.feedback_ridge_search_thresh,
            feedback_ridge_lock_thresh=args.feedback_ridge_lock_thresh,
            feedback_shock_mass_low=args.feedback_shock_mass_low,
            feedback_shock_mass_high=args.feedback_shock_mass_high,
            feedback_shock_mass_lock_min=args.feedback_shock_mass_lock_min,
            feedback_shock_mass_release=args.feedback_shock_mass_release,
            feedback_cover_halfwidth_lock_max=args.feedback_cover_halfwidth_lock_max,
            feedback_cover_halfwidth_release=args.feedback_cover_halfwidth_release,
            feedback_ridge_width_mismatch_lock_max=args.feedback_ridge_width_mismatch_lock_max,
            feedback_ridge_width_mismatch_release=args.feedback_ridge_width_mismatch_release,
            feedback_ema=args.feedback_ema,
            feedback_shock_prior_mix_search=args.feedback_shock_prior_mix_search,
            feedback_shock_prior_mix_overfocus=args.feedback_shock_prior_mix_overfocus,
            feedback_shock_prior_mix_micro_frozen=args.feedback_shock_prior_mix_micro_frozen,
            feedback_residual_focus_power_micro_frozen=args.feedback_residual_focus_power_micro_frozen,
            dd_lloyd_steps=args.dd_lloyd_steps,
            dd_warmup_iters=args.dd_warmup_iters,
            dd_residual_kappa=args.dd_residual_kappa,
            dd_cov_eps=args.dd_cov_eps,
            dd_eig_min=args.dd_eig_min,
            dd_eig_max=args.dd_eig_max,
            dd_cond_max=args.dd_cond_max,
            dd_axis_min_scale=args.dd_axis_min_scale,
            dd_axis_max_scale=args.dd_axis_max_scale,
            dd_radius_smooth_tau=args.dd_radius_smooth_tau,
            dd_radius_growth_cap_rel=args.dd_radius_growth_cap_rel,
            dd_radius_margin=args.dd_radius_margin,
            dd_radius_quantile=args.dd_radius_quantile,
            dd_weight_power=args.dd_weight_power,
            dd_center_smooth_tau=args.dd_center_smooth_tau,
            dd_metric_smooth_tau=args.dd_metric_smooth_tau,
            dd_cover_margin=args.dd_cover_margin,
            dd_cover_mix_frac=args.dd_cover_mix_frac,
            dd_cover_weight=args.dd_cover_weight,
            dd_empty_keep_old=args.dd_empty_keep_old,
            dd_weak_repair_margin=args.dd_weak_repair_margin,
            dd_late_shrink_gain=args.dd_late_shrink_gain,
            dd_micro_shrink_gain=args.dd_micro_shrink_gain,
            dd_freeze_enable=args.dd_freeze_enable,
            dd_semi_freeze_rel_l2=args.dd_semi_freeze_rel_l2,
            dd_semi_freeze_patience=args.dd_semi_freeze_patience,
            dd_full_freeze_rel_l2=args.dd_full_freeze_rel_l2,
            dd_full_freeze_patience=args.dd_full_freeze_patience,
            dd_micro_freeze_ridge_dx=args.dd_micro_freeze_ridge_dx,
            dd_micro_freeze_min_iter=args.dd_micro_freeze_min_iter,
            dd_semi_radius_tau=args.dd_semi_radius_tau,
            dd_semi_radius_growth_cap_rel=args.dd_semi_radius_growth_cap_rel,
            dd_freeze_improve_tol=args.dd_freeze_improve_tol,
            dd_freeze_cond_thresh=args.dd_freeze_cond_thresh,
            dd_micro_center_tau=args.dd_micro_center_tau,
            dd_micro_metric_tau=args.dd_micro_metric_tau,
            dd_micro_radius_tau=args.dd_micro_radius_tau,
            dd_micro_radius_growth_cap_rel=args.dd_micro_radius_growth_cap_rel,
            dd_micro_radius_quantile=args.dd_micro_radius_quantile,
            dd_micro_axis_max_scale=args.dd_micro_axis_max_scale,
            dd_micro_update_every=args.dd_micro_update_every,
            late_shock_rel_l2=args.late_shock_rel_l2,
            late_shock_plateau_patience=args.late_shock_plateau_patience,
            late_shock_freeze_ridge_dx=args.late_shock_freeze_ridge_dx,
            late_shock_freeze_shock_mass_min=args.late_shock_freeze_shock_mass_min,
            late_shock_freeze_peakr_min=args.late_shock_freeze_peakr_min,
            late_shock_theta_inner_steps=args.late_shock_theta_inner_steps,
            late_shock_refresh_every=args.late_shock_refresh_every,
            late_shock_residual_frac=args.late_shock_residual_frac,
            late_shock_residual_min=args.late_shock_residual_min,
            late_shock_residual_max=args.late_shock_residual_max,
            late_shock_anchor_points=args.late_shock_anchor_points,
            late_shock_residual_weight=args.late_shock_residual_weight,
            late_shock_anchor_weight=args.late_shock_anchor_weight,
            late_shock_band_halfwidth_scale=args.late_shock_band_halfwidth_scale,
            late_shock_time_start_frac=args.late_shock_time_start_frac,
            late_shock_freeze_dd=args.late_shock_freeze_dd,
        )

        solver = PINNFFusionSolver(pde=pde, cfg=cfg, rng=rng)
        out_dir = Path("./PINN")
        out_dir.mkdir(parents=True, exist_ok=True)
        total_geom_params = solver.cfg.n_balls * (solver.d_in + 1 + solver.d_in * solver.d_in)
        total_params = solver.theta_size + total_geom_params

        


if __name__ == "__main__":
    main()