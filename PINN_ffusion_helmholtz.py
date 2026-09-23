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

from scipy import ndimage as sp_ndimage
from scipy.interpolate import RegularGridInterpolator

from PDE import make_pde as torch_make_pde

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg


Array = jnp.ndarray
NpArray = np.ndarray
BoundsType = Tuple[Tuple[float, float], ...]

DEFAULT_GT_HELMHOLTZ = ""


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


def require_jax_gpu() -> Any:
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if not gpus:
        raise RuntimeError("A JAX GPU backend is required.")
    return gpus[0]


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
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
            raise ValueError(f"Invalid bounds: {b}")
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


def _sync_torch_seed_from_rng(rng: np.random.Generator) -> None:
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)


def _to_numpy(x: torch.Tensor) -> NpArray:
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def bilinear_interp_np(qx: NpArray, qy: NpArray, grid_x: NpArray, grid_y: NpArray, values: NpArray) -> NpArray:
    qx = np.asarray(qx, dtype=np.float64).reshape(-1)
    qy = np.asarray(qy, dtype=np.float64).reshape(-1)
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    gy = np.asarray(grid_y, dtype=np.float64).reshape(-1)
    V = np.asarray(values, dtype=np.float64)

    qx = np.clip(qx, gx[0], gx[-1])
    qy = np.clip(qy, gy[0], gy[-1])

    ix = np.searchsorted(gx, qx, side="right")
    iy = np.searchsorted(gy, qy, side="right")
    ix = np.clip(ix, 1, gx.size - 1)
    iy = np.clip(iy, 1, gy.size - 1)

    x0 = gx[ix - 1]
    x1 = gx[ix]
    y0 = gy[iy - 1]
    y1 = gy[iy]

    wx = (qx - x0) / (x1 - x0 + 1e-12)
    wy = (qy - y0) / (y1 - y0 + 1e-12)

    v00 = V[ix - 1, iy - 1]
    v01 = V[ix - 1, iy]
    v10 = V[ix, iy - 1]
    v11 = V[ix, iy]

    out = (1.0 - wx) * (1.0 - wy) * v00 + (1.0 - wx) * wy * v01 + wx * (1.0 - wy) * v10 + wx * wy * v11
    return out.reshape(-1)


# ============================================================
# PDE adapter
# ============================================================
class HelmholtzPDEAdapter:
    def __init__(self, gt_helmholtz: str):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.torch_pde = torch_make_pde(
            "helmholtz_2d",
            gt_mat_path=gt_helmholtz,
            use_gt_bounds=True,
            device=self.device,
            dtype=self.dtype,
            use_lhs=True,
        )
        self.name = self.torch_pde.name
        if self.name != "helmholtz_2d":
            raise ValueError(f"Expected helmholtz_2d, got {self.name}")

        self.coord_kind = self.torch_pde.coord_kind
        self.d_in = self.torch_pde.d_in
        self.d_out = self.torch_pde.d_out
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in self.torch_pde.bounds)
        self.reaction_coeff = float(self.torch_pde.reaction_coeff)

        self.x_gt = _to_numpy(self.torch_pde._gt_x).reshape(-1)
        self.y_gt = _to_numpy(self.torch_pde._gt_y).reshape(-1)
        self.u_gt = _to_numpy(self.torch_pde._gt_u)
        self.f_gt = _to_numpy(self.torch_pde._gt_f)

        self.source_mode = "gt_grid"
        self.source_grid = self._build_source_grid()
        self.exact_mode = "gt_grid"

    def _call_grid_candidate(self, fn, X: NpArray, Y: NpArray) -> Optional[NpArray]:
        coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        coords_t = torch.tensor(coords, dtype=self.dtype, device=self.device)
        tries = []
        tries.append(lambda: fn(coords_t))
        tries.append(lambda: fn(coords_t[:, 0:1], coords_t[:, 1:2]))
        tries.append(lambda: fn(coords_t[:, 0], coords_t[:, 1]))
        tries.append(lambda: fn(torch.tensor(X, dtype=self.dtype), torch.tensor(Y, dtype=self.dtype)))
        for trial in tries:
            try:
                out = trial()
                if isinstance(out, tuple):
                    out = out[0]
                if torch.is_tensor(out):
                    out_np = _to_numpy(out).reshape(-1)
                    if out_np.size == coords.shape[0]:
                        return out_np.reshape(X.shape)
            except Exception:
                pass
        return None

    def _build_source_grid(self) -> NpArray:
        X, Y = meshgrid_ij_np(self.x_gt, self.y_gt)
        source_names = [
            "source_term",
            "forcing",
            "rhs",
            "source",
            "f_exact",
            "source_fn",
        ]
        for name in source_names:
            fn = getattr(self.torch_pde, name, None)
            if callable(fn):
                out = self._call_grid_candidate(fn, X, Y)
                if out is not None:
                    self.source_mode = "callable_grid"
                    return np.asarray(out, dtype=np.float64)

        # exact_solution-based source reconstruction is intentionally skipped
        # here to keep startup simple and robust. Fall back to GT forcing grid.
        self.source_mode = "gt_grid"
        return np.asarray(self.f_gt, dtype=np.float64)

    def exact_from_grid(self, coords: NpArray) -> NpArray:
        vals = bilinear_interp_np(coords[:, 0], coords[:, 1], self.x_gt, self.y_gt, self.u_gt)
        return vals.reshape(-1, 1)

    def source_from_grid(self, coords: NpArray) -> NpArray:
        vals = bilinear_interp_np(coords[:, 0], coords[:, 1], self.x_gt, self.y_gt, self.source_grid)
        return vals.reshape(-1, 1)

    def eval_rel_l2(self, params: Any, geom: Dict[str, Array], predict_fn, batch_size: int) -> Tuple[float, Dict[str, Any]]:
        X, Y = meshgrid_ij_np(self.x_gt, self.y_gt)
        coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        pred = predict_fn(params, geom, coords, batch_size)
        u_pred = pred.reshape(self.x_gt.size, self.y_gt.size, -1)[..., 0]
        rel = rel_l2_np(u_pred, self.u_gt)
        return rel, {
            "grid_coords": coords,
            "horizontal": self.x_gt,
            "vertical": self.y_gt,
            "pred_grid": u_pred,
            "true_grid": self.u_gt,
            "coord_kind": "xy",
            "h_label": "x",
            "v_label": "y",
            "aspect": "equal",
        }


# ============================================================
# geometry template
# ============================================================
def make_diag_metric_from_axes(ax: float, ay: float) -> NpArray:
    ax = max(float(ax), 1e-6)
    ay = max(float(ay), 1e-6)
    return np.diag(np.array([1.0 / (ax * ax), 1.0 / (ay * ay)], dtype=np.float64))


def cover_sum_np(coords: NpArray, centers: NpArray, radii: NpArray, G_mats: NpArray) -> NpArray:
    diff = coords[:, None, :] - centers[None, :, :]
    d2 = np.einsum("nmd,mde,nme->nm", diff, G_mats, diff)
    s = 1.0 - d2 / (radii[None, :] ** 2 + 1e-12)
    ph = np.maximum(s, 0.0) ** 2
    return np.sum(ph, axis=1)


def build_helmholtz_interface_template(bounds: BoundsType) -> Dict[str, NpArray]:
    (x0, x1), (y0, y1) = bounds
    Lx = x1 - x0
    Ly = y1 - y0
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    centers: List[NpArray] = []
    G_list: List[NpArray] = []
    radii: List[float] = []
    roles: List[int] = []

    # radii are all 1.0; axis lengths are encoded in G.
    r = 1.0

    # 0: global anchor, slightly reduced
    g_ax = 0.56 * Lx
    g_ay = 0.56 * Ly
    centers.append(np.array([cx, cy], dtype=np.float64))
    G_list.append(make_diag_metric_from_axes(g_ax, g_ay))
    radii.append(r)
    roles.append(0)

    # 1-4: overlapping quadrants, larger
    q_ax = 0.40 * Lx
    q_ay = 0.40 * Ly
    q_centers = [
        (x0 + 0.30 * Lx, y0 + 0.70 * Ly),
        (x0 + 0.70 * Lx, y0 + 0.70 * Ly),
        (x0 + 0.30 * Lx, y0 + 0.30 * Ly),
        (x0 + 0.70 * Lx, y0 + 0.30 * Ly),
    ]
    for c in q_centers:
        centers.append(np.array(c, dtype=np.float64))
        G_list.append(make_diag_metric_from_axes(q_ax, q_ay))
        radii.append(r)
        roles.append(1)

    # top band 5: valley ellipses, larger vertically and a bit horizontally
    xs = np.linspace(x0 + 0.11 * Lx, x1 - 0.11 * Lx, 5)
    top_y = y0 + 0.78 * Ly
    bot_y = y0 + 0.22 * Ly
    v_ax = 0.135 * Lx
    v_ay = 0.26 * Ly
    for x in xs:
        centers.append(np.array([x, top_y], dtype=np.float64))
        G_list.append(make_diag_metric_from_axes(v_ax, v_ay))
        radii.append(r)
        roles.append(2)
    for x in xs:
        centers.append(np.array([x, bot_y], dtype=np.float64))
        G_list.append(make_diag_metric_from_axes(v_ax, v_ay))
        radii.append(r)
        roles.append(3)

    # center interface/local 5: thinner vertically
    c_y = y0 + 0.50 * Ly
    c_ax = 0.128 * Lx
    c_ay = 0.105 * Ly
    for x in xs:
        centers.append(np.array([x, c_y], dtype=np.float64))
        G_list.append(make_diag_metric_from_axes(c_ax, c_ay))
        radii.append(r)
        roles.append(4)

    centers_np = np.stack(centers, axis=0)
    G_np = np.stack(G_list, axis=0)
    radii_np = np.asarray(radii, dtype=np.float64)
    roles_np = np.asarray(roles, dtype=np.int32)

    # Guarantee uncover = 0 by weak inflation if needed.
    gx = np.linspace(x0, x1, 121, dtype=np.float64)
    gy = np.linspace(y0, y1, 121, dtype=np.float64)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    probe = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    for _ in range(12):
        cover = cover_sum_np(probe, centers_np, radii_np, G_np)
        if np.all(cover > 0.0):
            break
        # Inflate non-global ellipses slightly first, then global if needed.
        for rid in [1, 2, 3, 4, 0]:
            mask = roles_np == rid
            if np.any(mask):
                G_np[mask] *= (1.0 / (1.04 ** 2))
        # do not touch radii; enlarge via axes in G

    cover = cover_sum_np(probe, centers_np, radii_np, G_np)
    if np.any(cover <= 0.0):
        raise RuntimeError("Failed to build a covering Helmholtz interface template.")

    return {
        "centers": centers_np,
        "G_mats": G_np,
        "radii": radii_np,
        "roles": roles_np,
    }


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


def make_expert_params(
    rng: np.random.Generator,
    d_in: int,
    d_out: int,
    hidden_layers: int,
    hidden_width: int,
) -> Tuple[Tuple[Array, Array], ...]:
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


def mahalanobis_d2(x: Array, centers: Array, G_mats: Array) -> Array:
    diff = x[:, None, :] - centers[None, :, :]
    return jnp.einsum("nmd,mde,nme->nm", diff, G_mats, diff)


def anisotropic_phi_basis(x: Array, geom: Dict[str, Array], eps: float = 1e-12) -> Array:
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
# configuration
# ============================================================
@dataclass
class StageSpec:
    grid_n: int
    particles: int
    n_tiles: int
    uniform_mix: float
    refresh_every: int
    cg_maxiter: int
    cg_tol: float
    focus_power: float
    focus_sigma: float
    anchor_weight: float
    ema_rho: float
    retain_frac: float
    weight_gamma: float
    weight_clip_min: float
    weight_clip_max: float
    tile_size: int
    residual_beta: float
    line_search_maxiter: int
    tile_lambda: float = ""
    stage_lambda: float = ""
    anchor_balance_ratio: float = ""
    anchor_aw_min: float = ""
    anchor_aw_max: float = ""
    anchor_aw_ema_tau: float = ""
    anchor_aw_change_limit: float = ""


@dataclass
class Config:
    # model
    n_balls: int = ""
    layers: int = ""
    width: int = ""
    act: str = ""
    freqs: Tuple[float, ...] = ""

    # training
    iters: int = ""
    theta_inner_steps: int = ""
    hf_damping_init: float = ""
    hf_damping_min: float = ""
    hf_damping_up: float = ""
    hf_damping_down: float = ""
    hf_cg_tol: float = ""
    hf_cg_maxiter_late: int = ""
    hf_max_trials: int = ""
    hf_line_search_c1: float = ""
    hf_line_search_tau: float = ""
    hf_line_search_maxiter: int = ""

    # sampler / PDE residual
    diffusion_dt: float = ""
    diffusion_time_scale: float = ""
    diffusion_steps: int = ""
    residual_focus_power: float = ""
    residual_focus_eps: float = ""
    residual_focus_smoothing_sigma: float = ""
    state_particle_tau_blend: float = ""  # explicit only
    tile_size: int = ""

    # anchor loss
    n_anchor_x: int = ""
    n_anchor_y: int = ""
    anchor_weight: float = ""

    # eval / plotting
    rel_l2_eval_every: int = ""
    monitor_eval_every: int = ""
    monitor_grid_n: int = ""
    print_every: int = ""
    test_batch_size: int = ""
    save_plot_every_best_delta: float = ""


# ============================================================
# solver
# ============================================================
class PINNFFusionHelmholtzSolver:
    def __init__(self, pde: HelmholtzPDEAdapter, cfg: Config, rng: np.random.Generator):
        self.pde = pde
        self.cfg = cfg
        self.rng = rng
        self.bounds = normalize_bounds(pde.bounds)
        self.d_in = pde.d_in
        self.d_out = pde.d_out
        self.freqs = jnp.asarray(np.asarray(cfg.freqs, dtype=np.float64))
        self.feature_dim = self.d_in * (1 + 2 * len(cfg.freqs))
        self.act_name = cfg.act

        self.mins_np = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.maxs_np = np.array([b[1] for b in self.bounds], dtype=np.float64)
        self.geom_np = build_helmholtz_interface_template(self.bounds)
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

        self.anchor_coords_np, self.anchor_vals_np = self._build_anchor_set()
        self.anchor_coords = jax.device_put(jnp.asarray(self.anchor_coords_np), self.jax_device)
        self.anchor_vals = jax.device_put(jnp.asarray(self.anchor_vals_np), self.jax_device)

        self.stage_specs = {
            "early": StageSpec(grid_n=48, particles=800, n_tiles=96, uniform_mix=0.20, refresh_every=1, cg_maxiter=32, cg_tol=1.0e-4, focus_power=1.0, focus_sigma=1.0, anchor_weight=100.0, ema_rho=0.20, retain_frac=0.50, weight_gamma=0.25, weight_clip_min=0.5, weight_clip_max=2.5, tile_size=7, residual_beta=0.20, line_search_maxiter=12, tile_lambda=0.25, stage_lambda=1.0, anchor_balance_ratio=0.05, anchor_aw_min=50.0, anchor_aw_max=2.0e4, anchor_aw_ema_tau=0.10, anchor_aw_change_limit=1.5),
            "mid": StageSpec(grid_n=96, particles=1600, n_tiles=160, uniform_mix=0.15, refresh_every=2, cg_maxiter=64, cg_tol=1.0e-4, focus_power=1.2, focus_sigma=0.9, anchor_weight=200.0, ema_rho=0.15, retain_frac=0.40, weight_gamma=0.25, weight_clip_min=0.5, weight_clip_max=2.5, tile_size=7, residual_beta=0.20, line_search_maxiter=12, tile_lambda=0.50, stage_lambda=1.0, anchor_balance_ratio=0.10, anchor_aw_min=50.0, anchor_aw_max=2.0e4, anchor_aw_ema_tau=0.10, anchor_aw_change_limit=1.5),
            "late": StageSpec(grid_n=128, particles=3200, n_tiles=320, uniform_mix=0.10, refresh_every=4, cg_maxiter=128, cg_tol=1.0e-6, focus_power=1.7, focus_sigma=0.7, anchor_weight=500.0, ema_rho=0.03, retain_frac=0.80, weight_gamma=0.35, weight_clip_min=0.5, weight_clip_max=2.5, tile_size=9, residual_beta=0.10, line_search_maxiter=16, tile_lambda=1.0, stage_lambda=1.0, anchor_balance_ratio=0.20, anchor_aw_min=50.0, anchor_aw_max=2.0e4, anchor_aw_ema_tau=0.08, anchor_aw_change_limit=1.5),
            "ultra": StageSpec(grid_n=192, particles=4800, n_tiles=512, uniform_mix=0.08, refresh_every=6, cg_maxiter=128, cg_tol=1.0e-6, focus_power=2.0, focus_sigma=0.6, anchor_weight=800.0, ema_rho=0.01, retain_frac=0.90, weight_gamma=0.50, weight_clip_min=0.5, weight_clip_max=2.5, tile_size=9, residual_beta=0.0, line_search_maxiter=16, tile_lambda=1.0, stage_lambda=2.0, anchor_balance_ratio=0.30, anchor_aw_min=100.0, anchor_aw_max=2.0e4, anchor_aw_ema_tau=0.05, anchor_aw_change_limit=1.5),
        }

        self.stage_cache: Dict[str, Dict[str, Any]] = {}
        self.current_stage_name = "early"
        self.current_stage = self._make_stage(self.stage_specs["early"])
        self.cached_batch: Optional[Dict[str, Any]] = None
        self.stage_density_state: Dict[str, NpArray] = {}
        self.prev_tile_centers_by_stage: Dict[str, NpArray] = {}
        self.monitor_cache: Optional[Dict[str, Any]] = None

        self.hf_damping = float(cfg.hf_damping_init)
        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.best_params = self.params
        self.best_snapshot = None
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_sampling_info: Dict[str, Any] = {}
        self.last_dd_info: Dict[str, Any] = {}
        self.last_refresh_info: Dict[str, Any] = {"refreshed": 1, "block_pos": 1, "block_len": 1}
        self.latest_relL2 = float("inf")
        self.latest_monitor_f = float("inf")
        self.last_anchor_balance_info: Dict[str, Any] = {
            "aw_old": float(self.current_stage["spec"].anchor_weight),
            "aw_target": float(self.current_stage["spec"].anchor_weight),
            "aw_new": float(self.current_stage["spec"].anchor_weight),
            "g_res": float("nan"),
            "g_anchor": float("nan"),
            "res_loss": float("nan"),
            "anchor_loss": float("nan"),
            "status": "init",
        }

    # -----------------------
    # template / anchors
    # -----------------------
    def _build_anchor_set(self) -> Tuple[NpArray, NpArray]:
        (x0, x1), (y0, y1) = self.bounds
        xs = np.linspace(x0 + 0.08 * (x1 - x0), x1 - 0.08 * (x1 - x0), self.cfg.n_anchor_x, dtype=np.float64)
        ys = np.linspace(y0 + 0.08 * (y1 - y0), y1 - 0.08 * (y1 - y0), self.cfg.n_anchor_y, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        vals = self.pde.exact_from_grid(coords)
        return coords, vals

    def _effective_axis_stats(self, geom_np: Dict[str, NpArray]) -> Tuple[float, float, float]:
        G = geom_np["G_mats"]
        radii = geom_np["radii"]
        vals_all = np.linalg.eigvalsh(0.5 * (G + np.transpose(G, (0, 2, 1))))
        vals_all = np.clip(vals_all, 1e-12, None)
        axes = radii[:, None] / np.sqrt(vals_all)
        cond = np.max(np.max(vals_all, axis=1) / np.maximum(np.min(vals_all, axis=1), 1e-12))
        return float(np.min(axes)), float(np.max(axes)), float(cond)

    # -----------------------
    # stage helpers
    # -----------------------
    def _stage_name(self, it: int) -> str:
        rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
        mon = float(self.latest_monitor_f) if np.isfinite(self.latest_monitor_f) else float("inf")
        if rel <= 0.08 or mon <= 1.0e4:
            return "ultra"
        if rel <= 0.25:
            return "late"
        if rel <= 0.75 or mon <= 1.0e6:
            return "mid"
        return "early"

    def _make_stage(self, spec: StageSpec) -> Dict[str, Any]:
        key = f"{spec.grid_n}_{spec.tile_size}"
        if key in self.stage_cache:
            return self.stage_cache[key]

        (x0, x1), (y0, y1) = self.bounds
        x = np.linspace(x0, x1, spec.grid_n, dtype=np.float64)
        y = np.linspace(y0, y1, spec.grid_n, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

        hx = float((x1 - x0) / max(spec.grid_n - 1, 1))
        hy = float((y1 - y0) / max(spec.grid_n - 1, 1))

        source_grid = bilinear_interp_np(coords[:, 0], coords[:, 1], self.pde.x_gt, self.pde.y_gt, self.pde.source_grid).reshape(spec.grid_n, spec.grid_n)

        tile_half = spec.tile_size // 2
        i_valid = np.arange(tile_half, spec.grid_n - tile_half, dtype=np.int32)
        j_valid = np.arange(tile_half, spec.grid_n - tile_half, dtype=np.int32)
        II, JJ = np.meshgrid(i_valid, j_valid, indexing="ij")
        centers_ij = np.stack([II.reshape(-1), JJ.reshape(-1)], axis=1)

        # Only interior residual nodes of each tile (tile_size-2)^2.
        offsets = np.arange(-tile_half, tile_half + 1, dtype=np.int32)
        DI, DJ = np.meshgrid(offsets, offsets, indexing="ij")
        tile_offsets = np.stack([DI.reshape(-1), DJ.reshape(-1)], axis=1)

        cache = {
            "spec": spec,
            "x": x,
            "y": y,
            "coords": coords,
            "source_grid": source_grid,
            "hx": hx,
            "hy": hy,
            "tile_half": tile_half,
            "centers_ij": centers_ij,
            "tile_offsets": tile_offsets,
        }
        self.stage_cache[key] = cache
        return cache

    def _make_monitor_cache(self) -> Dict[str, Any]:
        if self.monitor_cache is not None:
            return self.monitor_cache
        (x0, x1), (y0, y1) = self.bounds
        grid_n = int(self.cfg.monitor_grid_n)
        x = np.linspace(x0, x1, grid_n, dtype=np.float64)
        y = np.linspace(y0, y1, grid_n, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        hx = float((x1 - x0) / max(grid_n - 1, 1))
        hy = float((y1 - y0) / max(grid_n - 1, 1))
        source_grid = bilinear_interp_np(coords[:, 0], coords[:, 1], self.pde.x_gt, self.pde.y_gt, self.pde.source_grid).reshape(grid_n, grid_n)
        self.monitor_cache = {
            "grid_n": grid_n,
            "x": x,
            "y": y,
            "coords": coords,
            "hx": hx,
            "hy": hy,
            "source_grid": source_grid,
        }
        return self.monitor_cache

    def _compute_monitor_loss(self, params, geom: Dict[str, Array]) -> Dict[str, float]:
        cache = self._make_monitor_cache()
        grid_n = int(cache["grid_n"])
        pred = self.predict_batched(params, geom, cache["coords"], self.cfg.test_batch_size).reshape(grid_n, grid_n)
        hx2 = cache["hx"] * cache["hx"]
        hy2 = cache["hy"] * cache["hy"]
        lap = (
            (pred[2:, 1:-1] - 2.0 * pred[1:-1, 1:-1] + pred[:-2, 1:-1]) / hx2
            + (pred[1:-1, 2:] - 2.0 * pred[1:-1, 1:-1] + pred[1:-1, :-2]) / hy2
        )
        src = cache["source_grid"][1:-1, 1:-1]
        r = lap + self.pde.reaction_coeff * pred[1:-1, 1:-1] - src
        r2 = r * r
        return {
            "monitor_mse_f": float(np.mean(r2)),
            "monitor_tail_f": float(np.quantile(r2, 0.9)),
            "monitor_max_abs_r": float(np.max(np.abs(r))),
        }

    # -----------------------
    # model forward
    # -----------------------
    def _hard_bc_envelope(self, x: Array) -> Array:
        tx = (x[:, 0] - self.mins_np[0]) / (self.maxs_np[0] - self.mins_np[0] + 1e-12)
        ty = (x[:, 1] - self.mins_np[1]) / (self.maxs_np[1] - self.mins_np[1] + 1e-12)
        bx = 4.0 * tx * (1.0 - tx)
        by = 4.0 * ty * (1.0 - ty)
        return (bx * by).reshape(-1, 1)

    def model_forward(self, params, geom: Dict[str, Array], x: Array) -> Array:
        d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
        ph = anisotropic_phi_basis(x, geom)
        lam = lambdas_from_phi(ph, d2)
        z = local_coords(x, geom)
        zf = fourier_features(z.reshape(-1, self.d_in), self.freqs).reshape(x.shape[0], self.cfg.n_balls, self.feature_dim)
        outs = [expert_apply(params[j], zf[:, j, :], self.act_name) for j in range(self.cfg.n_balls)]
        Y = jnp.stack(outs, axis=1)
        out = jnp.sum(lam[:, :, None] * Y, axis=1)
        return self._hard_bc_envelope(x) * out

    def predict_batched(self, params, geom: Dict[str, Array], coords_np: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jnp.asarray(coords_np[st:ed], dtype=jnp.float64)
            yb = self.model_forward(params, geom, xb)
            outs.append(np.asarray(yb))
        return np.concatenate(outs, axis=0)

    # -----------------------
    # residual map + heat sampler
    # -----------------------
    def _compute_stage_residual_map(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> NpArray:
        pred = self.predict_batched(params, geom, stage["coords"], self.cfg.test_batch_size).reshape(stage["spec"].grid_n, stage["spec"].grid_n)
        hx2 = stage["hx"] * stage["hx"]
        hy2 = stage["hy"] * stage["hy"]
        lap = (
            (pred[2:, 1:-1] - 2.0 * pred[1:-1, 1:-1] + pred[:-2, 1:-1]) / hx2
            + (pred[1:-1, 2:] - 2.0 * pred[1:-1, 1:-1] + pred[1:-1, :-2]) / hy2
        )
        src = stage["source_grid"][1:-1, 1:-1]
        r = lap + self.pde.reaction_coeff * pred[1:-1, 1:-1] - src

        full = np.zeros_like(pred)
        full[1:-1, 1:-1] = r * r
        return full

    def _prepare_fourier_ops(self, axes: Tuple[NpArray, NpArray]) -> Dict[str, Any]:
        ops = []
        for d, ax in enumerate(axes):
            N = len(ax)
            lo, hi = self.bounds[d]
            L = max(float(hi - lo), 1e-12)
            xnorm = (ax - lo) / L
            n = np.arange(N, dtype=np.float64)
            C = np.cos(np.pi * xnorm[:, None] * n[None, :])
            Cinv = np.linalg.inv(C)
            S = -(np.pi / L) * np.sin(np.pi * xnorm[:, None] * n[None, :]) * n[None, :]
            lam = (np.pi * n / L) ** 2
            ops.append({"C": C, "Cinv": Cinv, "S": S, "lam": lam})
        return {"ops": ops}

    def _heat_density_and_score(self, p0: NpArray, axes: Tuple[NpArray, NpArray], tau: float) -> Tuple[NpArray, NpArray]:
        ops = self._prepare_fourier_ops(axes)
        op0 = ops["ops"][0]
        op1 = ops["ops"][1]
        coeff0 = op0["Cinv"] @ p0 @ op1["Cinv"].T
        lam = op0["lam"][:, None] + op1["lam"][None, :]
        coeff_t = coeff0 * np.exp(-float(max(tau, 0.0)) * lam)

        p = op0["C"] @ coeff_t @ op1["C"].T
        dp0 = op0["S"] @ coeff_t @ op1["C"].T
        dp1 = op0["C"] @ coeff_t @ op1["S"].T

        p = np.maximum(p, 1e-14)
        p = p / np.maximum(np.sum(p), 1e-12)
        score = np.stack([dp0 / p, dp1 / p], axis=-1)
        return p, score

    def _sample_from_density_grid(self, p_grid: NpArray, axes: Tuple[NpArray, NpArray], n: int) -> NpArray:
        p = np.maximum(np.asarray(p_grid, dtype=np.float64), 0.0)
        p = p / np.maximum(np.sum(p), 1e-12)
        flat = p.reshape(-1)
        idx = self.rng.choice(flat.size, size=int(n), replace=True, p=flat)
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)[idx]
        return pts

    def _blend_density_ema(self, stage_name: str, p_new: NpArray, rho: float) -> NpArray:
        prev = self.stage_density_state.get(stage_name, None)
        if prev is not None and prev.shape == p_new.shape:
            p_state = (1.0 - float(rho)) * np.asarray(prev, dtype=np.float64) + float(rho) * np.asarray(p_new, dtype=np.float64)
        else:
            p_state = np.asarray(p_new, dtype=np.float64)
        p_state = np.maximum(p_state, 1e-14)
        p_state = p_state / np.maximum(np.sum(p_state), 1e-12)
        self.stage_density_state[stage_name] = p_state
        return p_state

    def _importance_weights_from_q(self, q: Array, stage: Dict[str, Any]) -> Array:
        q = jnp.asarray(q).reshape(-1)
        spec = stage["spec"]
        gamma = float(spec.weight_gamma)
        w = (q + 1e-12) ** (-gamma)
        w = w / (jnp.mean(w) + 1e-12)
        w = jnp.clip(w, float(spec.weight_clip_min), float(spec.weight_clip_max))
        w = w / (jnp.mean(w) + 1e-12)
        return w

    def _merge_centers_with_retention(self, stage_name: str, new_centers: NpArray, n_tiles: int, retain_frac: float) -> NpArray:
        prev = self.prev_tile_centers_by_stage.get(stage_name, None)
        n_keep = 0
        keep = np.empty((0, 2), dtype=np.int32)
        if prev is not None and prev.size > 0:
            n_keep = min(int(round(float(retain_frac) * n_tiles)), prev.shape[0], n_tiles)
            if n_keep > 0:
                idx = self.rng.choice(prev.shape[0], size=n_keep, replace=False)
                keep = np.asarray(prev[idx], dtype=np.int32)
        n_new = max(n_tiles - n_keep, 0)
        if n_new > 0:
            if new_centers.shape[0] > n_new:
                idx_new = self.rng.choice(new_centers.shape[0], size=n_new, replace=False)
                fresh = np.asarray(new_centers[idx_new], dtype=np.int32)
            else:
                fresh = np.asarray(new_centers, dtype=np.int32)
        else:
            fresh = np.empty((0, 2), dtype=np.int32)
        merged = np.concatenate([keep, fresh], axis=0) if (keep.size or fresh.size) else np.empty((0, 2), dtype=np.int32)
        if merged.shape[0] < n_tiles and new_centers.shape[0] > 0:
            extra_n = n_tiles - merged.shape[0]
            idx_extra = self.rng.choice(new_centers.shape[0], size=extra_n, replace=True)
            merged = np.concatenate([merged, np.asarray(new_centers[idx_extra], dtype=np.int32)], axis=0)
        self.prev_tile_centers_by_stage[stage_name] = np.asarray(merged, dtype=np.int32)
        return np.asarray(merged, dtype=np.int32)

    def _run_ddim_particles(self, p_explicit: NpArray, stage: Dict[str, Any]) -> NpArray:
        axes = (stage["x"], stage["y"])
        steps = max(int(self.cfg.diffusion_steps), 1)
        total_tau = float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt)
        taus = np.linspace(total_tau, 0.0, steps + 1, dtype=np.float64)

        p_tau_max, _ = self._heat_density_and_score(p_explicit, axes, taus[0])
        x = self._sample_from_density_grid(p_tau_max, axes, stage["spec"].particles)

        for m in range(steps):
            tau_cur = taus[m]
            tau_nxt = taus[m + 1]
            dtau = float(tau_cur - tau_nxt)
            _, score_grid = self._heat_density_and_score(p_explicit, axes, tau_cur)
            vals = np.zeros((x.shape[0], 2), dtype=np.float64)
            for d in range(2):
                interp = RegularGridInterpolator(axes, score_grid[..., d], bounds_error=False, fill_value=None)
                vals[:, d] = interp(x)
            x = x - dtau * vals
            x[:, 0] = np.clip(x[:, 0], self.bounds[0][0], self.bounds[0][1])
            x[:, 1] = np.clip(x[:, 1], self.bounds[1][0], self.bounds[1][1])
        return x

    def _build_sampling_state(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> Dict[str, Any]:
        r2 = self._compute_stage_residual_map(params, geom, stage)
        p0 = np.maximum(r2, 0.0) + float(self.cfg.residual_focus_eps)
        sigma = float(stage["spec"].focus_sigma)
        if sigma > 0.0:
            p0 = sp_ndimage.gaussian_filter(p0, sigma=sigma, mode="nearest")
        power = float(stage["spec"].focus_power)
        if abs(power - 1.0) > 1e-15:
            p0 = p0 ** power
        p0 = p0 / np.maximum(np.sum(p0), 1e-12)

        p_explicit, _ = self._heat_density_and_score(p0, (stage["x"], stage["y"]), float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt))
        p_state = self._blend_density_ema(self.current_stage_name, p_explicit, float(stage["spec"].ema_rho))
        particles = self._run_ddim_particles(p_state, stage)

        centers_ij = stage["centers_ij"]
        tile_half = stage["tile_half"]
        xax, yax = stage["x"], stage["y"]
        particle_i = np.clip(np.searchsorted(xax, particles[:, 0]), 1, len(xax) - 1)
        particle_j = np.clip(np.searchsorted(yax, particles[:, 1]), 1, len(yax) - 1)
        particle_i = np.where(np.abs(xax[np.maximum(particle_i - 1, 0)] - particles[:, 0]) <= np.abs(xax[particle_i] - particles[:, 0]), np.maximum(particle_i - 1, 0), particle_i)
        particle_j = np.where(np.abs(yax[np.maximum(particle_j - 1, 0)] - particles[:, 1]) <= np.abs(yax[particle_j] - particles[:, 1]), np.maximum(particle_j - 1, 0), particle_j)

        valid = (
            (particle_i >= tile_half)
            & (particle_i < len(xax) - tile_half)
            & (particle_j >= tile_half)
            & (particle_j < len(yax) - tile_half)
        )
        particle_ij = np.stack([particle_i[valid], particle_j[valid]], axis=1) if np.any(valid) else np.empty((0, 2), dtype=np.int32)

        if particle_ij.shape[0] > 0:
            uniq, counts = np.unique(particle_ij, axis=0, return_counts=True)
            w_part = counts.astype(np.float64)
            w_part = w_part / np.maximum(np.sum(w_part), 1e-12)
        else:
            uniq = np.empty((0, 2), dtype=np.int32)
            w_part = np.empty((0,), dtype=np.float64)

        valid_center_mass = p_state[centers_ij[:, 0], centers_ij[:, 1]]
        valid_center_mass = valid_center_mass / np.maximum(np.sum(valid_center_mass), 1e-12)

        top5_thr = np.quantile(r2[1:-1, 1:-1], 0.95) if r2.shape[0] > 2 else float("nan")
        hotspot = np.zeros_like(r2, dtype=bool)
        if np.isfinite(top5_thr):
            hotspot[1:-1, 1:-1] = r2[1:-1, 1:-1] >= top5_thr

        return {
            "residual_map": r2,
            "p_explicit": p_explicit,
            "p_state": p_state,
            "particle_centers": uniq,
            "particle_weights": w_part,
            "valid_centers": centers_ij,
            "valid_center_mass": valid_center_mass,
            "top5_overlap": float(np.sum(p_state[hotspot])) if np.any(hotspot) else float("nan"),
        }

    def _sample_tile_centers(self, sampling_state: Dict[str, Any], stage: Dict[str, Any]) -> Tuple[NpArray, NpArray]:
        n_tiles = int(stage["spec"].n_tiles)
        uniform_mix = float(stage["spec"].uniform_mix)
        retain_frac = float(stage["spec"].retain_frac)
        n_uni = int(round(uniform_mix * n_tiles))
        n_adapt = max(n_tiles - n_uni, 0)

        centers_ij = sampling_state["valid_centers"]
        p_mass = sampling_state["valid_center_mass"]

        chosen = []
        if n_adapt > 0:
            if sampling_state["particle_centers"].shape[0] > 0:
                idx = self.rng.choice(
                    sampling_state["particle_centers"].shape[0],
                    size=n_adapt,
                    replace=True,
                    p=sampling_state["particle_weights"],
                )
                chosen.append(sampling_state["particle_centers"][idx])
            else:
                idx = self.rng.choice(centers_ij.shape[0], size=n_adapt, replace=True, p=p_mass)
                chosen.append(centers_ij[idx])

        if n_uni > 0:
            idx = self.rng.choice(centers_ij.shape[0], size=n_uni, replace=True)
            chosen.append(centers_ij[idx])

        new_centers = np.concatenate(chosen, axis=0) if chosen else np.empty((0, 2), dtype=np.int32)
        centers = self._merge_centers_with_retention(self.current_stage_name, new_centers, n_tiles, retain_frac)
        q = sampling_state["p_state"][centers[:, 0], centers[:, 1]].reshape(-1)
        q = np.maximum(q, 1e-12)
        q = q / np.maximum(np.sum(q), 1e-12)
        return centers, q

    def _build_tile_batch(self, sampling_state: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
        centers_ij, q = self._sample_tile_centers(sampling_state, stage)
        offsets = stage["tile_offsets"]
        tile_size = int(stage["spec"].tile_size)
        n_tiles = centers_ij.shape[0]

        patch_ij = centers_ij[:, None, :] + offsets[None, :, :]
        ii = patch_ij[:, :, 0]
        jj = patch_ij[:, :, 1]

        x = stage["x"][ii]
        y = stage["y"][jj]
        coords = np.stack([x, y], axis=-1).reshape(n_tiles, tile_size, tile_size, 2)

        src = stage["source_grid"][ii, jj].reshape(n_tiles, tile_size, tile_size)

        return {
            "coords": coords.astype(np.float64),
            "source": src.astype(np.float64),
            "proposal_center": q.astype(np.float64).reshape(-1, 1),
            "hx": float(stage["hx"]),
            "hy": float(stage["hy"]),
        }

    # -----------------------
    # losses
    # -----------------------
    def anchor_loss_and_diff(self, params, geom: Dict[str, Array]) -> Tuple[Array, Array]:
        pred = self.model_forward(params, geom, self.anchor_coords)
        diff = pred - self.anchor_vals
        mse = jnp.mean(diff * diff)
        return mse, diff

    def tile_residual_tensor(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any]) -> Array:
        coords = jnp.asarray(tile_batch["coords"])
        source = jnp.asarray(tile_batch["source"])
        nt, K, _, _ = coords.shape

        pred = self.model_forward(params, geom, coords.reshape(-1, 2)).reshape(nt, K, K, 1)[..., 0]
        hx2 = float(tile_batch["hx"]) ** 2
        hy2 = float(tile_batch["hy"]) ** 2

        lap = (
            (pred[:, 2:, 1:-1] - 2.0 * pred[:, 1:-1, 1:-1] + pred[:, :-2, 1:-1]) / hx2
            + (pred[:, 1:-1, 2:] - 2.0 * pred[:, 1:-1, 1:-1] + pred[:, 1:-1, :-2]) / hy2
        )
        src = source[:, 1:-1, 1:-1]
        r = lap + float(self.pde.reaction_coeff) * pred[:, 1:-1, 1:-1] - src
        return r.reshape(nt, -1)

    def stage_residual_grid(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> Array:
        grid_n = int(stage["spec"].grid_n)
        coords = jnp.asarray(stage["coords"])
        source_grid = jnp.asarray(stage["source_grid"])
        pred = self.model_forward(params, geom, coords).reshape(grid_n, grid_n)
        hx2 = float(stage["hx"]) ** 2
        hy2 = float(stage["hy"]) ** 2
        lap = (
            (pred[2:, 1:-1] - 2.0 * pred[1:-1, 1:-1] + pred[:-2, 1:-1]) / hx2
            + (pred[1:-1, 2:] - 2.0 * pred[1:-1, 1:-1] + pred[1:-1, :-2]) / hy2
        )
        src = source_grid[1:-1, 1:-1]
        return lap + float(self.pde.reaction_coeff) * pred[1:-1, 1:-1] - src

    def residual_scalar_loss(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        spec = stage["spec"]
        total = jnp.array(0.0, dtype=jnp.float64)
        if tile_batch["coords"].shape[0] > 0:
            r = self.tile_residual_tensor(params, geom, tile_batch)
            q = jnp.asarray(tile_batch["proposal_center"]).reshape(-1)
            r2 = jnp.mean(r * r, axis=1)
            w = self._importance_weights_from_q(q, stage)
            mse_f_weighted = jnp.mean(r2 * w)
            mse_f_unweighted = jnp.mean(r2)
            beta = float(spec.residual_beta)
            mse_f_blended = beta * mse_f_weighted + (1.0 - beta) * mse_f_unweighted
            total = total + float(spec.tile_lambda) * mse_f_blended
        r_stage = self.stage_residual_grid(params, geom, stage)
        mse_f_stage = jnp.mean(r_stage * r_stage)
        total = total + float(spec.stage_lambda) * mse_f_stage
        return total

    def anchor_scalar_loss(self, params, geom: Dict[str, Array]) -> Array:
        anchor_mse, _ = self.anchor_loss_and_diff(params, geom)
        return anchor_mse

    def _grad_norm_from_pytree(self, pytree: Any) -> float:
        flat, _ = ravel_pytree(pytree)
        return float(np.linalg.norm(np.asarray(flat, dtype=np.float64)))

    def update_dynamic_anchor_weight(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any], stage: Dict[str, Any]) -> float:
        spec = stage["spec"]
        aw_old = float(stage.get("dynamic_anchor_weight", spec.anchor_weight))

        tile_batch_jax = {
            "coords": jax.device_put(jnp.asarray(tile_batch["coords"]), self.jax_device),
            "source": jax.device_put(jnp.asarray(tile_batch["source"]), self.jax_device),
            "proposal_center": jax.device_put(jnp.asarray(tile_batch["proposal_center"]), self.jax_device),
            "hx": float(tile_batch["hx"]),
            "hy": float(tile_batch["hy"]),
        }
        params = jax.device_put(params, self.jax_device)
        geom = jax.device_put(geom, self.jax_device)

        res_fn = lambda p: self.residual_scalar_loss(p, geom, tile_batch_jax, stage)
        anc_fn = lambda p: self.anchor_scalar_loss(p, geom)

        try:
            res_loss = float(np.asarray(res_fn(params), dtype=np.float64))
            anc_loss = float(np.asarray(anc_fn(params), dtype=np.float64))
            g_res = self._grad_norm_from_pytree(jax.grad(res_fn)(params))
            g_anchor = self._grad_norm_from_pytree(jax.grad(anc_fn)(params))
            eps = 1.0e-12
            ratio = float(spec.anchor_balance_ratio) * g_res / max(g_anchor, eps)
            aw_target = float(np.clip(ratio, float(spec.anchor_aw_min), float(spec.anchor_aw_max)))
            lim = max(float(spec.anchor_aw_change_limit), 1.0)
            aw_target = min(max(aw_target, aw_old / lim), aw_old * lim)
            tau = float(np.clip(spec.anchor_aw_ema_tau, 1.0e-3, 1.0))
            aw_new = float(np.exp((1.0 - tau) * math.log(max(aw_old, 1.0e-12)) + tau * math.log(max(aw_target, 1.0e-12))))
            stage["dynamic_anchor_weight"] = aw_new
            self.last_anchor_balance_info = {
                "aw_old": aw_old,
                "aw_target": aw_target,
                "aw_new": aw_new,
                "g_res": float(g_res),
                "g_anchor": float(g_anchor),
                "res_loss": float(res_loss),
                "anchor_loss": float(anc_loss),
                "status": "updated",
            }
            return aw_new
        except Exception:
            stage["dynamic_anchor_weight"] = aw_old
            self.last_anchor_balance_info = {
                "aw_old": aw_old,
                "aw_target": aw_old,
                "aw_new": aw_old,
                "g_res": float("nan"),
                "g_anchor": float("nan"),
                "res_loss": float("nan"),
                "anchor_loss": float("nan"),
                "status": "fallback",
            }
            return aw_old

    def global_residual_vector(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        anchor_mse, anchor_diff = self.anchor_loss_and_diff(params, geom)
        del anchor_mse
        anchor_vec = jnp.reshape(anchor_diff, (-1,))
        spec = stage["spec"]
        anchor_weight = float(stage.get("dynamic_anchor_weight", spec.anchor_weight))
        anchor_scale = math.sqrt(anchor_weight / max(anchor_vec.shape[0], 1))

        parts = [anchor_scale * anchor_vec]

        if tile_batch["coords"].shape[0] > 0:
            r = self.tile_residual_tensor(params, geom, tile_batch)
            q = jnp.asarray(tile_batch["proposal_center"]).reshape(-1)
            w = self._importance_weights_from_q(q, stage).reshape(-1, 1)
            base_scale = math.sqrt(float(r.shape[0] * r.shape[1]))
            weighted_vec = r * jnp.sqrt(w) / base_scale
            unweighted_vec = r / base_scale
            beta = float(spec.residual_beta)
            tile_scale = math.sqrt(float(spec.tile_lambda))
            if beta > 0.0:
                parts.append(tile_scale * math.sqrt(beta) * jnp.reshape(weighted_vec, (-1,)))
            if beta < 1.0:
                parts.append(tile_scale * math.sqrt(1.0 - beta) * jnp.reshape(unweighted_vec, (-1,)))

        r_stage = self.stage_residual_grid(params, geom, stage)
        stage_vec = jnp.reshape(r_stage, (-1,)) / math.sqrt(float(r_stage.size))
        stage_scale = math.sqrt(float(spec.stage_lambda))
        parts.append(stage_scale * stage_vec)
        return jnp.concatenate(parts, axis=0)

    def fixed_metrics(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, float]:
        anchor_mse, _ = self.anchor_loss_and_diff(params, geom)
        if tile_batch["coords"].shape[0] == 0:
            mse_f_weighted = jnp.array(0.0, dtype=jnp.float64)
            mse_f_unweighted = jnp.array(0.0, dtype=jnp.float64)
            tail_loss = jnp.array(0.0, dtype=jnp.float64)
        else:
            r = self.tile_residual_tensor(params, geom, tile_batch)
            q = jnp.asarray(tile_batch["proposal_center"]).reshape(-1)
            r2 = jnp.mean(r * r, axis=1)
            w = self._importance_weights_from_q(q, stage)
            mse_f_weighted = jnp.mean(r2 * w)
            mse_f_unweighted = jnp.mean(r2)
            thr = jnp.quantile(r2, 0.9)
            tail_loss = jnp.mean(r2[r2 >= thr])
        r_stage = self.stage_residual_grid(params, geom, stage)
        mse_f_stage = jnp.mean(r_stage * r_stage)
        spec = stage["spec"]
        anchor_weight = float(stage.get("dynamic_anchor_weight", spec.anchor_weight))
        beta = float(spec.residual_beta)
        tile_lambda = float(spec.tile_lambda)
        stage_lambda = float(spec.stage_lambda)
        mse_f_blended = beta * float(mse_f_weighted) + (1.0 - beta) * float(mse_f_unweighted)
        loss_total = anchor_weight * float(anchor_mse) + tile_lambda * mse_f_blended + stage_lambda * float(mse_f_stage)
        return {
            "anchor_loss": float(anchor_mse),
            "anchor_weight": float(anchor_weight),
            "mse_f_weighted": float(mse_f_weighted),
            "mse_f_unweighted": float(mse_f_unweighted),
            "mse_f_blended": float(mse_f_blended),
            "mse_f_stage": float(mse_f_stage),
            "tile_lambda": float(tile_lambda),
            "stage_lambda": float(stage_lambda),
            "residual_beta": float(beta),
            "tail_loss": float(tail_loss),
            "loss_total": float(loss_total),
        }

    # -----------------------
    # HF theta update
    # -----------------------
    def _hf_clip_damping(self, damping: float) -> float:
        return max(float(damping), float(self.cfg.hf_damping_min))

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
        return x, {
            "cg_iters": int(max(int(cg_maxiter), 1)),
            "cg_res_norm": float(res_norm),
        }

    def theta_step(self, params, geom: Dict[str, Array], tile_batch: Dict[str, Any], stage: Dict[str, Any]):
        tile_batch_jax = {
            "coords": jax.device_put(jnp.asarray(tile_batch["coords"]), self.jax_device),
            "source": jax.device_put(jnp.asarray(tile_batch["source"]), self.jax_device),
            "proposal_center": jax.device_put(jnp.asarray(tile_batch["proposal_center"]), self.jax_device),
            "hx": float(tile_batch["hx"]),
            "hy": float(tile_batch["hy"]),
        }

        params = jax.device_put(params, self.jax_device)
        geom = jax.device_put(geom, self.jax_device)

        theta0, unravel = ravel_pytree(params)
        theta0 = jax.device_put(jnp.asarray(theta0), self.jax_device)

        def residual_from_flat(theta_flat: Array) -> Array:
            params_trial = unravel(theta_flat)
            return self.global_residual_vector(params_trial, geom, tile_batch_jax, stage)

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
            self.last_theta_step_info = {
                "status": "rollback:nonfinite",
                "obj_old": float(obj_old),
                "obj_trial": float(obj_old),
                "grad_norm": float(grad_norm),
                "step_norm": 0.0,
                "cg_iters": 0,
                "cg_res_norm": float("nan"),
                "damping": float(self.hf_damping),
            }
            return params

        _, jvp_res = jax.linearize(residual_from_flat, theta0)
        damping = self._hf_clip_damping(self.hf_damping)

        accepted = False
        params_trial = params
        obj_trial = obj_old
        step_norm = 0.0
        cg_info = {"cg_iters": 0, "cg_res_norm": float("nan")}
        ls_iters = 0
        status = "rollback:not_improved"

        cg_maxiter = int(stage["spec"].cg_maxiter)
        cg_tol = float(stage["spec"].cg_tol)
        line_search_maxiter = int(stage["spec"].line_search_maxiter)

        for _trial in range(max(int(self.cfg.hf_max_trials), 1)):
            damping_arr = jax.device_put(jnp.asarray(damping, dtype=theta0.dtype), self.jax_device)

            def linop_fn(v: Array) -> Array:
                jv = jvp_res(v)
                return vjp_res(jv)[0] + damping_arr * v

            tol_abs = max(cg_tol * max(float(np.linalg.norm(np.asarray(jt_e))), 1e-12), 1e-10)
            step, cg_info = self._hf_cg_solve(linop_fn, -jt_e, tol_abs=tol_abs, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
            step_np = np.asarray(step, dtype=np.float64)
            if not np.all(np.isfinite(step_np)):
                step_np = -g0_np

            directional = float(np.dot(g0_np, step_np))
            if directional >= 0.0 or (not np.isfinite(directional)):
                step_np = -g0_np
                directional = -float(np.dot(g0_np, g0_np))

            step_norm = float(np.linalg.norm(step_np))
            step_dev = jax.device_put(jnp.asarray(step_np), self.jax_device)

            alpha = 1.0
            accepted_ls = False
            for ls in range(1, max(line_search_maxiter, 1) + 1):
                theta_try = theta0 + alpha * step_dev
                obj_try = float(np.asarray(value_only(theta_try), dtype=np.float64))
                if np.isfinite(obj_try) and obj_try <= obj_old + float(self.cfg.hf_line_search_c1) * alpha * directional:
                    params_trial = unravel(theta_try)
                    params_trial = jax.device_put(params_trial, self.jax_device)
                    obj_trial = obj_try
                    ls_iters = ls
                    accepted_ls = True
                    break
                alpha *= float(self.cfg.hf_line_search_tau)

            if accepted_ls:
                damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_down))
                accepted = True
                status = "applied:hf_jtj"
                break

            damping = self._hf_clip_damping(damping * float(self.cfg.hf_damping_up))

        self.hf_damping = float(damping)
        self.last_theta_step_info = {
            "status": status,
            "obj_old": float(obj_old),
            "obj_trial": float(obj_trial),
            "grad_norm": float(grad_norm),
            "step_norm": float(step_norm if accepted else 0.0),
            "cg_iters": int(cg_info.get("cg_iters", 0)),
            "cg_res_norm": float(cg_info.get("cg_res_norm", float("nan"))),
            "ls_iters": int(ls_iters),
            "damping": float(self.hf_damping),
        }
        return params_trial if accepted else params

    # -----------------------
    # plotting
    # -----------------------
    def _draw_dd_ellipses(self, ax, extent=None):
        centers = np.asarray(self.geom_np["centers"], dtype=np.float64)
        radii = np.asarray(self.geom_np["radii"], dtype=np.float64)
        G_mats = np.asarray(self.geom_np["G_mats"], dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] < 2:
            return
        ax.scatter(centers[:, 0], centers[:, 1], s=12, c="white", edgecolors="white", linewidths=0.4, zorder=4, clip_on=True)
        for j in range(centers.shape[0]):
            G2 = 0.5 * (G_mats[j][:2, :2] + G_mats[j][:2, :2].T)
            vals, vecs = np.linalg.eigh(G2)
            vals = np.clip(vals, 1e-12, None)
            width = 2.0 * float(radii[j]) / math.sqrt(float(vals[0]))
            height = 2.0 * float(radii[j]) / math.sqrt(float(vals[1]))
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            ell = matplotlib.patches.Ellipse((float(centers[j, 0]), float(centers[j, 1])), width=width, height=height, angle=angle, fill=False, color="white", linewidth=0.8, alpha=0.95)
            ell.set_clip_path(ax.patch)
            ax.add_patch(ell)
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.margins(x=0.0, y=0.0)

    def save_best_snapshot_plot(self, out_dir: Path, snapshot: Dict[str, Any], rel_l2_epoch: float, it: int, loss_total: float):
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = snapshot["pred_grid"]
        true = snapshot["true_grid"]
        coords = snapshot["grid_coords"]
        cover = cover_sum_np(coords, self.geom_np["centers"], self.geom_np["radii"], self.geom_np["G_mats"])
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

        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), constrained_layout=True)
        im0 = axes[0].imshow(pred.T, origin="lower", extent=extent, aspect="equal")
        self._draw_dd_ellipses(axes[0], extent)
        axes[0].set_title("Pred + DD ellipses")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(true.T, origin="lower", extent=extent, aspect="equal")
        axes[1].set_title("True")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        im2 = axes[2].imshow(err.T, origin="lower", extent=extent, aspect="equal")
        axes[2].set_title(r"$|u^*-\hat u|^2$")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)

        im3 = axes[3].imshow(cover_grid.T, origin="lower", extent=extent, aspect="equal")
        self._draw_dd_ellipses(axes[3], extent)
        axes[3].set_title(r"cover sum + DD ellipses")
        fig.colorbar(im3, ax=axes[3], fraction=0.046)

        fig.suptitle(f"BEST relL2 | iter={it} | relL2={rel_l2_epoch:.3e} | loss_total={loss_total:.3e}")
        out_path = out_dir / f"best_relL2_snapshot_helmholtz_2d_iter{it:06d}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
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
            if "dynamic_anchor_weight" not in self.current_stage:
                self.current_stage["dynamic_anchor_weight"] = float(self.current_stage["spec"].anchor_weight)

            refresh_every = int(self.current_stage["spec"].refresh_every)
            refresh_due = (
                self.cached_batch is None
                or (self.cached_batch.get("stage_name") != stage_name)
                or ((it - 1) % refresh_every == 0)
            )

            if refresh_due:
                sampling_state = self._build_sampling_state(self.params, self.geom_state, self.current_stage)
                tile_batch = self._build_tile_batch(sampling_state, self.current_stage)
                self.update_dynamic_anchor_weight(self.params, self.geom_state, tile_batch, self.current_stage)
                self.cached_batch = {
                    "stage_name": stage_name,
                    "sampling_state": sampling_state,
                    "tile_batch": tile_batch,
                    "anchor_weight": float(self.current_stage.get("dynamic_anchor_weight", self.current_stage["spec"].anchor_weight)),
                }
                block_pos = 1
            else:
                block_pos = ((it - 1) % refresh_every) + 1

            self.last_refresh_info = {
                "refreshed": int(refresh_due),
                "block_pos": int(block_pos),
                "block_len": int(refresh_every),
                "theta_inner_steps": 1,
            }

            tile_batch = self.cached_batch["tile_batch"]
            if "anchor_weight" in self.cached_batch:
                self.current_stage["dynamic_anchor_weight"] = float(self.cached_batch["anchor_weight"])
            self.params = self.theta_step(self.params, self.geom_state, tile_batch, self.current_stage)

            sampling_state = self.cached_batch["sampling_state"]
            axis_min, axis_max, cond_max = self._effective_axis_stats(self.geom_np)
            self.last_sampling_info = {
                "status": "applied:explicit_heat_ddim_sampling",
                "score_type": "analytic_fourier+ema",
                "top5_overlap": float(sampling_state["top5_overlap"]),
                "mix_exp": 1.0,
                "mix_part": 0.0,
                "mix_uni": float(self.current_stage["spec"].uniform_mix),
                "grid_pocc_l1": float("nan"),
                "grid_ptau_l1": float("nan"),
                "diff_steps": int(self.cfg.diffusion_steps),
                "rho": float(self.current_stage["spec"].ema_rho),
                "retain": float(self.current_stage["spec"].retain_frac),
                "gamma": float(self.current_stage["spec"].weight_gamma),
            }
            self.last_dd_info = {
                "status": "fixed:helmholtz_interface_template",
                "alpha": 0.0,
                "mean_r": float(np.mean(self.geom_np["radii"])),
                "q": 0.0,
                "margin": 0.0,
                "tau_r": 0.0,
                "wp": 0.0,
                "kappa": 0.0,
                "tau_c": 0.0,
                "tau_g": 0.0,
                "axis_min": float(axis_min),
                "axis_max": float(axis_max),
                "cond_max": float(cond_max),
                "uncover": 0.0,
                "drift": 0.0,
            }

            metrics = self.fixed_metrics(self.params, self.geom_state, tile_batch, self.current_stage)

            should_eval = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.rel_l2_eval_every == 0)
            should_monitor = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.monitor_eval_every == 0)
            rel_l2_epoch = float("nan")
            monitor_metrics = {"monitor_mse_f": float("nan"), "monitor_tail_f": float("nan"), "monitor_max_abs_r": float("nan")}
            if should_eval:
                rel_l2_epoch, snapshot = self.pde.eval_rel_l2(self.params, self.geom_state, self.predict_batched, self.cfg.test_batch_size)
                self.latest_relL2 = float(rel_l2_epoch)
                if rel_l2_epoch < self.best_relL2:
                    delta = (best_save_rel - rel_l2_epoch) / max(best_save_rel, 1e-12) if np.isfinite(best_save_rel) else float("inf")
                    self.best_relL2 = rel_l2_epoch
                    self.best_relL2_iter = it
                    self.best_params = self.params
                    self.best_snapshot = snapshot
                    if (not np.isfinite(best_save_rel)) or (delta >= self.cfg.save_plot_every_best_delta) or (it <= 3):
                        self.save_best_snapshot_plot(
                            out_dir / "result_ffusion_helmholtz",
                            snapshot,
                            rel_l2_epoch,
                            it,
                            metrics["loss_total"],
                        )
                        best_save_rel = rel_l2_epoch
            if should_monitor:
                monitor_metrics = self._compute_monitor_loss(self.params, self.geom_state)
                self.latest_monitor_f = float(monitor_metrics["monitor_mse_f"])

            if (it % self.cfg.print_every == 0) or (it == 1) or (it == self.cfg.iters):
                elapsed = time.time() - t0
                eta_txt = estimate_eta(t0, it, self.cfg.iters)
                th = self.last_theta_step_info
                sm = self.last_sampling_info
                dd = self.last_dd_info
                log(
                    f"[ITER {it:6d}/{self.cfg.iters}] "
                    f"loss_total={metrics['loss_total']:.3e}  "
                    f"anchor_loss={metrics['anchor_loss']:.3e}  aw={metrics['anchor_weight']:.1f}  aw_tgt={self.last_anchor_balance_info.get('aw_target', float('nan')):.1f}  g_res={self.last_anchor_balance_info.get('g_res', float('nan')):.3e}  g_anchor={self.last_anchor_balance_info.get('g_anchor', float('nan')):.3e}  "
                    f"mse_f_train={metrics['mse_f_blended']:.3e}  mse_f_w={metrics['mse_f_weighted']:.3e}  mse_f_u={metrics['mse_f_unweighted']:.3e}  beta={metrics['residual_beta']:.2f}  "
                    f"monitor_f={monitor_metrics['monitor_mse_f']:.3e}  monitor_maxr={monitor_metrics['monitor_max_abs_r']:.3e}  "
                    f"tail_loss={metrics['tail_loss']:.3e}  "
                    f"relL2={rel_l2_epoch:.3e}  "
                    f"theta={th.get('status','na')}  sample={sm.get('status','na')}[{sm.get('score_type','na')}]  dd={dd.get('status','na')}  stage={self.current_stage_name}  "
                    f"theta_opt=hf_jtj_gpu  refresh={self.last_refresh_info['refreshed']}  block={self.last_refresh_info['block_pos']}/{self.last_refresh_info['block_len']}  theta_inner=1  "
                    f"diff_steps={sm.get('diff_steps',0)}  top5={sm.get('top5_overlap', float('nan')):.3f}  "
                    f"mix_exp={sm.get('mix_exp', float('nan')):.2f}  mix_part={sm.get('mix_part', float('nan')):.2f}  mix_uni={sm.get('mix_uni', float('nan')):.2f}  "
                    f"rho={sm.get('rho', float('nan')):.2f}  retain={sm.get('retain', float('nan')):.2f}  gamma={sm.get('gamma', float('nan')):.2f}  "
                    f"alpha={dd.get('alpha', float('nan')):.3f}  mean_r={dd.get('mean_r', float('nan')):.3e}  "
                    f"axis_min={dd.get('axis_min', float('nan')):.3e}  axis_max={dd.get('axis_max', float('nan')):.3e}  cond_max={dd.get('cond_max', float('nan')):.3e}  "
                    f"uncover={dd.get('uncover', float('nan')):.3e}  drift={dd.get('drift', float('nan')):.3e}  "
                    f"cg_iters={th.get('cg_iters', 0)}  cg_res={th.get('cg_res_norm', float('nan')):.3e}  "
                    f"iter_time={format_seconds(time.time() - iter_t0)}  total={format_seconds(elapsed)}  eta={eta_txt}"
                )

        log(f"[TRAIN-SUMMARY] best_relL2(iter={self.best_relL2_iter})={self.best_relL2:.3e}")


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default="")
    p.add_argument("--gt_helmholtz", type=str, default="")
    p.add_argument("--log_file", type=str, default="")

    p.add_argument("--n_balls", type=int, default="")
    p.add_argument("--layers", type=int, default="")
    p.add_argument("--width", type=int, default="")
    p.add_argument("--act", type=str, default="", choices=["tanh", "relu", "silu"])

    p.add_argument("--iters", type=int, default="")
    p.add_argument("--hf_damping_init", type=float, default="")
    p.add_argument("--hf_damping_up", type=float, default="")
    p.add_argument("--hf_damping_down", type=float, default="")
    p.add_argument("--hf_cg_tol", type=float, default="")
    p.add_argument("--hf_line_search_maxiter", type=int, default="")
    p.add_argument("--diffusion_dt", type=float, default="")
    p.add_argument("--diffusion_time_scale", type=float, default="")
    p.add_argument("--diffusion_steps", type=int, default="")
    p.add_argument("--tile_size", type=int, default="")
    p.add_argument("--n_anchor_x", type=int, default="")
    p.add_argument("--n_anchor_y", type=int, default="")
    p.add_argument("--anchor_weight", type=float, default="")
    p.add_argument("--rel_l2_eval_every", type=int, default="")
    p.add_argument("--monitor_eval_every", type=int, default="")
    p.add_argument("--monitor_grid_n", type=int, default="")
    p.add_argument("--print_every", type=int, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_handle = setup_logging(args.log_file)
    try:
        rng = set_seed(args.seed)
        pde = HelmholtzPDEAdapter(args.gt_helmholtz)
        cfg = Config(
            n_balls=args.n_balls,
            layers=args.layers,
            width=args.width,
            act=args.act,
            iters=args.iters,
            hf_damping_init=args.hf_damping_init,
            hf_damping_up=args.hf_damping_up,
            hf_damping_down=args.hf_damping_down,
            hf_cg_tol=args.hf_cg_tol,
            hf_line_search_maxiter=args.hf_line_search_maxiter,
            diffusion_dt=args.diffusion_dt,
            diffusion_time_scale=args.diffusion_time_scale,
            diffusion_steps=args.diffusion_steps,
            tile_size=args.tile_size,
            n_anchor_x=args.n_anchor_x,
            n_anchor_y=args.n_anchor_y,
            anchor_weight=args.anchor_weight,
            rel_l2_eval_every=args.rel_l2_eval_every,
            monitor_eval_every=args.monitor_eval_every,
            monitor_grid_n=args.monitor_grid_n,
            print_every=args.print_every,
        )

        solver = PINNFFusionHelmholtzSolver(pde, cfg, rng)
        out_dir = Path("./PINN")
        out_dir.mkdir(parents=True, exist_ok=True)

        total_geom_params = solver.geom_np["centers"].shape[0] * (solver.d_in + 1 + solver.d_in * solver.d_in)
        total_params = solver.theta_size + total_geom_params

        


if __name__ == "__main__":
    main()


