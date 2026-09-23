#!/usr/bin/env python3
from __future__ import annotations




import argparse
import builtins
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as sp_ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from PDE import make_pde as torch_make_pde

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.sparse.linalg import cg as jax_cg


Array = jnp.ndarray
NpArray = np.ndarray
BoundsType = Tuple[Tuple[float, float], ...]

DEFAULT_GT_NAVIER = ""


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
        raise RuntimeError("A JAX GPU backend is required. Set CUDA_VISIBLE_DEVICES and use a JAX CUDA build.")
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


def rel_l2_masked_np(pred: NpArray, truth: NpArray, mask: Optional[NpArray] = None, eps: float = 1e-18) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        pred = pred[m]
        truth = truth[m]
    num = float(np.sqrt(np.mean((pred - truth) ** 2) + eps))
    den = float(np.sqrt(np.mean(truth ** 2) + eps))
    return num / den


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


def bilinear_interp_np(qx: NpArray, qy: NpArray, grid_x: NpArray, grid_y: NpArray, values_xy: NpArray) -> NpArray:
    qx = np.asarray(qx, dtype=np.float64).reshape(-1)
    qy = np.asarray(qy, dtype=np.float64).reshape(-1)
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    gy = np.asarray(grid_y, dtype=np.float64).reshape(-1)
    V = np.asarray(values_xy, dtype=np.float64)

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

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + (1.0 - wx) * wy * v01
        + wx * (1.0 - wy) * v10
        + wx * wy * v11
    ).reshape(-1)


def make_diag_metric_from_axes(axes: Sequence[float]) -> NpArray:
    a = np.asarray(axes, dtype=np.float64)
    a = np.maximum(a, 1.0e-8)
    return np.diag(1.0 / (a * a))


# ============================================================
# Navier GT adapter
# ============================================================
class NavierGTAdapter:
    def __init__(
        self,
        gt_navier: str,
        Re: float = 100.0,
        cylinder_cx: float = 0.0,
        cylinder_cy: float = 0.0,
        use_gt_radius: bool = True,
    ):
        if not Path(gt_navier).exists():
            raise FileNotFoundError(f"Navier GT mat file not found: {gt_navier}")

        mat = loadmat(gt_navier)

        def pick(*names):
            for n in names:
                if n in mat:
                    return mat[n]
            return None

        XX = pick("XX", "X", "xmesh")
        YY = pick("YY", "Y", "ymesh")
        U = pick("U_gt", "u_gt", "U", "u")
        V = pick("V_gt", "v_gt", "V", "v")
        P = pick("P_gt", "p_gt", "P", "p")
        fluid_mask = pick("fluid_mask", "mask", "FluidMask")
        radius = pick("radius", "r", "cylinder_radius")
        matched_time = pick("matched_time", "t", "time", "snapshot_time")

        if XX is None or YY is None or U is None or V is None or P is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                "Navier GT must contain at least XX, YY, U_gt, V_gt, P_gt "
                f"(or compatible names). Found keys={keys}"
            )

        XX = np.asarray(XX, dtype=np.float64)
        YY = np.asarray(YY, dtype=np.float64)
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        P = np.asarray(P, dtype=np.float64)

        if XX.shape != YY.shape or U.shape != XX.shape or V.shape != XX.shape or P.shape != XX.shape:
            raise ValueError(
                f"Inconsistent Navier GT shapes: XX={XX.shape}, YY={YY.shape}, U={U.shape}, V={V.shape}, P={P.shape}"
            )

        # The supplied GT is usually meshgrid-style: XX.shape=(Ny,Nx), YY.shape=(Ny,Nx).
        # Convert all fields to values_xy.shape=(Nx,Ny).
        if np.allclose(XX, XX[0:1, :]) and np.allclose(YY, YY[:, 0:1]):
            x = XX[0, :].copy()
            y = YY[:, 0].copy()
            Uxy = U.T.copy()
            Vxy = V.T.copy()
            Pxy = P.T.copy()
            mask_xy = None if fluid_mask is None else np.asarray(fluid_mask).astype(bool).T.copy()
        elif np.allclose(XX, XX[:, 0:1]) and np.allclose(YY, YY[0:1, :]):
            x = XX[:, 0].copy()
            y = YY[0, :].copy()
            Uxy = U.copy()
            Vxy = V.copy()
            Pxy = P.copy()
            mask_xy = None if fluid_mask is None else np.asarray(fluid_mask).astype(bool).copy()
        else:
            raise ValueError("GT mesh is not recognized as a tensor-product XX/YY grid.")

        x_order = np.argsort(x)
        y_order = np.argsort(y)
        x = x[x_order]
        y = y[y_order]
        Uxy = Uxy[np.ix_(x_order, y_order)]
        Vxy = Vxy[np.ix_(x_order, y_order)]
        Pxy = Pxy[np.ix_(x_order, y_order)]
        if mask_xy is None:
            mask_xy = np.ones_like(Uxy, dtype=bool)
        else:
            mask_xy = mask_xy[np.ix_(x_order, y_order)].astype(bool)

        self.gt_navier = str(gt_navier)
        self.name = "navier_stokes_2d"
        self.coord_kind = "txy"
        self.d_in = 3
        self.d_out = 3
        self.Re = float(Re)
        self.nu = 1.0 / self.Re
        self.cx = float(cylinder_cx)
        self.cy = float(cylinder_cy)
        self.radius = float(np.asarray(radius).squeeze()) if (radius is not None and use_gt_radius) else 0.5
        self.matched_time = float(np.asarray(matched_time).squeeze()) if matched_time is not None else 20.0

        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.U = np.asarray(Uxy, dtype=np.float64)
        self.V = np.asarray(Vxy, dtype=np.float64)
        self.P = np.asarray(Pxy, dtype=np.float64)
        self.fluid_mask = np.asarray(mask_xy, dtype=bool)

        # If the provided mask is missing/too permissive, enforce cylinder exclusion.
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        outside_cyl = ((X - self.cx) ** 2 + (Y - self.cy) ** 2) >= (self.radius * self.radius)
        self.fluid_mask = self.fluid_mask & outside_cyl
        self.pressure_mean = float(np.mean(self.P[self.fluid_mask]))
        gx_target = self.cx + 0.25 * (float(self.x[-1]) - float(self.x[0]))
        gy_target = self.cy
        Xg, Yg = np.meshgrid(self.x, self.y, indexing="ij")
        dist2 = (Xg - gx_target) ** 2 + (Yg - gy_target) ** 2
        dist2 = np.where(self.fluid_mask, dist2, np.inf)
        self._gauge_flat_idx = int(np.argmin(dist2.reshape(-1)))

        # Safe scale-normalization and pressure-gradient targets.
        fluid_vals = self.fluid_mask & np.isfinite(self.U) & np.isfinite(self.V) & np.isfinite(self.P)
        if not np.any(fluid_vals):
            raise RuntimeError("No finite fluid values found in Navier GT.")

        def _safe_std(A, mask, floor=1.0e-6):
            vals = np.asarray(A[mask], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return float(floor)
            return float(max(np.std(vals), floor))

        self.u_scale = _safe_std(self.U, fluid_vals)
        self.v_scale = _safe_std(self.V, fluid_vals)
        self.p_scale = _safe_std(self.P, fluid_vals)
        self.data_scales = np.asarray([self.u_scale, self.v_scale, self.p_scale], dtype=np.float64)
        self.data_inv_scales = 1.0 / np.maximum(self.data_scales, 1.0e-6)

        p_fill = float(np.mean(self.P[fluid_vals]))
        P_safe = np.nan_to_num(np.asarray(self.P, dtype=np.float64), nan=p_fill, posinf=p_fill, neginf=p_fill)
        P_safe = np.where(self.fluid_mask, P_safe, p_fill)
        try:
            Px, Py = np.gradient(P_safe, self.x, self.y, edge_order=1)
        except Exception:
            Px, Py = np.gradient(P_safe, edge_order=1)
        Px = np.nan_to_num(np.asarray(Px, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Py = np.nan_to_num(np.asarray(Py, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        grad_mask = sp_ndimage.binary_erosion(self.fluid_mask, structure=np.ones((5, 5), dtype=bool), border_value=0)
        grad_mask = grad_mask & np.isfinite(Px) & np.isfinite(Py)
        if np.count_nonzero(grad_mask) < 16:
            grad_mask = self.fluid_mask & np.isfinite(Px) & np.isfinite(Py)
        if np.count_nonzero(grad_mask) < 16:
            grad_mask = fluid_vals
        self.Px = Px
        self.Py = Py
        self.pressure_grad_mask = np.asarray(grad_mask, dtype=bool)
        self.px_scale = _safe_std(self.Px, self.pressure_grad_mask)
        self.py_scale = _safe_std(self.Py, self.pressure_grad_mask)
        self.pressure_grad_scales = np.asarray([self.px_scale, self.py_scale], dtype=np.float64)

        self.bounds = (
            (self.matched_time - 0.5, self.matched_time + 0.5),
            (float(self.x[0]), float(self.x[-1])),
            (float(self.y[0]), float(self.y[-1])),
        )
        self.xy_bounds = (self.bounds[1], self.bounds[2])

        # Keep PDE.py aligned with this adapter. We do not use the torch residual in the solver;
        # this object is created to inherit the same Navier-Stokes convention and arguments.
        self.torch_pde = torch_make_pde(
            "navier_stokes_2d",
            Re=self.Re,
            cylinder_radius=self.radius,
            cylinder_center=(self.cx, self.cy),
            t_bounds=self.bounds[0],
            x_bounds=self.bounds[1],
            y_bounds=self.bounds[2],
            inlet_x=self.bounds[1][0],
            wall_bc="noslip",
            device=torch.device("cpu"),
            dtype=torch.float64,
            use_lhs=True,
        )

    def inside_fluid_xy(self, xy: NpArray) -> NpArray:
        xy = np.asarray(xy, dtype=np.float64)
        x = xy[:, 0]
        y = xy[:, 1]
        return ((x - self.cx) ** 2 + (y - self.cy) ** 2) >= (self.radius * self.radius)

    def project_outside_cylinder(self, xy: NpArray, margin: float = 1.0e-4) -> NpArray:
        xy = np.asarray(xy, dtype=np.float64).copy()
        dx = xy[:, 0] - self.cx
        dy = xy[:, 1] - self.cy
        rr = np.sqrt(dx * dx + dy * dy)
        bad = rr < (self.radius + margin)
        if np.any(bad):
            # Avoid undefined radial direction at the exact center.
            dx0 = np.where(rr[bad] > 1e-12, dx[bad] / rr[bad], 1.0)
            dy0 = np.where(rr[bad] > 1e-12, dy[bad] / rr[bad], 0.0)
            xy[bad, 0] = self.cx + (self.radius + margin) * dx0
            xy[bad, 1] = self.cy + (self.radius + margin) * dy0
        xy[:, 0] = np.clip(xy[:, 0], self.bounds[1][0], self.bounds[1][1])
        xy[:, 1] = np.clip(xy[:, 1], self.bounds[2][0], self.bounds[2][1])
        return xy

    def exact_uvp(self, txy: NpArray) -> NpArray:
        txy = np.asarray(txy, dtype=np.float64)
        x = txy[:, 1]
        y = txy[:, 2]
        u = bilinear_interp_np(x, y, self.x, self.y, self.U)
        v = bilinear_interp_np(x, y, self.x, self.y, self.V)
        p = bilinear_interp_np(x, y, self.x, self.y, self.P)
        return np.stack([u, v, p], axis=1)

    def sample_data_points(self, rng: np.random.Generator, n: int) -> Tuple[NpArray, NpArray, NpArray]:
        idx_all = np.flatnonzero(self.fluid_mask.reshape(-1))
        if idx_all.size == 0:
            raise RuntimeError("No fluid points found in GT mask.")
        idx = rng.choice(idx_all, size=int(n), replace=True)
        Nx, Ny = self.fluid_mask.shape
        ix = idx // Ny
        iy = idx % Ny
        t = np.full((n,), self.matched_time, dtype=np.float64)
        x = self.x[ix]
        y = self.y[iy]
        coords = np.stack([t, x, y], axis=1)
        vals = np.stack([self.U[ix, iy], self.V[ix, iy], self.P[ix, iy]], axis=1)
        mask = np.ones_like(vals, dtype=np.float64)
        return coords, vals, mask

    def pressure_gauge_point(self) -> Tuple[NpArray, NpArray]:
        Nx, Ny = self.fluid_mask.shape
        ix = self._gauge_flat_idx // Ny
        iy = self._gauge_flat_idx % Ny
        coord = np.asarray([[self.matched_time, self.x[ix], self.y[iy]]], dtype=np.float64)
        value = np.asarray([self.P[ix, iy]], dtype=np.float64)
        value = np.nan_to_num(value, nan=self.pressure_mean, posinf=self.pressure_mean, neginf=self.pressure_mean)
        return coord, value

    def sample_pressure_grad_points(self, rng: np.random.Generator, n: int) -> Tuple[NpArray, NpArray]:
        n = int(max(n, 0))
        if n <= 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        idx_all = np.flatnonzero(self.pressure_grad_mask.reshape(-1))
        if idx_all.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        idx = rng.choice(idx_all, size=n, replace=True)
        Nx, Ny = self.pressure_grad_mask.shape
        ix = idx // Ny
        iy = idx % Ny
        coords = np.stack([np.full(n, self.matched_time, dtype=np.float64), self.x[ix], self.y[iy]], axis=1)
        vals = np.stack([self.Px[ix, iy], self.Py[ix, iy]], axis=1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        return coords.astype(np.float64), vals.astype(np.float64)

    def uniform_fluid_points(self, rng: np.random.Generator, n: int) -> NpArray:
        xmin, xmax = self.bounds[1]
        ymin, ymax = self.bounds[2]
        pts = []
        need = int(n)
        while need > 0:
            m = max(int(need * 1.4) + 128, 256)
            xy = np.column_stack([
                rng.random(m) * (xmax - xmin) + xmin,
                rng.random(m) * (ymax - ymin) + ymin,
            ])
            mask = self.inside_fluid_xy(xy)
            xy = xy[mask]
            take = min(need, xy.shape[0])
            if take > 0:
                pts.append(xy[:take])
                need -= take
        xy = np.concatenate(pts, axis=0)
        t = np.full((xy.shape[0], 1), self.matched_time, dtype=np.float64)
        return np.concatenate([t, xy], axis=1)

    def eval_rel_l2(self, params: Any, geom: Dict[str, Array], predict_fn, batch_size: int) -> Tuple[float, Dict[str, Any]]:
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        coords = np.stack([
            np.full(X.size, self.matched_time, dtype=np.float64),
            X.reshape(-1),
            Y.reshape(-1),
        ], axis=1)
        pred = predict_fn(params, geom, coords, batch_size).reshape(self.x.size, self.y.size, 3)
        true = np.stack([self.U, self.V, self.P], axis=-1)

        rel_uvp = rel_l2_masked_np(pred, true, self.fluid_mask[..., None].repeat(3, axis=-1))
        rel_vel = rel_l2_masked_np(pred[..., :2], true[..., :2], self.fluid_mask[..., None].repeat(2, axis=-1))
        rel_p = rel_l2_masked_np(pred[..., 2], true[..., 2], self.fluid_mask)

        return rel_uvp, {
            "grid_coords": coords,
            "x": self.x,
            "y": self.y,
            "pred_grid": pred,
            "true_grid": true,
            "fluid_mask": self.fluid_mask,
            "rel_uvp": float(rel_uvp),
            "rel_vel": float(rel_vel),
            "rel_p": float(rel_p),
            "coord_kind": "txy_snapshot",
            "h_label": "x",
            "v_label": "y",
            "aspect": "equal",
        }


# ============================================================
# Elliptic DD geometry: forced 10 balls
# ============================================================
def cover_sum_txy_np(coords_txy: NpArray, centers: NpArray, radii: NpArray, G_mats: NpArray) -> NpArray:
    diff = coords_txy[:, None, :] - centers[None, :, :]
    d2 = np.einsum("nmd,mde,nme->nm", diff, G_mats, diff)
    s = 1.0 - d2 / (radii[None, :] ** 2 + 1e-12)
    ph = np.maximum(s, 0.0) ** 2
    return np.sum(ph, axis=1)


def build_navier_elliptic_template(pde: NavierGTAdapter, n_balls: int = 10) -> Dict[str, NpArray]:
    if int(n_balls) != 10:
        raise ValueError("Navier-Stokes solver requires exactly n_balls=10.")

    t0 = pde.matched_time
    (xmin, xmax), (ymin, ymax) = pde.xy_bounds
    Lx = xmax - xmin
    Ly = ymax - ymin
    cx_dom = 0.5 * (xmin + xmax)
    cy_dom = 0.5 * (ymin + ymax)

    def c(x, y):
        return np.array([t0, float(x), float(y)], dtype=np.float64)

    # Radii are fixed to 1.0. Axis lengths are encoded in G.
    # The template is cylinder-flow-specific:
    # global, inlet/upstream, cylinder, near/mid/far wake, top/bottom wall,
    # upper/lower shear layers.
    centers = [
        c(cx_dom, cy_dom),                          # global
        c(xmin + 0.12 * Lx, cy_dom),                # inlet/upstream
        c(pde.cx, pde.cy),                          # cylinder neighborhood
        c(pde.cx + 0.12 * Lx, pde.cy),              # near wake
        c(pde.cx + 0.30 * Lx, pde.cy),              # mid wake
        c(pde.cx + 0.52 * Lx, pde.cy),              # far wake
        c(cx_dom, ymin + 0.86 * Ly),                # top wall band
        c(cx_dom, ymin + 0.14 * Ly),                # bottom wall band
        c(pde.cx + 0.20 * Lx, pde.cy + 0.18 * Ly),  # upper shear
        c(pde.cx + 0.20 * Lx, pde.cy - 0.18 * Ly),  # lower shear
    ]

    # (t-axis, x-axis, y-axis) lengths. t-axis is deliberately broad because
    # the default solver is a fixed-time snapshot.
    axes = [
        (1.5, 0.78 * Lx, 0.88 * Ly),
        (1.5, 0.23 * Lx, 0.72 * Ly),
        (1.5, 0.16 * Lx, 0.36 * Ly),
        (1.5, 0.22 * Lx, 0.30 * Ly),
        (1.5, 0.26 * Lx, 0.28 * Ly),
        (1.5, 0.34 * Lx, 0.34 * Ly),
        (1.5, 0.70 * Lx, 0.22 * Ly),
        (1.5, 0.70 * Lx, 0.22 * Ly),
        (1.5, 0.28 * Lx, 0.18 * Ly),
        (1.5, 0.28 * Lx, 0.18 * Ly),
    ]

    centers_np = np.stack(centers, axis=0)
    G_np = np.stack([make_diag_metric_from_axes(a) for a in axes], axis=0)
    radii_np = np.ones((10,), dtype=np.float64)
    roles_np = np.arange(10, dtype=np.int32)

    # Guarantee coverage over the fluid snapshot by mild inflation if necessary.
    gx = np.linspace(xmin, xmax, 151, dtype=np.float64)
    gy = np.linspace(ymin, ymax, 91, dtype=np.float64)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    fluid = pde.inside_fluid_xy(xy)
    probe = np.stack([
        np.full(np.sum(fluid), t0, dtype=np.float64),
        xy[fluid, 0],
        xy[fluid, 1],
    ], axis=1)

    for _ in range(20):
        cover = cover_sum_txy_np(probe, centers_np, radii_np, G_np)
        if np.all(cover > 0.0):
            break
        G_np *= (1.0 / (1.04 ** 2))

    cover = cover_sum_txy_np(probe, centers_np, radii_np, G_np)
    if np.any(cover <= 0.0):
        raise RuntimeError("Failed to build a covering 10-ball Navier elliptic DD template.")

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
    grid_x: int
    grid_y: int
    particles: int
    n_f: int
    n_data: int
    uniform_mix: float
    refresh_every: int
    cg_maxiter: int
    cg_tol: float
    focus_power: float
    focus_sigma: float
    data_weight: float
    ema_rho: float
    retain_frac: float
    weight_gamma: float
    weight_clip_min: float
    weight_clip_max: float
    residual_beta: float
    line_search_maxiter: int
    residual_lambda: float = 1.0
    data_balance_ratio: float = 0.10
    data_w_min: float = 1.0
    data_w_max: float = 2.0e4
    data_w_ema_tau: float = 0.10
    data_w_change_limit: float = 1.5


@dataclass
class Config:
    # forced Navier model
    n_balls: int = 10
    layers: int = 1
    width: int = 19
    act: str = "silu"
    freqs: Tuple[float, ...] = (1.0, 2.0, 4.0)

    # training
    iters: int = 2000
    theta_inner_steps: int = 1
    hf_damping_init: float = 1e-2
    hf_damping_min: float = 1e-8
    hf_damping_up: float = 2.0
    hf_damping_down: float = 0.8
    hf_cg_tol: float = 1e-4
    hf_max_trials: int = 4
    hf_line_search_c1: float = 1e-4
    hf_line_search_tau: float = 0.5
    hf_line_search_maxiter: int = 12

    # Navier residual
    Re: float = 100.0
    include_time_derivative: bool = False
    residual_comp_weights: Tuple[float, float, float] = (0.1, 0.1, 1.0)
    data_comp_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # pressure stabilization without dynamic loss reweighting
    pressure_anchor_frac: float = 0.15
    pressure_mean_n: int = 768
    pressure_mean_loss_weight: float = 1.0
    pressure_gauge_loss_weight: float = 1.0
    pressure_grad_n: int = 512
    pressure_grad_loss_weight: float = 0.05

    tail_anchor_stage: str = "late"
    tail_anchor_frac: float = 0.25
    tail_anchor_quantile: float = 0.90

    # weak residual-CVT DD update
    dd_update_stage: str = "late"
    dd_update_every: int = 3
    dd_center_alpha: float = 0.03
    dd_metric_beta: float = 0.0
    dd_freeze_global: bool = True

    # sampling stabilization/aggressiveness
    sampler_top5_cap: float = 0.18

    # diffusion sampler over xy
    diffusion_dt: float = 0.015
    diffusion_time_scale: float = 1.0
    diffusion_steps: int = 8
    residual_focus_eps: float = 1e-12

    # eval / plotting
    rel_l2_eval_every: int = 1
    monitor_eval_every: int = 1
    monitor_grid_x: int = 100
    monitor_grid_y: int = 50
    print_every: int = 1
    test_batch_size: int = 65536
    residual_eval_batch_size: int = 2048
    save_plot_every_best_delta: float = 0.0


# ============================================================
# solver
# ============================================================
class PINNFFusionNavierSolver:
    def __init__(self, pde: NavierGTAdapter, cfg: Config, rng: np.random.Generator):
        if int(cfg.n_balls) != 10:
            raise ValueError("Navier-Stokes solver requires exactly --n_balls 10.")
        self.pde = pde
        self.cfg = cfg
        self.rng = rng
        self.bounds = normalize_bounds(pde.bounds)
        self.xy_bounds = normalize_bounds(pde.xy_bounds)
        self.d_in = pde.d_in
        self.d_out = pde.d_out
        self.freqs = jnp.asarray(np.asarray(cfg.freqs, dtype=np.float64))
        self.feature_dim = self.d_in * (1 + 2 * len(cfg.freqs))
        self.act_name = cfg.act
        self.residual_comp_weights = jnp.asarray(np.asarray(cfg.residual_comp_weights, dtype=np.float64))
        self.data_comp_weights = jnp.asarray(np.asarray(cfg.data_comp_weights, dtype=np.float64))
        self.data_inv_scales = jnp.asarray(np.asarray(pde.data_inv_scales, dtype=np.float64))
        self.pressure_grad_inv_scales = jnp.asarray(1.0 / np.maximum(np.asarray(pde.pressure_grad_scales, dtype=np.float64), 1.0e-6))

        self.geom_np = build_navier_elliptic_template(pde, cfg.n_balls)
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

        # Loss weights are deliberately off: data_weight=1, residual_beta=0,
        # importance gamma=0. Diffusion controls *where* we sample, not how much
        # each residual is reweighted in the least-squares system.
        self.stage_specs = {
            "early": StageSpec(
                grid_x=56, grid_y=32, particles=800, n_f=512, n_data=768,
                uniform_mix=0.25, refresh_every=1, cg_maxiter=24, cg_tol=1.0e-4,
                focus_power=1.0, focus_sigma=1.0, data_weight=1.0, ema_rho=0.20,
                retain_frac=0.35, weight_gamma=0.0, weight_clip_min=1.0, weight_clip_max=1.0,
                residual_beta=0.0, line_search_maxiter=10, residual_lambda=1.0,
                data_balance_ratio=0.10, data_w_min=1.0, data_w_max=1.0,
            ),
            "mid": StageSpec(
                grid_x=72, grid_y=40, particles=1200, n_f=768, n_data=1024,
                uniform_mix=0.20, refresh_every=2, cg_maxiter=48, cg_tol=1.0e-4,
                focus_power=1.2, focus_sigma=0.9, data_weight=1.0, ema_rho=0.12,
                retain_frac=0.45, weight_gamma=0.0, weight_clip_min=1.0, weight_clip_max=1.0,
                residual_beta=0.0, line_search_maxiter=12, residual_lambda=1.0,
                data_balance_ratio=0.12, data_w_min=1.0, data_w_max=1.0,
            ),
            "late": StageSpec(
                grid_x=96, grid_y=52, particles=2200, n_f=1152, n_data=1792,
                uniform_mix=0.17, refresh_every=3, cg_maxiter=96, cg_tol=5.0e-5,
                focus_power=1.55, focus_sigma=0.65, data_weight=1.0, ema_rho=0.08,
                retain_frac=0.55, weight_gamma=0.0, weight_clip_min=1.0, weight_clip_max=1.0,
                residual_beta=0.0, line_search_maxiter=14, residual_lambda=1.0,
                data_balance_ratio=0.15, data_w_min=1.0, data_w_max=1.0,
            ),
        }

        self.stage_cache: Dict[str, Dict[str, Any]] = {}
        self.current_stage_name = "early"
        self.current_stage = self._make_stage(self.stage_specs["early"])
        self.cached_batch: Optional[Dict[str, Any]] = None
        self.stage_density_state: Dict[str, NpArray] = {}
        self.prev_points_by_stage: Dict[str, NpArray] = {}
        self.monitor_cache: Optional[Dict[str, Any]] = None
        self.dd_late_refresh_count = 0

        self.hf_damping = float(cfg.hf_damping_init)
        self.best_relL2 = float("inf")
        self.best_relL2_iter = 0
        self.best_params = self.params
        self.best_snapshot = None
        self.latest_relL2 = float("inf")
        self.latest_monitor_f = float("inf")
        self.last_theta_step_info: Dict[str, Any] = {}
        self.last_sampling_info: Dict[str, Any] = {}
        self.last_dd_info: Dict[str, Any] = {}
        self.last_refresh_info: Dict[str, Any] = {"refreshed": 1, "block_pos": 1, "block_len": 1}
        self.last_data_balance_info: Dict[str, Any] = {
            "dw_old": 1.0,
            "dw_target": 1.0,
            "dw_new": 1.0,
            "g_res": float("nan"),
            "g_data": float("nan"),
            "res_loss": float("nan"),
            "data_loss": float("nan"),
            "status": "off",
        }

    # -----------------------
    # stage helpers
    # -----------------------
    def _stage_name(self, it: int) -> str:
        # Ultra is intentionally disabled. The previous relL2<=0.08 ultra
        # transition over-focused the sampler and repeatedly destabilized v-residuals.
        rel = float(self.latest_relL2) if np.isfinite(self.latest_relL2) else float("inf")
        mon = float(self.latest_monitor_f) if np.isfinite(self.latest_monitor_f) else float("inf")
        if rel <= 0.20 or mon <= 1.0e-2:
            return "late"
        if rel <= 0.60 or mon <= 1.0e-1:
            return "mid"
        return "early"

    def _make_stage(self, spec: StageSpec) -> Dict[str, Any]:
        key = f"{spec.grid_x}_{spec.grid_y}"
        if key in self.stage_cache:
            return self.stage_cache[key]

        xmin, xmax = self.xy_bounds[0]
        ymin, ymax = self.xy_bounds[1]
        x = np.linspace(xmin, xmax, spec.grid_x, dtype=np.float64)
        y = np.linspace(ymin, ymax, spec.grid_y, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        fluid = self.pde.inside_fluid_xy(xy).reshape(spec.grid_x, spec.grid_y)

        coords = np.stack([
            np.full(xy.shape[0], self.pde.matched_time, dtype=np.float64),
            xy[:, 0],
            xy[:, 1],
        ], axis=1)

        cache = {
            "spec": spec,
            "x": x,
            "y": y,
            "coords": coords,
            "xy": xy,
            "fluid_mask_grid": fluid,
        }
        self.stage_cache[key] = cache
        return cache

    def _make_monitor_cache(self) -> Dict[str, Any]:
        if self.monitor_cache is not None:
            return self.monitor_cache
        spec = StageSpec(
            grid_x=int(self.cfg.monitor_grid_x), grid_y=int(self.cfg.monitor_grid_y),
            particles=1, n_f=1, n_data=1, uniform_mix=0.0, refresh_every=1,
            cg_maxiter=1, cg_tol=1.0e-4, focus_power=1.0, focus_sigma=0.0,
            data_weight=1.0, ema_rho=1.0, retain_frac=0.0,
            weight_gamma=0.0, weight_clip_min=1.0, weight_clip_max=1.0,
            residual_beta=0.0, line_search_maxiter=1,
        )
        self.monitor_cache = self._make_stage(spec)
        return self.monitor_cache

    # -----------------------
    # model forward
    # -----------------------
    def model_forward(self, params, geom: Dict[str, Array], x: Array) -> Array:
        d2 = mahalanobis_d2(x, geom["centers"], geom["G_mats"])
        ph = anisotropic_phi_basis(x, geom)
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
    # Navier residual
    # -----------------------
    def ns_residual_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        nu = float(self.pde.nu)
        include_t = bool(self.cfg.include_time_derivative)

        def f_single(z: Array) -> Array:
            return self.model_forward(params, geom, z[None, :])[0]

        def r_single(z: Array) -> Array:
            out = f_single(z)
            u = out[0]
            v = out[1]
            J = jax.jacfwd(f_single)(z)      # output x input: (u,v,p) x (t,x,y)
            Hu = jax.hessian(lambda zz: f_single(zz)[0])(z)
            Hv = jax.hessian(lambda zz: f_single(zz)[1])(z)

            u_t = J[0, 0] if include_t else jnp.array(0.0, dtype=jnp.float64)
            v_t = J[1, 0] if include_t else jnp.array(0.0, dtype=jnp.float64)

            u_x = J[0, 1]
            u_y = J[0, 2]
            v_x = J[1, 1]
            v_y = J[1, 2]
            p_x = J[2, 1]
            p_y = J[2, 2]

            u_xx = Hu[1, 1]
            u_yy = Hu[2, 2]
            v_xx = Hv[1, 1]
            v_yy = Hv[2, 2]

            f_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) + p_x
            f_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + p_y
            f_div = u_x + v_y
            return jnp.stack([f_u, f_v, f_div], axis=0)

        return jax.vmap(r_single)(coords)

    def residual_scalar_loss(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        coords = jnp.asarray(batch["colloc_coords"])
        q = jnp.asarray(batch["proposal_q"]).reshape(-1)
        spec = stage["spec"]
        r = self.ns_residual_at(params, geom, coords)
        r = r * self.residual_comp_weights[None, :]
        r2 = jnp.mean(r * r, axis=1)

        w = self._importance_weights_from_q(q, stage)
        mse_w = jnp.mean(r2 * w)
        mse_u = jnp.mean(r2)
        beta = float(spec.residual_beta)
        return float(spec.residual_lambda) * (beta * mse_w + (1.0 - beta) * mse_u)

    def pressure_correction_losses(self, params, geom: Dict[str, Array], batch: Dict[str, Any]) -> Tuple[Array, Array]:
        p_mean_coords = jnp.asarray(batch.get("p_mean_coords", batch["data_coords"]))
        p_mean_true = jnp.asarray(batch.get("p_mean_true", jnp.asarray(0.0, dtype=jnp.float64))).reshape(())
        p_pred_mean = jnp.mean(self.model_forward(params, geom, p_mean_coords)[:, 2])
        mean_loss = jnp.nan_to_num((p_pred_mean - p_mean_true) ** 2, nan=0.0, posinf=0.0, neginf=0.0)

        p_gauge_coord = jnp.asarray(batch.get("p_gauge_coord", batch["data_coords"][:1]))
        p_gauge_value = jnp.asarray(batch.get("p_gauge_value", jnp.asarray([0.0], dtype=jnp.float64))).reshape(())
        p_gauge_pred = self.model_forward(params, geom, p_gauge_coord)[0, 2]
        gauge_loss = jnp.nan_to_num((p_gauge_pred - p_gauge_value) ** 2, nan=0.0, posinf=0.0, neginf=0.0)
        return mean_loss, gauge_loss

    def pressure_grad_at(self, params, geom: Dict[str, Array], coords: Array) -> Array:
        def p_single(z: Array) -> Array:
            return self.model_forward(params, geom, z[None, :])[0, 2]

        def g_single(z: Array) -> Array:
            g = jax.grad(p_single)(z)
            return jnp.stack([g[1], g[2]], axis=0)

        return jax.vmap(g_single)(coords)

    def pressure_grad_loss(self, params, geom: Dict[str, Array], batch: Dict[str, Any]) -> Array:
        n = int(batch.get("p_grad_n", 0))
        if n <= 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        coords = jnp.asarray(batch.get("p_grad_coords", jnp.zeros((0, 3), dtype=jnp.float64)))
        targets = jnp.asarray(batch.get("p_grad_values", jnp.zeros((0, 2), dtype=jnp.float64)))
        pred = jnp.nan_to_num(self.pressure_grad_at(params, geom, coords), nan=0.0, posinf=0.0, neginf=0.0)
        targets = jnp.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        diff = jnp.nan_to_num((pred - targets) * self.pressure_grad_inv_scales[None, :], nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.mean(diff * diff)

    def data_scalar_loss(self, params, geom: Dict[str, Array], batch: Dict[str, Any]) -> Array:
        coords = jnp.asarray(batch["data_coords"])
        vals = jnp.asarray(batch["data_values"])
        mask = jnp.asarray(batch["data_mask"])
        pred = self.model_forward(params, geom, coords)
        diff = jnp.nan_to_num((pred - vals) * mask * self.data_inv_scales[None, :] * self.data_comp_weights[None, :], nan=0.0, posinf=0.0, neginf=0.0)
        data_loss = jnp.mean(diff * diff)
        p_mean_loss, p_gauge_loss = self.pressure_correction_losses(params, geom, batch)
        p_grad_loss = self.pressure_grad_loss(params, geom, batch)
        total = data_loss + float(self.cfg.pressure_mean_loss_weight) * p_mean_loss + float(self.cfg.pressure_gauge_loss_weight) * p_gauge_loss + float(self.cfg.pressure_grad_loss_weight) * p_grad_loss
        return jnp.nan_to_num(total, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)

    def global_residual_vector(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Array:
        spec = stage["spec"]
        parts = []

        coords = jnp.asarray(batch["colloc_coords"])
        q = jnp.asarray(batch["proposal_q"]).reshape(-1)
        r = self.ns_residual_at(params, geom, coords)
        r = r * self.residual_comp_weights[None, :]
        w = self._importance_weights_from_q(q, stage).reshape(-1, 1)

        base_scale = math.sqrt(float(max(r.size, 1)))
        beta = float(spec.residual_beta)
        if beta > 0.0:
            parts.append(math.sqrt(float(spec.residual_lambda) * beta) * jnp.reshape(r * jnp.sqrt(w) / base_scale, (-1,)))
        if beta < 1.0:
            parts.append(math.sqrt(float(spec.residual_lambda) * (1.0 - beta)) * jnp.reshape(r / base_scale, (-1,)))

        coords_d = jnp.asarray(batch["data_coords"])
        vals_d = jnp.asarray(batch["data_values"])
        mask_d = jnp.asarray(batch["data_mask"])
        pred_d = self.model_forward(params, geom, coords_d)
        diff = jnp.nan_to_num((pred_d - vals_d) * mask_d * self.data_inv_scales[None, :] * self.data_comp_weights[None, :], nan=0.0, posinf=0.0, neginf=0.0)
        data_weight = float(stage.get("dynamic_data_weight", spec.data_weight))
        data_scale = math.sqrt(data_weight / max(int(diff.size), 1))
        parts.append(data_scale * jnp.reshape(diff, (-1,)))

        p_mean_coords = jnp.asarray(batch.get("p_mean_coords", batch["data_coords"]))
        p_mean_true = jnp.asarray(batch.get("p_mean_true", 0.0), dtype=jnp.float64).reshape(())
        p_mean_res = jnp.mean(self.model_forward(params, geom, p_mean_coords)[:, 2]) - p_mean_true
        p_gauge_coord = jnp.asarray(batch.get("p_gauge_coord", batch["data_coords"][:1]))
        p_gauge_value = jnp.asarray(batch.get("p_gauge_value", 0.0), dtype=jnp.float64).reshape(())
        p_gauge_res = self.model_forward(params, geom, p_gauge_coord)[0, 2] - p_gauge_value
        if float(self.cfg.pressure_mean_loss_weight) > 0.0:
            parts.append(jnp.sqrt(jnp.asarray(float(self.cfg.pressure_mean_loss_weight), dtype=jnp.float64)) * jnp.reshape(p_mean_res, (1,)))
        if float(self.cfg.pressure_gauge_loss_weight) > 0.0:
            parts.append(jnp.sqrt(jnp.asarray(float(self.cfg.pressure_gauge_loss_weight), dtype=jnp.float64)) * jnp.reshape(p_gauge_res, (1,)))

        if float(self.cfg.pressure_grad_loss_weight) > 0.0 and int(batch.get("p_grad_n", 0)) > 0:
            p_grad_coords = jnp.asarray(batch.get("p_grad_coords", jnp.zeros((0, 3), dtype=jnp.float64)))
            p_grad_vals = jnp.asarray(batch.get("p_grad_values", jnp.zeros((0, 2), dtype=jnp.float64)))
            pred_pg = jnp.nan_to_num(self.pressure_grad_at(params, geom, p_grad_coords), nan=0.0, posinf=0.0, neginf=0.0)
            p_grad_vals = jnp.nan_to_num(p_grad_vals, nan=0.0, posinf=0.0, neginf=0.0)
            diff_pg = jnp.nan_to_num((pred_pg - p_grad_vals) * self.pressure_grad_inv_scales[None, :], nan=0.0, posinf=0.0, neginf=0.0)
            pg_scale = math.sqrt(float(self.cfg.pressure_grad_loss_weight) / max(int(diff_pg.size), 1))
            parts.append(pg_scale * jnp.reshape(diff_pg, (-1,)))

        return jnp.nan_to_num(jnp.concatenate(parts, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    def fixed_metrics(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, float]:
        coords = jnp.asarray(batch["colloc_coords"])
        q = jnp.asarray(batch["proposal_q"]).reshape(-1)
        r = self.ns_residual_at(params, geom, coords)
        rw = r * self.residual_comp_weights[None, :]
        r2 = jnp.mean(rw * rw, axis=1)
        w = self._importance_weights_from_q(q, stage)

        mse_f_weighted = jnp.mean(r2 * w)
        mse_f_unweighted = jnp.mean(r2)
        thr = jnp.quantile(r2, 0.9)
        tail_loss = jnp.mean(r2[r2 >= thr])
        mse_fu = jnp.mean(r[:, 0] * r[:, 0])
        mse_fv = jnp.mean(r[:, 1] * r[:, 1])
        mse_div = jnp.mean(r[:, 2] * r[:, 2])

        data_loss = self.data_scalar_loss(params, geom, batch)
        p_mean_loss, p_gauge_loss = self.pressure_correction_losses(params, geom, batch)
        p_grad_loss = self.pressure_grad_loss(params, geom, batch)
        spec = stage["spec"]
        beta = float(spec.residual_beta)
        mse_f_blended = beta * float(mse_f_weighted) + (1.0 - beta) * float(mse_f_unweighted)
        data_weight = float(stage.get("dynamic_data_weight", spec.data_weight))
        loss_total = float(spec.residual_lambda) * mse_f_blended + data_weight * float(data_loss)

        return {
            "data_loss": float(data_loss),
            "data_weight": float(data_weight),
            "mse_f_weighted": float(mse_f_weighted),
            "mse_f_unweighted": float(mse_f_unweighted),
            "mse_f_blended": float(mse_f_blended),
            "mse_fu": float(mse_fu),
            "mse_fv": float(mse_fv),
            "mse_div": float(mse_div),
            "residual_beta": float(beta),
            "tail_loss": float(tail_loss),
            "loss_total": float(loss_total),
            "p_mean_loss": float(p_mean_loss),
            "p_gauge_loss": float(p_gauge_loss),
            "p_grad_loss": float(p_grad_loss),
            "p_grad_n": int(batch.get("p_grad_n", 0)),
            "p_anchor_n": int(batch.get("p_anchor_n", 0)),
            "tail_anchor_n": int(batch.get("tail_anchor_n", 0)),
        }

    def _grad_norm_from_pytree(self, pytree: Any) -> float:
        flat, _ = ravel_pytree(pytree)
        return float(np.linalg.norm(np.asarray(flat, dtype=np.float64)))

    def update_dynamic_data_weight(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]) -> float:
        # Explicitly disabled. The loss scale is kept fixed so diffusion sampling
        # affects point selection only, not the least-squares objective geometry.
        stage["dynamic_data_weight"] = 1.0
        self.last_data_balance_info = {
            "dw_old": 1.0,
            "dw_target": 1.0,
            "dw_new": 1.0,
            "g_res": float("nan"),
            "g_data": float("nan"),
            "res_loss": float("nan"),
            "data_loss": float("nan"),
            "status": "off",
        }
        return 1.0

    # -----------------------
    # residual map + diffusion sampler over xy
    # -----------------------
    def _batch_to_jax(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "colloc_coords": jax.device_put(jnp.asarray(batch["colloc_coords"], dtype=jnp.float64), self.jax_device),
            "proposal_q": jax.device_put(jnp.asarray(batch["proposal_q"], dtype=jnp.float64), self.jax_device),
            "data_coords": jax.device_put(jnp.asarray(batch["data_coords"], dtype=jnp.float64), self.jax_device),
            "data_values": jax.device_put(jnp.asarray(batch["data_values"], dtype=jnp.float64), self.jax_device),
            "data_mask": jax.device_put(jnp.asarray(batch["data_mask"], dtype=jnp.float64), self.jax_device),
            "p_mean_coords": jax.device_put(jnp.asarray(batch.get("p_mean_coords", batch["data_coords"]), dtype=jnp.float64), self.jax_device),
            "p_mean_true": jax.device_put(jnp.asarray(batch.get("p_mean_true", 0.0), dtype=jnp.float64), self.jax_device),
            "p_gauge_coord": jax.device_put(jnp.asarray(batch.get("p_gauge_coord", batch["data_coords"][:1]), dtype=jnp.float64), self.jax_device),
            "p_gauge_value": jax.device_put(jnp.asarray(batch.get("p_gauge_value", 0.0), dtype=jnp.float64), self.jax_device),
            "p_grad_coords": jax.device_put(jnp.asarray(batch.get("p_grad_coords", np.empty((0, 3), dtype=np.float64)), dtype=jnp.float64), self.jax_device),
            "p_grad_values": jax.device_put(jnp.asarray(batch.get("p_grad_values", np.empty((0, 2), dtype=np.float64)), dtype=jnp.float64), self.jax_device),
            "p_grad_n": int(batch.get("p_grad_n", 0)),
            "p_anchor_n": int(batch.get("p_anchor_n", 0)),
            "tail_anchor_n": int(batch.get("tail_anchor_n", 0)),
        }

    def _importance_weights_from_q(self, q: Array, stage: Dict[str, Any]) -> Array:
        q = jnp.asarray(q).reshape(-1)
        spec = stage["spec"]
        gamma = float(spec.weight_gamma)
        w = (q + 1e-12) ** (-gamma)
        w = w / (jnp.mean(w) + 1e-12)
        w = jnp.clip(w, float(spec.weight_clip_min), float(spec.weight_clip_max))
        return w / (jnp.mean(w) + 1e-12)

    def _compute_residual_r2_batched(self, params, geom: Dict[str, Array], coords_np: NpArray, batch_size: int) -> NpArray:
        outs = []
        for st in range(0, coords_np.shape[0], batch_size):
            ed = min(st + batch_size, coords_np.shape[0])
            xb = jax.device_put(jnp.asarray(coords_np[st:ed], dtype=jnp.float64), self.jax_device)
            rb = self.ns_residual_at(params, geom, xb)
            rb = rb * self.residual_comp_weights[None, :]
            r2 = jnp.mean(rb * rb, axis=1)
            outs.append(np.asarray(r2))
        return np.concatenate(outs, axis=0)

    def _compute_stage_residual_map(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> NpArray:
        r2 = self._compute_residual_r2_batched(
            params, geom, stage["coords"], batch_size=int(self.cfg.residual_eval_batch_size)
        ).reshape(stage["spec"].grid_x, stage["spec"].grid_y)
        r2 = np.where(stage["fluid_mask_grid"], r2, 0.0)
        return r2

    def _prepare_fourier_ops(self, axes: Tuple[NpArray, NpArray]) -> Dict[str, Any]:
        ops = []
        for d, ax in enumerate(axes):
            N = len(ax)
            lo, hi = self.xy_bounds[d]
            L = max(float(hi - lo), 1e-12)
            xnorm = (ax - lo) / L
            n = np.arange(N, dtype=np.float64)
            C = np.cos(np.pi * xnorm[:, None] * n[None, :])
            Cinv = np.linalg.inv(C)
            S = -(np.pi / L) * np.sin(np.pi * xnorm[:, None] * n[None, :]) * n[None, :]
            lam = (np.pi * n / L) ** 2
            ops.append({"C": C, "Cinv": Cinv, "S": S, "lam": lam})
        return {"ops": ops}

    def _heat_density_and_score(self, p0: NpArray, axes: Tuple[NpArray, NpArray], tau: float, fluid_mask_grid: Optional[NpArray]) -> Tuple[NpArray, NpArray]:
        ops = self._prepare_fourier_ops(axes)
        op0 = ops["ops"][0]
        op1 = ops["ops"][1]
        coeff0 = op0["Cinv"] @ p0 @ op1["Cinv"].T
        lam = op0["lam"][:, None] + op1["lam"][None, :]
        coeff_t = coeff0 * np.exp(-float(max(tau, 0.0)) * lam)

        p = op0["C"] @ coeff_t @ op1["C"].T
        dp0 = op0["S"] @ coeff_t @ op1["C"].T
        dp1 = op0["C"] @ coeff_t @ op1["S"].T

        if fluid_mask_grid is not None:
            p = np.where(fluid_mask_grid, p, 0.0)
        p = np.maximum(p, 1e-14)
        if fluid_mask_grid is not None:
            p = np.where(fluid_mask_grid, p, 0.0)
        p = p / np.maximum(np.sum(p), 1e-12)

        safe = np.maximum(p, 1e-14)
        score = np.stack([dp0 / safe, dp1 / safe], axis=-1)
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return p, score

    def _sample_from_density_grid(self, p_grid: NpArray, axes: Tuple[NpArray, NpArray], n: int) -> NpArray:
        p = np.maximum(np.asarray(p_grid, dtype=np.float64), 0.0)
        p = p / np.maximum(np.sum(p), 1e-12)
        flat = p.reshape(-1)
        idx = self.rng.choice(flat.size, size=int(n), replace=True, p=flat)
        X, Y = np.meshgrid(axes[0], axes[1], indexing="ij")
        xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)[idx]
        return self.pde.project_outside_cylinder(xy)

    def _blend_density_ema(self, stage_name: str, p_new: NpArray, rho: float, fluid_mask_grid: NpArray) -> NpArray:
        prev = self.stage_density_state.get(stage_name, None)
        if prev is not None and prev.shape == p_new.shape:
            p_state = (1.0 - float(rho)) * np.asarray(prev, dtype=np.float64) + float(rho) * np.asarray(p_new, dtype=np.float64)
        else:
            p_state = np.asarray(p_new, dtype=np.float64)
        p_state = np.where(fluid_mask_grid, np.maximum(p_state, 1e-14), 0.0)
        p_state = p_state / np.maximum(np.sum(p_state), 1e-12)
        self.stage_density_state[stage_name] = p_state
        return p_state

    def _run_ddim_particles(self, p_explicit: NpArray, stage: Dict[str, Any]) -> NpArray:
        axes = (stage["x"], stage["y"])
        fluid = stage["fluid_mask_grid"]
        steps = max(int(self.cfg.diffusion_steps), 1)
        total_tau = float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt)
        taus = np.linspace(total_tau, 0.0, steps + 1, dtype=np.float64)

        p_tau_max, _ = self._heat_density_and_score(p_explicit, axes, taus[0], fluid)
        x = self._sample_from_density_grid(p_tau_max, axes, stage["spec"].particles)

        for m in range(steps):
            tau_cur = taus[m]
            tau_nxt = taus[m + 1]
            dtau = float(tau_cur - tau_nxt)
            _, score_grid = self._heat_density_and_score(p_explicit, axes, tau_cur, fluid)
            vals = np.zeros((x.shape[0], 2), dtype=np.float64)
            for d in range(2):
                interp = RegularGridInterpolator(axes, score_grid[..., d], bounds_error=False, fill_value=0.0)
                vals[:, d] = interp(x)
            x = x - dtau * vals
            x[:, 0] = np.clip(x[:, 0], self.xy_bounds[0][0], self.xy_bounds[0][1])
            x[:, 1] = np.clip(x[:, 1], self.xy_bounds[1][0], self.xy_bounds[1][1])
            x = self.pde.project_outside_cylinder(x)
        return x

    def _interp_density_to_xy(self, p_grid: NpArray, axes: Tuple[NpArray, NpArray], xy: NpArray) -> NpArray:
        interp = RegularGridInterpolator(axes, p_grid, bounds_error=False, fill_value=1e-12)
        q = interp(np.asarray(xy, dtype=np.float64))
        q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
        return q

    def _merge_points_with_retention(self, stage_name: str, new_txy: NpArray, n_f: int, retain_frac: float) -> NpArray:
        prev = self.prev_points_by_stage.get(stage_name, None)
        keep = np.empty((0, 3), dtype=np.float64)
        if prev is not None and prev.size > 0:
            n_keep = min(int(round(float(retain_frac) * n_f)), prev.shape[0], n_f)
            if n_keep > 0:
                idx = self.rng.choice(prev.shape[0], size=n_keep, replace=False)
                keep = np.asarray(prev[idx], dtype=np.float64)
        n_new = max(n_f - keep.shape[0], 0)
        if new_txy.shape[0] > n_new:
            idx_new = self.rng.choice(new_txy.shape[0], size=n_new, replace=False)
            fresh = np.asarray(new_txy[idx_new], dtype=np.float64)
        else:
            fresh = np.asarray(new_txy, dtype=np.float64)
        merged = np.concatenate([keep, fresh], axis=0) if keep.size or fresh.size else np.empty((0, 3), dtype=np.float64)
        if merged.shape[0] < n_f:
            extra = self.pde.uniform_fluid_points(self.rng, n_f - merged.shape[0])
            merged = np.concatenate([merged, extra], axis=0)
        self.prev_points_by_stage[stage_name] = np.asarray(merged, dtype=np.float64)
        return np.asarray(merged, dtype=np.float64)

    def _build_sampling_state(self, params, geom: Dict[str, Array], stage: Dict[str, Any]) -> Dict[str, Any]:
        r2 = self._compute_stage_residual_map(params, geom, stage)
        p0 = np.maximum(r2, 0.0) + float(self.cfg.residual_focus_eps)
        p0 = np.where(stage["fluid_mask_grid"], p0, 0.0)

        sigma = float(stage["spec"].focus_sigma)
        if sigma > 0.0:
            p0 = sp_ndimage.gaussian_filter(p0, sigma=sigma, mode="nearest")
            p0 = np.where(stage["fluid_mask_grid"], p0, 0.0)

        power = float(stage["spec"].focus_power)
        if abs(power - 1.0) > 1e-15:
            p0 = np.where(stage["fluid_mask_grid"], np.maximum(p0, 1e-14) ** power, 0.0)

        p0 = p0 / np.maximum(np.sum(p0), 1e-12)
        p_explicit, _ = self._heat_density_and_score(
            p0, (stage["x"], stage["y"]),
            float(self.cfg.diffusion_time_scale) * float(self.cfg.diffusion_dt),
            stage["fluid_mask_grid"],
        )
        p_state = self._blend_density_ema(self.current_stage_name, p_explicit, float(stage["spec"].ema_rho), stage["fluid_mask_grid"])

        top5_thr = np.quantile(r2[stage["fluid_mask_grid"]], 0.95) if np.any(stage["fluid_mask_grid"]) else float("nan")
        hotspot = (r2 >= top5_thr) & stage["fluid_mask_grid"] if np.isfinite(top5_thr) else np.zeros_like(r2, dtype=bool)
        top5_mass_pre = float(np.sum(p_state[hotspot])) if np.any(hotspot) else float("nan")
        cap_mix = 0.0
        cap = float(self.cfg.sampler_top5_cap)
        if np.any(hotspot) and np.isfinite(top5_mass_pre) and top5_mass_pre > cap:
            uni = np.where(stage["fluid_mask_grid"], 1.0, 0.0)
            uni = uni / np.maximum(np.sum(uni), 1e-12)
            uni_hot = float(np.sum(uni[hotspot]))
            denom = max(top5_mass_pre - uni_hot, 1e-12)
            cap_mix = float(np.clip((top5_mass_pre - cap) / denom, 0.0, 0.95))
            p_state = (1.0 - cap_mix) * p_state + cap_mix * uni
            p_state = np.where(stage["fluid_mask_grid"], np.maximum(p_state, 1e-14), 0.0)
            p_state = p_state / np.maximum(np.sum(p_state), 1e-12)
            self.stage_density_state[self.current_stage_name] = p_state

        particles_xy = self._run_ddim_particles(p_state, stage)
        top5_mass_post = float(np.sum(p_state[hotspot])) if np.any(hotspot) else float("nan")

        return {
            "residual_map": r2,
            "p_explicit": p_explicit,
            "p_state": p_state,
            "particles_xy": particles_xy,
            "top5_overlap": top5_mass_post,
            "top5_cap_mix": float(cap_mix),
            "top5_overlap_pre": top5_mass_pre,
        }

    def _sample_tail_anchor_points(self, sampling_state: Dict[str, Any], stage: Dict[str, Any], n: int) -> NpArray:
        n = int(max(n, 0))
        if n <= 0:
            return np.empty((0, 3), dtype=np.float64)
        r2 = np.asarray(sampling_state.get("residual_map"), dtype=np.float64)
        fluid = np.asarray(stage["fluid_mask_grid"], dtype=bool)
        vals = r2[fluid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return self.pde.uniform_fluid_points(self.rng, n)
        thr = float(np.quantile(vals, float(np.clip(self.cfg.tail_anchor_quantile, 0.0, 0.999))))
        top = fluid & np.isfinite(r2) & (r2 >= thr)
        if np.count_nonzero(top) == 0:
            top = fluid & np.isfinite(r2)
        weights = np.where(top, np.maximum(r2, 0.0), 0.0).reshape(-1)
        if (not np.isfinite(weights).all()) or float(np.sum(weights)) <= 0.0:
            weights = top.astype(np.float64).reshape(-1)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights / np.maximum(np.sum(weights), 1e-12)
        idx = self.rng.choice(weights.size, size=n, replace=True, p=weights)
        gx, gy = stage["x"], stage["y"]
        ix = idx // len(gy)
        iy = idx % len(gy)
        dx = float(gx[1] - gx[0]) if len(gx) > 1 else 0.0
        dy = float(gy[1] - gy[0]) if len(gy) > 1 else 0.0
        x = gx[ix] + (self.rng.random(n) - 0.5) * dx
        y = gy[iy] + (self.rng.random(n) - 0.5) * dy
        xy = self.pde.project_outside_cylinder(np.stack([x, y], axis=1))
        t = np.full((n, 1), self.pde.matched_time, dtype=np.float64)
        return np.concatenate([t, xy], axis=1).astype(np.float64)

    def _build_training_batch(self, sampling_state: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
        spec = stage["spec"]
        n_f = int(spec.n_f)
        n_tail = 0
        if self.current_stage_name == str(self.cfg.tail_anchor_stage) and float(self.cfg.tail_anchor_frac) > 0.0:
            n_tail = int(round(float(self.cfg.tail_anchor_frac) * n_f))
        n_tail = min(max(n_tail, 0), max(n_f - 1, 0))
        n_uni = int(round(float(spec.uniform_mix) * n_f))
        n_uni = min(max(n_uni, 0), max(n_f - n_tail, 0))
        n_adapt = max(n_f - n_uni - n_tail, 0)

        pts = []
        if n_adapt > 0:
            particles = sampling_state["particles_xy"]
            if particles.shape[0] >= n_adapt:
                idx = self.rng.choice(particles.shape[0], size=n_adapt, replace=False)
            else:
                idx = self.rng.choice(particles.shape[0], size=n_adapt, replace=True)
            xy_adapt = particles[idx]
            t_adapt = np.full((xy_adapt.shape[0], 1), self.pde.matched_time, dtype=np.float64)
            pts.append(np.concatenate([t_adapt, xy_adapt], axis=1))

        if n_tail > 0:
            pts.append(self._sample_tail_anchor_points(sampling_state, stage, n_tail))

        if n_uni > 0:
            pts.append(self.pde.uniform_fluid_points(self.rng, n_uni))

        colloc = np.concatenate(pts, axis=0) if pts else self.pde.uniform_fluid_points(self.rng, n_f)
        colloc = self._merge_points_with_retention(self.current_stage_name, colloc, n_f, float(spec.retain_frac))

        q = self._interp_density_to_xy(sampling_state["p_state"], (stage["x"], stage["y"]), colloc[:, 1:3])
        q = q / np.maximum(np.sum(q), 1e-12)

        n_data = int(spec.n_data)
        n_p_anchor = int(round(float(self.cfg.pressure_anchor_frac) * n_data))
        n_main = max(n_data - n_p_anchor, 1)
        data_coords, data_values, data_mask = self.pde.sample_data_points(self.rng, n_main)
        if n_p_anchor > 0:
            pc, pv, _pm = self.pde.sample_data_points(self.rng, n_p_anchor)
            p_mask = np.zeros_like(pv, dtype=np.float64)
            p_mask[:, 2] = 1.0
            data_coords = np.concatenate([data_coords, pc], axis=0)
            data_values = np.concatenate([data_values, pv], axis=0)
            data_mask = np.concatenate([data_mask, p_mask], axis=0)

        p_mean_n = max(int(self.cfg.pressure_mean_n), 8)
        p_mean_coords, _pm_vals, _pm_mask = self.pde.sample_data_points(self.rng, p_mean_n)
        p_gauge_coord, p_gauge_value = self.pde.pressure_gauge_point()
        p_grad_n = max(int(self.cfg.pressure_grad_n), 0) if float(self.cfg.pressure_grad_loss_weight) > 0.0 else 0
        p_grad_coords, p_grad_values = self.pde.sample_pressure_grad_points(self.rng, p_grad_n)

        return {
            "colloc_coords": colloc.astype(np.float64),
            "proposal_q": q.astype(np.float64).reshape(-1, 1),
            "data_coords": data_coords.astype(np.float64),
            "data_values": data_values.astype(np.float64),
            "data_mask": data_mask.astype(np.float64),
            "p_anchor_n": int(n_p_anchor),
            "p_mean_coords": p_mean_coords.astype(np.float64),
            "p_mean_true": np.asarray(self.pde.pressure_mean, dtype=np.float64),
            "p_gauge_coord": p_gauge_coord.astype(np.float64),
            "p_gauge_value": np.asarray(p_gauge_value, dtype=np.float64),
            "p_grad_coords": p_grad_coords.astype(np.float64),
            "p_grad_values": p_grad_values.astype(np.float64),
            "p_grad_n": int(p_grad_coords.shape[0]),
            "tail_anchor_n": int(n_tail),
        }

    def _effective_axis_stats(self, geom_np: Dict[str, NpArray]) -> Tuple[float, float, float]:
        G = geom_np["G_mats"]
        radii = geom_np["radii"]
        vals_all = np.linalg.eigvalsh(0.5 * (G + np.transpose(G, (0, 2, 1))))
        vals_all = np.clip(vals_all, 1e-12, None)
        axes = radii[:, None] / np.sqrt(vals_all)
        cond = np.max(np.max(vals_all, axis=1) / np.maximum(np.min(vals_all, axis=1), 1e-12))
        return float(np.min(axes)), float(np.max(axes)), float(cond)

    def _compute_monitor_loss(self, params, geom: Dict[str, Array]) -> Dict[str, float]:
        cache = self._make_monitor_cache()
        r2 = self._compute_stage_residual_map(params, geom, cache)
        fluid = cache["fluid_mask_grid"]
        if not np.any(fluid):
            return {"monitor_mse_f": float("nan"), "monitor_tail_f": float("nan"), "monitor_max_abs_r": float("nan")}
        r2f = r2[fluid]
        return {
            "monitor_mse_f": float(np.mean(r2f)),
            "monitor_tail_f": float(np.quantile(r2f, 0.9)),
            "monitor_max_abs_r": float(np.sqrt(np.max(r2f))),
        }

    # -----------------------
    # weak residual-CVT DD update
    # -----------------------
    def _lambda_np_for_stage_coords(self, coords: NpArray) -> NpArray:
        centers = np.asarray(self.geom_np["centers"], dtype=np.float64)
        G = np.asarray(self.geom_np["G_mats"], dtype=np.float64)
        radii = np.asarray(self.geom_np["radii"], dtype=np.float64)
        diff = coords[:, None, :] - centers[None, :, :]
        d2 = np.einsum("nmd,mde,nme->nm", diff, G, diff)
        ph = np.maximum(1.0 - d2 / (radii[None, :] ** 2 + 1e-12), 0.0) ** 2
        s = np.sum(ph, axis=1, keepdims=True)
        lam = np.divide(ph, s + 1e-12)
        uncovered = (s[:, 0] <= 1e-12)
        if np.any(uncovered):
            nearest = np.argmin(d2[uncovered], axis=1)
            lam[uncovered, :] = 0.0
            lam[uncovered, nearest] = 1.0
        return lam

    def maybe_update_weak_dd(self, sampling_state: Dict[str, Any], stage: Dict[str, Any], refreshed: bool) -> Dict[str, Any]:
        axis_min0, axis_max0, cond_max0 = self._effective_axis_stats(self.geom_np)
        base = {
            "status": "held:weak_residual_cvt_dd",
            "dd_alpha": 0.0,
            "dd_beta": 0.0,
            "dd_updated": 0,
            "axis_min": float(axis_min0),
            "axis_max": float(axis_max0),
            "cond_max": float(cond_max0),
            "uncover": 0.0,
            "drift": 0.0,
        }
        stage_name = str(stage.get("name", self.current_stage_name))
        if stage_name != str(self.cfg.dd_update_stage):
            base["status"] = f"skipped:dd_stage_{stage_name}"
            return base
        if not refreshed:
            return base

        self.dd_late_refresh_count += 1
        every = max(int(self.cfg.dd_update_every), 1)
        if (self.dd_late_refresh_count % every) != 0:
            base["status"] = f"held:dd_cadence_{self.dd_late_refresh_count % every}/{every}"
            return base

        alpha = float(self.cfg.dd_center_alpha)
        beta = float(self.cfg.dd_metric_beta)
        if alpha <= 0.0 and beta <= 0.0:
            base["status"] = "skipped:dd_alpha_beta_zero"
            return base

        coords = np.asarray(stage["coords"], dtype=np.float64)
        xy = np.asarray(stage["xy"], dtype=np.float64)
        fluid = np.asarray(stage["fluid_mask_grid"], dtype=bool).reshape(-1)
        p = np.asarray(sampling_state["p_state"], dtype=np.float64).reshape(-1)
        p = np.where(fluid, np.maximum(p, 0.0), 0.0)
        p = p / np.maximum(np.sum(p), 1e-12)

        lam = self._lambda_np_for_stage_coords(coords)
        centers_old = np.asarray(self.geom_np["centers"], dtype=np.float64)
        centers_new = centers_old.copy()
        roles = np.asarray(self.geom_np.get("roles", np.arange(centers_old.shape[0])), dtype=np.int32)

        updated = 0
        drift_vals = []
        for j in range(centers_old.shape[0]):
            if bool(self.cfg.dd_freeze_global) and int(roles[j]) == 0:
                continue
            w = p * lam[:, j]
            mass = float(np.sum(w))
            if mass <= 1e-8:
                continue
            target_xy = np.sum(xy * w[:, None], axis=0) / mass
            target_xy = self.pde.project_outside_cylinder(target_xy.reshape(1, 2))[0]
            old_xy = centers_old[j, 1:3].copy()
            new_xy = (1.0 - alpha) * old_xy + alpha * target_xy
            new_xy = self.pde.project_outside_cylinder(new_xy.reshape(1, 2))[0]
            centers_new[j, 1:3] = new_xy
            d = float(np.linalg.norm(new_xy - old_xy))
            if d > 0.0:
                updated += 1
                drift_vals.append(d)

        self.geom_np["centers"] = centers_new
        if beta > 0.0:
            self.geom_np["G_mats"] = (1.0 - beta) * self.geom_np["G_mats"] + beta * self.geom_np["G_mats"]

        self.geom_state = {
            "centers": jax.device_put(jnp.asarray(self.geom_np["centers"]), self.jax_device),
            "G_mats": jax.device_put(jnp.asarray(self.geom_np["G_mats"]), self.jax_device),
            "radii": jax.device_put(jnp.asarray(self.geom_np["radii"]), self.jax_device),
            "roles": jax.device_put(jnp.asarray(self.geom_np["roles"]), self.jax_device),
        }
        axis_min, axis_max, cond_max = self._effective_axis_stats(self.geom_np)
        cover = cover_sum_txy_np(coords[fluid], self.geom_np["centers"], self.geom_np["radii"], self.geom_np["G_mats"])
        uncover = float(np.mean(cover <= 0.0)) if cover.size else 0.0
        return {
            "status": "applied:weak_residual_cvt_dd" if updated > 0 else "skipped:dd_no_mass",
            "dd_alpha": float(alpha if updated > 0 else 0.0),
            "dd_beta": float(beta if updated > 0 else 0.0),
            "dd_updated": int(updated),
            "axis_min": float(axis_min),
            "axis_max": float(axis_max),
            "cond_max": float(cond_max),
            "uncover": float(uncover),
            "drift": float(np.mean(drift_vals) if drift_vals else 0.0),
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

    def theta_step(self, params, geom: Dict[str, Array], batch: Dict[str, Any], stage: Dict[str, Any]):
        batch_jax = self._batch_to_jax(batch)
        params = jax.device_put(params, self.jax_device)
        geom = jax.device_put(geom, self.jax_device)

        theta0, unravel = ravel_pytree(params)
        theta0 = jax.device_put(jnp.asarray(theta0), self.jax_device)

        def residual_from_flat(theta_flat: Array) -> Array:
            params_trial = unravel(theta_flat)
            return self.global_residual_vector(params_trial, geom, batch_jax, stage)

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
        ax.scatter(centers[:, 1], centers[:, 2], s=12, c="white", edgecolors="black", linewidths=0.4, zorder=4, clip_on=True)
        for j in range(centers.shape[0]):
            G2 = 0.5 * (G_mats[j][1:3, 1:3] + G_mats[j][1:3, 1:3].T)
            vals, vecs = np.linalg.eigh(G2)
            vals = np.clip(vals, 1e-12, None)
            width = 2.0 * float(radii[j]) / math.sqrt(float(vals[0]))
            height = 2.0 * float(radii[j]) / math.sqrt(float(vals[1]))
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            ell = matplotlib.patches.Ellipse(
                (float(centers[j, 1]), float(centers[j, 2])),
                width=width, height=height, angle=angle,
                fill=False, color="white", linewidth=0.9, alpha=0.95,
            )
            ell.set_clip_path(ax.patch)
            ax.add_patch(ell)
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.margins(x=0.0, y=0.0)

    def save_best_snapshot_plot(self, out_dir: Path, snapshot: Dict[str, Any], rel_l2_epoch: float, it: int, loss_total: float):
        out_dir.mkdir(parents=True, exist_ok=True)
        x = snapshot["x"]
        y = snapshot["y"]
        pred = snapshot["pred_grid"]
        true = snapshot["true_grid"]
        mask = snapshot["fluid_mask"]
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

        err_u = (pred[..., 0] - true[..., 0]) ** 2
        err_v = (pred[..., 1] - true[..., 1]) ** 2
        err_p = (pred[..., 2] - true[..., 2]) ** 2
        speed_pred = np.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
        speed_true = np.sqrt(true[..., 0] ** 2 + true[..., 1] ** 2)
        speed_err = (speed_pred - speed_true) ** 2

        cover = cover_sum_txy_np(snapshot["grid_coords"], self.geom_np["centers"], self.geom_np["radii"], self.geom_np["G_mats"])
        cover_grid = cover.reshape(x.size, y.size)

        def masked(A):
            B = np.asarray(A, dtype=np.float64).copy()
            B[~mask] = np.nan
            return B

        fig, axes = plt.subplots(3, 4, figsize=(18.8, 12.4), constrained_layout=True)

        pred_titles = ["Pred u + DD", "Pred v + DD", "Pred p + DD"]
        true_titles = ["True u", "True v", "True p"]
        err_titles = [r"|u-u*|^2", r"|v-v*|^2", r"|p-p*|^2"]
        for k in range(3):
            im = axes[0, k].imshow(masked(pred[..., k]).T, origin="lower", extent=extent, aspect="equal")
            self._draw_dd_ellipses(axes[0, k], extent)
            axes[0, k].set_title(pred_titles[k])
            fig.colorbar(im, ax=axes[0, k], fraction=0.046)

            imt = axes[1, k].imshow(masked(true[..., k]).T, origin="lower", extent=extent, aspect="equal")
            axes[1, k].set_title(true_titles[k])
            fig.colorbar(imt, ax=axes[1, k], fraction=0.046)

        imc = axes[0, 3].imshow(masked(cover_grid).T, origin="lower", extent=extent, aspect="equal")
        self._draw_dd_ellipses(axes[0, 3], extent)
        axes[0, 3].set_title("cover sum + DD")
        fig.colorbar(imc, ax=axes[0, 3], fraction=0.046)

        imm = axes[1, 3].imshow(mask.T.astype(float), origin="lower", extent=extent, aspect="equal")
        axes[1, 3].set_title("fluid mask")
        fig.colorbar(imm, ax=axes[1, 3], fraction=0.046)

        for ax, data, title in zip(axes[2, :3], [err_u, err_v, err_p], err_titles):
            ime = ax.imshow(masked(data).T, origin="lower", extent=extent, aspect="equal")
            ax.set_title(title)
            fig.colorbar(ime, ax=ax, fraction=0.046)

        ims = axes[2, 3].imshow(masked(speed_err).T, origin="lower", extent=extent, aspect="equal")
        axes[2, 3].set_title("speed error")
        fig.colorbar(ims, ax=axes[2, 3], fraction=0.046)

        for ax in axes.reshape(-1):
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        fig.suptitle(
            f"BEST Navier relL2 | iter={it} | relL2_uvp={rel_l2_epoch:.3e} "
            f"| relL2_vel={snapshot['rel_vel']:.3e} | relL2_p={snapshot['rel_p']:.3e} | loss_total={loss_total:.3e}"
        )
        out_path = out_dir / f"best_relL2_snapshot_navier_stokes_2d_iter{it:06d}.png"
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
            self.current_stage["dynamic_data_weight"] = 1.0

            refresh_every = int(self.current_stage["spec"].refresh_every)
            refresh_due = (
                self.cached_batch is None
                or (self.cached_batch.get("stage_name") != stage_name)
                or ((it - 1) % refresh_every == 0)
            )

            if refresh_due:
                sampling_state = self._build_sampling_state(self.params, self.geom_state, self.current_stage)
                self.last_dd_info = self.maybe_update_weak_dd(sampling_state, self.current_stage, refreshed=True)
                batch = self._build_training_batch(sampling_state, self.current_stage)
                self.update_dynamic_data_weight(self.params, self.geom_state, batch, self.current_stage)
                self.cached_batch = {
                    "stage_name": stage_name,
                    "sampling_state": sampling_state,
                    "batch": batch,
                    "data_weight": 1.0,
                }
                block_pos = 1
            else:
                block_pos = ((it - 1) % refresh_every) + 1
                self.last_dd_info = self.maybe_update_weak_dd(self.cached_batch["sampling_state"], self.current_stage, refreshed=False)

            self.last_refresh_info = {
                "refreshed": int(refresh_due),
                "block_pos": int(block_pos),
                "block_len": int(refresh_every),
                "theta_inner_steps": 1,
            }

            batch = self.cached_batch["batch"]
            if "data_weight" in self.cached_batch:
                self.current_stage["dynamic_data_weight"] = float(self.cached_batch["data_weight"])

            self.params = self.theta_step(self.params, self.geom_state, batch, self.current_stage)

            sampling_state = self.cached_batch["sampling_state"]
            if not self.last_dd_info:
                axis_min, axis_max, cond_max = self._effective_axis_stats(self.geom_np)
                self.last_dd_info = {
                    "status": "held:weak_residual_cvt_dd",
                    "dd_alpha": 0.0,
                    "dd_beta": 0.0,
                    "dd_updated": 0,
                    "axis_min": float(axis_min),
                    "axis_max": float(axis_max),
                    "cond_max": float(cond_max),
                    "uncover": 0.0,
                    "drift": 0.0,
                }
            self.last_sampling_info = {
                "status": "applied:navier_xy_heat_ddim_sampling",
                "score_type": "analytic_fourier+ema+top5_cap",
                "top5_overlap": float(sampling_state["top5_overlap"]),
                "top5_cap_mix": float(sampling_state.get("top5_cap_mix", 0.0)),
                "mix_exp": 1.0,
                "mix_part": 0.0,
                "mix_uni": float(self.current_stage["spec"].uniform_mix),
                "diff_steps": int(self.cfg.diffusion_steps),
                "rho": float(self.current_stage["spec"].ema_rho),
                "retain": float(self.current_stage["spec"].retain_frac),
                "gamma": float(self.current_stage["spec"].weight_gamma),
            }

            metrics = self.fixed_metrics(self.params, self.geom_state, batch, self.current_stage)

            should_eval = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.rel_l2_eval_every == 0)
            should_monitor = (it == 1) or (it == self.cfg.iters) or (it % self.cfg.monitor_eval_every == 0)
            rel_l2_epoch = float("nan")
            snapshot = None
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
                            out_dir / "result_ffusion_navier",
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
                rel_vel = snapshot["rel_vel"] if snapshot is not None else float("nan")
                rel_p = snapshot["rel_p"] if snapshot is not None else float("nan")
                log(
                    f"[ITER {it:6d}/{self.cfg.iters}] "
                    f"loss_total={metrics['loss_total']:.3e}  "
                    f"data_loss={metrics['data_loss']:.3e}  data_weight={metrics['data_weight']:.1f}  balance=off  "
                    f"p_anchor={metrics.get('p_anchor_n', 0)}  tail_anchor={metrics.get('tail_anchor_n', 0)}  "
                    f"p_mean={metrics.get('p_mean_loss', float('nan')):.3e}  p_gauge={metrics.get('p_gauge_loss', float('nan')):.3e}  "
                    f"p_grad={metrics.get('p_grad_loss', float('nan')):.3e}({metrics.get('p_grad_n', 0)})  "
                    f"mse_f_train={metrics['mse_f_blended']:.3e}  "
                    f"mse_fu={metrics['mse_fu']:.3e}  mse_fv={metrics['mse_fv']:.3e}  mse_div={metrics['mse_div']:.3e}  "
                    f"monitor_f={monitor_metrics['monitor_mse_f']:.3e}  monitor_maxr={monitor_metrics['monitor_max_abs_r']:.3e}  "
                    f"tail_loss={metrics['tail_loss']:.3e}  "
                    f"relL2={rel_l2_epoch:.3e}  relVel={rel_vel:.3e}  relP={rel_p:.3e}  "
                    f"theta={th.get('status','na')}  sample={sm.get('status','na')}[{sm.get('score_type','na')}]  "
                    f"dd={dd.get('status','na')}  stage={self.current_stage_name}  "
                    f"theta_opt=hf_jtj_gpu  refresh={self.last_refresh_info['refreshed']}  "
                    f"block={self.last_refresh_info['block_pos']}/{self.last_refresh_info['block_len']}  theta_inner=1  "
                    f"diff_steps={sm.get('diff_steps',0)}  top5={sm.get('top5_overlap', float('nan')):.3f}  top5_cap_mix={sm.get('top5_cap_mix', 0.0):.2f}  "
                    f"mix_exp={sm.get('mix_exp', float('nan')):.2f}  mix_part={sm.get('mix_part', float('nan')):.2f}  mix_uni={sm.get('mix_uni', float('nan')):.2f}  "
                    f"rho={sm.get('rho', float('nan')):.2f}  retain={sm.get('retain', float('nan')):.2f}  gamma={sm.get('gamma', float('nan')):.2f}  "
                    f"dd_alpha={dd.get('dd_alpha', 0.0):.3f}  dd_beta={dd.get('dd_beta', 0.0):.3f}  dd_updated={dd.get('dd_updated', 0)}  "
                    f"axis_min={dd.get('axis_min', float('nan')):.3e}  axis_max={dd.get('axis_max', float('nan')):.3e}  cond_max={dd.get('cond_max', float('nan')):.3e}  "
                    f"uncover={dd.get('uncover', float('nan')):.3e}  drift={dd.get('drift', float('nan')):.3e}  "
                    f"cg_iters={th.get('cg_iters', 0)}  cg_res={th.get('cg_res_norm', float('nan')):.3e}  "
                    f"iter_time={format_seconds(time.time() - iter_t0)}  total={format_seconds(elapsed)}  eta={eta_txt}"
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gt_navier", type=str, default=DEFAULT_GT_NAVIER)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="")

    # Navier-only model. n_balls must remain 10.
    p.add_argument("--n_balls", type=int, default="")
    p.add_argument("--layers", type=int, default="")
    p.add_argument("--width", type=int, default="")
    p.add_argument("--act", type=str, default="silu", choices=["tanh", "relu", "silu"])
    p.add_argument("--freqs", type=str, default="")

    p.add_argument("--iters", type=int, default="")
    p.add_argument("--navier_Re", type=float, default="")
    p.add_argument("--include_time_derivative", action="store_true")

    p.add_argument("--hf_damping_init", type=float, default="")
    p.add_argument("--hf_damping_up", type=float, default="")
    p.add_argument("--hf_damping_down", type=float, default="")
    p.add_argument("--hf_cg_tol", type=float, default="")
    p.add_argument("--hf_line_search_maxiter", type=int, default="")

    p.add_argument("--diffusion_dt", type=float, default="")
    p.add_argument("--diffusion_time_scale", type=float, default="")
    p.add_argument("--diffusion_steps", type=int, default="")
    p.add_argument("--pressure_anchor_frac", type=float, default="")
    p.add_argument("--pressure_mean_n", type=int, default="")
    p.add_argument("--pressure_mean_loss_weight", type=float, default="")
    p.add_argument("--pressure_gauge_loss_weight", type=float, default="")
    p.add_argument("--pressure_grad_n", type=int, default="")
    p.add_argument("--pressure_grad_loss_weight", type=float, default="")
    p.add_argument("--tail_anchor_frac", type=float, default="")
    p.add_argument("--tail_anchor_quantile", type=float, default="")
    p.add_argument("--dd_update_every", type=int, default="")
    p.add_argument("--dd_center_alpha", type=float, default="")
    p.add_argument("--dd_metric_beta", type=float, default="")
    p.add_argument("--sampler_top5_cap", type=float, default="")

    p.add_argument("--rel_l2_eval_every", type=int, default="")
    p.add_argument("--monitor_eval_every", type=int, default="")
    p.add_argument("--monitor_grid_x", type=int, default="")
    p.add_argument("--monitor_grid_y", type=int, default="")
    p.add_argument("--print_every", type=int, default="")
    p.add_argument("--test_batch_size", type=int, default="")
    p.add_argument("--residual_eval_batch_size", type=int, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_handle = setup_logging(args.log_file)
    try:
        if int(args.n_balls) != 10:
            raise ValueError("Navier-Stokes dedicated solver requires --n_balls 10. Do not change it.")

        rng = set_seed(args.seed)
        pde = NavierGTAdapter(
            gt_navier=args.gt_navier,
            Re=args.navier_Re,
            cylinder_cx="",
            cylinder_cy="",
            use_gt_radius=True,
        )
        cfg = Config(
            n_balls=10,
            layers=args.layers,
            width=args.width,
            act=args.act,
            freqs=parse_freqs(args.freqs),
            iters=args.iters,
            Re=args.navier_Re,
            include_time_derivative=bool(args.include_time_derivative),
            residual_comp_weights=(0.1, 0.1, 1.0),
            hf_damping_init=args.hf_damping_init,
            hf_damping_up=args.hf_damping_up,
            hf_damping_down=args.hf_damping_down,
            hf_cg_tol=args.hf_cg_tol,
            hf_line_search_maxiter=args.hf_line_search_maxiter,
            diffusion_dt=args.diffusion_dt,
            diffusion_time_scale=args.diffusion_time_scale,
            diffusion_steps=args.diffusion_steps,
            pressure_anchor_frac=args.pressure_anchor_frac,
            pressure_mean_n=args.pressure_mean_n,
            pressure_mean_loss_weight=args.pressure_mean_loss_weight,
            pressure_gauge_loss_weight=args.pressure_gauge_loss_weight,
            pressure_grad_n=args.pressure_grad_n,
            pressure_grad_loss_weight=args.pressure_grad_loss_weight,
            tail_anchor_frac=args.tail_anchor_frac,
            tail_anchor_quantile=args.tail_anchor_quantile,
            dd_update_every=args.dd_update_every,
            dd_center_alpha=args.dd_center_alpha,
            dd_metric_beta=args.dd_metric_beta,
            sampler_top5_cap=args.sampler_top5_cap,
            rel_l2_eval_every=args.rel_l2_eval_every,
            monitor_eval_every=args.monitor_eval_every,
            monitor_grid_x=args.monitor_grid_x,
            monitor_grid_y=args.monitor_grid_y,
            print_every=args.print_every,
            test_batch_size=args.test_batch_size,
            residual_eval_batch_size=args.residual_eval_batch_size,
        )

        solver = PINNFFusionNavierSolver(pde, cfg, rng)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total_geom_params = solver.geom_np["centers"].shape[0] * (solver.d_in + 1 + solver.d_in * solver.d_in)
        total_params = solver.theta_size + total_geom_params

        log(
            f"[CONFIG] pde=navier_stokes_2d n_balls={cfg.n_balls} layers={cfg.layers} width={cfg.width} "
            f"iters={cfg.iters} theta_optimizer=hf_jtj_gpu refresh_every=stage theta_inner_steps=1 "
            f"Re={pde.Re:.3g} nu={pde.nu:.3e} snapshot_t={pde.matched_time:.6g} steady_snapshot={not cfg.include_time_derivative} "
            f"hf_cg_tol={cfg.hf_cg_tol:.2e} hf_damping_init={cfg.hf_damping_init:.2e} hf_line_search_maxiter={cfg.hf_line_search_maxiter} "
            f"damp(up/down)=({cfg.hf_damping_up:.2f},{cfg.hf_damping_down:.2f}) diffusion_steps={cfg.diffusion_steps} "
            f"staged_grid=((56,32),(72,40),(96,52)) staged_n_f=(512,768,1152) staged_n_data=(768,1024,1792) "
            f"uniform_mix=(0.25,0.20,0.17) loss_weights=off data_weight=1 residual_beta=0 gamma=0 "
            f"ema_rho=(0.20,0.12,0.08) retain=(0.35,0.45,0.55) ultra=off top5_cap={cfg.sampler_top5_cap:.2f} "
            f"res_comp={cfg.residual_comp_weights} data_scales=(u={pde.u_scale:.3e},v={pde.v_scale:.3e},p={pde.p_scale:.3e}) "
            f"pressure_anchor_frac={cfg.pressure_anchor_frac:.2f} p_mean_w={cfg.pressure_mean_loss_weight:.2f} p_gauge_w={cfg.pressure_gauge_loss_weight:.2f} "
            f"p_grad_w={cfg.pressure_grad_loss_weight:.3f} p_grad_n={cfg.pressure_grad_n} p_grad_scales=(px={pde.px_scale:.3e},py={pde.py_scale:.3e}) "
            f"tail_anchor=(stage={cfg.tail_anchor_stage},frac={cfg.tail_anchor_frac:.2f},q={cfg.tail_anchor_quantile:.2f}) "
            f"dd_cvt=(stage={cfg.dd_update_stage},every={cfg.dd_update_every},alpha={cfg.dd_center_alpha:.3f},beta={cfg.dd_metric_beta:.3f},freeze_global={cfg.dd_freeze_global}) "
            f"monitor_grid=({cfg.monitor_grid_x},{cfg.monitor_grid_y}) freqs={cfg.freqs}"
        )
        log(f"[MODEL] approx_total_params={total_params} theta_params={solver.theta_size} geom_params~={total_geom_params}")
        log(
            f"[JAX] forced_device={solver.jax_device} default_backend={jax.default_backend()} "
            f"gt={args.gt_navier} gt_shape=({pde.x.size},{pde.y.size}) radius={pde.radius:.6g} "
            f"feature_mode=fourier dd_template=navier_10_ellipses"
        )

        solver.train(out_dir)
    finally:
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()




