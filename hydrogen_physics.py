#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import factorial, genlaguerre

try:  # Optional GPU backend
    import cupy as cp
    from cupyx.scipy import special as cpx_special

    _HAS_CUPY = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    cpx_special = None
    _HAS_CUPY = False

try:
    # SciPy >= 1.15 replaced sph_harm with sph_harm_y.
    from scipy.special import sph_harm_y as _sph_harm

    def _eval_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm(l, m, theta, phi)

except ImportError:  # pragma: no cover - older SciPy
    from scipy.special import sph_harm as _sph_harm

    def _eval_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return _sph_harm(m, l, phi, theta)

A0_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class QuantumState:
    n: int
    l: int
    m: int


def is_cuda_backend_available() -> bool:
    if not _HAS_CUPY:
        return False
    if cpx_special is None:
        return False

    has_sph_harm = hasattr(cpx_special, "sph_harm") or hasattr(cpx_special, "sph_harm_y")
    if not has_sph_harm:
        return False

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        # Force a tiny RNG call to ensure curand is available.
        _ = cp.random.random((1,), dtype=cp.float32)
    except Exception:
        return False

    return True


def _cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(np.divide(z, r, out=np.zeros_like(r), where=r > 1e-12), -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2.0 * np.pi)
    return r, theta, phi


def radial_wavefunction(n: int, l: int, r_angstrom: np.ndarray) -> np.ndarray:
    rho = 2.0 * r_angstrom / (n * A0_ANGSTROM)
    prefactor = np.sqrt(
        (2.0 / (n * A0_ANGSTROM)) ** 3
        * factorial(n - l - 1)
        / (2.0 * n * factorial(n + l))
    )
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return prefactor * np.exp(-rho / 2.0) * (rho**l) * laguerre_poly


def hydrogen_wavefunction(n: int, l: int, m: int, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r, theta, phi = _cartesian_to_spherical(x, y, z)
    radial = radial_wavefunction(n, l, r)
    angular = _eval_sph_harm(l, m, theta, phi)
    return radial * angular


def probability_density(n: int, l: int, m: int, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    psi = hydrogen_wavefunction(n, l, m, x, y, z)
    return np.abs(psi) ** 2


def sample_points_from_density(
    state: QuantumState,
    num_points: int = 100_000,
    box_extent_angstrom: float | None = None,
    batch_size: int = 200_000,
    rng: np.random.Generator | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device.lower() == "cuda" and is_cuda_backend_available():
        try:
            return _sample_points_from_density_cuda(
                state=state,
                num_points=num_points,
                box_extent_angstrom=box_extent_angstrom,
                batch_size=batch_size,
                seed=seed,
            )
        except Exception as exc:
            print(f"[WARN] CUDA no disponible en runtime ({exc}). Se usara CPU.")

    # CPU fallback (or explicit CPU request)
    return _sample_points_from_density_cpu(
        state=state,
        num_points=num_points,
        box_extent_angstrom=box_extent_angstrom,
        batch_size=batch_size,
        rng=rng,
        seed=seed,
    )


def _sample_points_from_density_cpu(
    state: QuantumState,
    num_points: int,
    box_extent_angstrom: float | None,
    batch_size: int,
    rng: np.random.Generator | None,
    seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(1234 if seed is None else seed)

    if box_extent_angstrom is None:
        box_extent_angstrom = max(8.0, float(state.n) * 10.0)

    accepted_xyz: list[np.ndarray] = []
    accepted_signs: list[np.ndarray] = []

    while sum(chunk.shape[0] for chunk in accepted_xyz) < num_points:
        xyz = rng.uniform(-box_extent_angstrom, box_extent_angstrom, size=(batch_size, 3))
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        psi = hydrogen_wavefunction(state.n, state.l, state.m, x, y, z)
        density = np.abs(psi) ** 2

        max_density = float(np.max(density)) if density.size else 0.0
        if max_density <= 0.0:
            continue

        threshold = rng.uniform(0.0, max_density, size=density.shape[0])
        mask = density > threshold
        if not np.any(mask):
            continue

        accepted_xyz.append(xyz[mask])
        accepted_signs.append(np.sign(np.real(psi[mask])).astype(np.float32))

    coords = np.concatenate(accepted_xyz, axis=0)[:num_points].astype(np.float32)
    signs = np.concatenate(accepted_signs, axis=0)[:num_points].astype(np.float32)

    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs * 35.0

    return coords, signs


def _cartesian_to_spherical_gpu(x, y, z):
    r = cp.sqrt(x * x + y * y + z * z)
    z_over_r = cp.where(r > 1e-12, z / r, cp.zeros_like(r))
    theta = cp.arccos(cp.clip(z_over_r, -1.0, 1.0))
    phi = cp.mod(cp.arctan2(y, x), 2.0 * cp.pi)
    return r, theta, phi


def _eval_sph_harm_gpu(l: int, m: int, theta, phi):
    if hasattr(cpx_special, "sph_harm_y"):
        return cpx_special.sph_harm_y(l, m, theta, phi)
    return cpx_special.sph_harm(m, l, phi, theta)


def _radial_wavefunction_gpu(n: int, l: int, r_angstrom):
    rho = 2.0 * r_angstrom / (n * A0_ANGSTROM)
    prefactor = math.sqrt(
        (2.0 / (n * A0_ANGSTROM)) ** 3
        * float(factorial(n - l - 1))
        / (2.0 * n * float(factorial(n + l)))
    )
    k = n - l - 1
    alpha = float(2 * l + 1)
    laguerre = _assoc_laguerre_gpu(k, alpha, rho)
    return prefactor * cp.exp(-rho / 2.0) * (rho**l) * laguerre


def _assoc_laguerre_gpu(k: int, alpha: float, x):
    # Recurrence for generalized Laguerre polynomial L_k^alpha(x)
    if k == 0:
        return cp.ones_like(x)
    if k == 1:
        return (1.0 + alpha) - x

    l_nm2 = cp.ones_like(x)
    l_nm1 = (1.0 + alpha) - x
    for n_idx in range(2, k + 1):
        n_float = float(n_idx)
        l_n = (((2.0 * n_float - 1.0 + alpha) - x) * l_nm1 - (n_float - 1.0 + alpha) * l_nm2) / n_float
        l_nm2 = l_nm1
        l_nm1 = l_n
    return l_nm1


def _hydrogen_wavefunction_gpu(n: int, l: int, m: int, x, y, z):
    r, theta, phi = _cartesian_to_spherical_gpu(x, y, z)
    radial = _radial_wavefunction_gpu(n, l, r)
    angular = _eval_sph_harm_gpu(l, m, theta, phi)
    return radial * angular


def _sample_points_from_density_cuda(
    state: QuantumState,
    num_points: int,
    box_extent_angstrom: float | None,
    batch_size: int,
    seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if box_extent_angstrom is None:
        box_extent_angstrom = max(8.0, float(state.n) * 10.0)

    rng = cp.random.RandomState(1234 if seed is None else int(seed))
    accepted_xyz = []
    accepted_signs = []
    accepted_count = 0

    while accepted_count < num_points:
        xyz = rng.uniform(-box_extent_angstrom, box_extent_angstrom, size=(batch_size, 3), dtype=cp.float32)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        psi = _hydrogen_wavefunction_gpu(state.n, state.l, state.m, x, y, z)
        density = cp.abs(psi) ** 2

        max_density = float(cp.max(density).item()) if density.size else 0.0
        if max_density <= 0.0:
            continue

        threshold = rng.uniform(0.0, max_density, size=density.shape[0], dtype=cp.float32)
        mask = density > threshold
        if not bool(cp.any(mask).item()):
            continue

        xyz_ok = xyz[mask]
        signs_ok = cp.sign(cp.real(psi[mask])).astype(cp.float32)
        accepted_xyz.append(xyz_ok)
        accepted_signs.append(signs_ok)
        accepted_count += int(xyz_ok.shape[0])

    coords_cp = cp.concatenate(accepted_xyz, axis=0)[:num_points].astype(cp.float32)
    signs_cp = cp.concatenate(accepted_signs, axis=0)[:num_points].astype(cp.float32)

    max_abs = float(cp.max(cp.abs(coords_cp)).item())
    if max_abs > 0.0:
        coords_cp = coords_cp / max_abs * 35.0

    coords = cp.asnumpy(coords_cp)
    signs = cp.asnumpy(signs_cp)
    return coords, signs
