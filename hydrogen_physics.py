#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.special import factorial, genlaguerre

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
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(1234)

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
