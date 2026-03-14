#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from quantum_sync.hydrogen_physics import QuantumState, is_cuda_backend_available, sample_points_from_density


def _build_state(event: Dict[str, Any]) -> QuantumState:
    n = int(event["n"])
    l = int(event["l"])
    m = int(event["m"])

    if n < 1:
        n = 1
    if l < 0:
        l = 0
    if l >= n:
        l = n - 1
    if m < -l:
        m = -l
    if m > l:
        m = l

    return QuantumState(n=n, l=l, m=m)


def generar_npz_desde_eventos(
    events_json_path: str,
    output_npz_path: str,
    num_points: int = 100_000,
    seed: int = 1234,
    device: str = "cpu",
) -> str:
    events: List[Dict[str, Any]] = json.loads(Path(events_json_path).read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("El JSON de eventos debe ser una lista")

    rng = np.random.default_rng(seed)
    resolved_device = "cuda" if device.lower() == "cuda" and is_cuda_backend_available() else "cpu"

    if device.lower() == "cuda" and resolved_device != "cuda":
        print("[WARN] CUDA/CuPy no disponible o incompleto. Se usara CPU.")

    coords_frames: list[np.ndarray] = []
    signs_frames: list[np.ndarray] = []

    for event in events:
        state = _build_state(event)
        coords, signs = sample_points_from_density(
            state=state,
            num_points=num_points,
            batch_size=max(200_000, num_points * 2),
            rng=rng,
            device=resolved_device,
            seed=seed,
        )
        coords_frames.append(coords)
        signs_frames.append(signs)

    coords_arr = np.stack(coords_frames, axis=0).astype(np.float32)
    signs_arr = np.stack(signs_frames, axis=0).astype(np.float32)

    out = Path(output_npz_path)
    np.savez_compressed(out, coords=coords_arr, signs=signs_arr)
    print(f"Backend de computo usado: {resolved_device.upper()}")
    return str(out.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera atomo_frames.npz (coords/signs) a partir de eventos cuánticos n,l,m"
    )
    parser.add_argument("--events", required=True, help="JSON enriquecido con n,l,m")
    parser.add_argument("--output", default="atomo_frames_music.npz", help="NPZ de salida")
    parser.add_argument("--points", type=int, default=50_000, help="Puntos por frame")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Backend de computo")
    args = parser.parse_args()

    output = generar_npz_desde_eventos(
        events_json_path=args.events,
        output_npz_path=args.output,
        num_points=args.points,
        seed=args.seed,
        device=args.device,
    )
    print(f"NPZ generado en: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
