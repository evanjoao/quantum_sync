#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PLANCK_EV_S = 4.135667696e-15



@dataclass(frozen=True)
class EstadoCuantico:
    n: int
    l: int
    m: int


class MapeoCuantico:
    """Mapeo físico de pitch dominante (1..12) a transición cuántica (n,l,m)."""

    # Convención: C=1, C#=2, ..., B=12
    PITCH_MIDI_BASE = {
        1: 60,
        2: 61,
        3: 62,
        4: 63,
        5: 64,
        6: 65,
        7: 66,
        8: 67,
        9: 68,
        10: 69,
        11: 70,
        12: 71,
    }

    @staticmethod
    def energy_level_ev(n: int) -> float:
        return -13.6 / float(n * n)

    @staticmethod
    def delta_energy_ev(n_initial: int, n_final: int) -> float:
        return MapeoCuantico.energy_level_ev(n_final) - MapeoCuantico.energy_level_ev(n_initial)

    @staticmethod
    def pitch_to_frequency_hz(pitch_dominant_id: int, octave_shift: int = 0) -> float:
        if pitch_dominant_id not in MapeoCuantico.PITCH_MIDI_BASE:
            pitch_dominant_id = 1
        midi = MapeoCuantico.PITCH_MIDI_BASE[pitch_dominant_id] + (12 * octave_shift)
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    @staticmethod
    def photon_energy_ev_from_pitch(pitch_dominant_id: int, octave_shift: int = 3) -> float:
        frequency = MapeoCuantico.pitch_to_frequency_hz(pitch_dominant_id, octave_shift=octave_shift)
        return PLANCK_EV_S * frequency

    @staticmethod
    def resolver_transicion_n(n_current: int, pitch_dominant_id: int, max_n: int = 5) -> int:
        photon_ev = MapeoCuantico.photon_energy_ev_from_pitch(pitch_dominant_id)
        candidates: List[Tuple[float, int]] = []

        for n_target in range(1, max_n + 1):
            if n_target == n_current:
                continue
            delta = abs(MapeoCuantico.delta_energy_ev(n_current, n_target))
            error = abs(delta - photon_ev)
            candidates.append((error, n_target))

        if not candidates:
            return n_current

        candidates.sort(key=lambda item: item[0])
        best_error, best_n = candidates[0]

        # Si no hay transición físicamente cercana, conserva estado.
        # El umbral se relaja con n para evitar bloqueos en secuencias largas.
        threshold = 0.35 / max(1, n_current)
        if best_error > threshold:
            return n_current
        return best_n

    @staticmethod
    def generar_l_m(n: int, pitch_dominant_id: int, rng: random.Random) -> Tuple[int, int]:
        if n <= 1:
            return 0, 0

        l = min(n - 1, max(0, pitch_dominant_id % n))
        if l == 0:
            return 0, 0

        base = (pitch_dominant_id % (2 * l + 1)) - l
        jitter = rng.choice([-1, 0, 1])
        m = int(max(-l, min(l, base + jitter)))
        return l, m

    @staticmethod
    def generar_estado(pitch_dominant_id: int, n_current: int, rng: random.Random) -> EstadoCuantico:
        n = MapeoCuantico.resolver_transicion_n(n_current=n_current, pitch_dominant_id=pitch_dominant_id)
        l, m = MapeoCuantico.generar_l_m(n, pitch_dominant_id, rng)
        return EstadoCuantico(n=n, l=l, m=m)

    @staticmethod
    def enriquecer_eventos(events: List[Dict[str, Any]], seed: int | None = None) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        enriched: List[Dict[str, Any]] = []
        n_current = 1

        for event in events:
            pitch_id = int(event["pitch_dominant_id"])
            estado = MapeoCuantico.generar_estado(pitch_id, n_current=n_current, rng=rng)
            n_current = estado.n

            copy_event = dict(event)
            copy_event["n"] = estado.n
            copy_event["l"] = estado.l
            copy_event["m"] = estado.m
            copy_event["photon_energy_ev"] = MapeoCuantico.photon_energy_ev_from_pitch(pitch_id)
            enriched.append(copy_event)

        return enriched

    @staticmethod
    def procesar_json(input_json: str, output_json: str | None = None, seed: int | None = None) -> str:
        src = Path(input_json)
        if not src.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {src}")

        raw_events = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(raw_events, list):
            raise ValueError("El JSON de entrada debe ser una lista de eventos.")

        enriched = MapeoCuantico.enriquecer_eventos(raw_events, seed=seed)
        dst = Path(output_json) if output_json else src
        dst.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(dst.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Mapeo de eventos acústicos a estados cuánticos (n,l,m)")
    parser.add_argument("--input", default="secuencia_estados.json", help="JSON de entrada con timestamp y pitch")
    parser.add_argument("--output", default=None, help="JSON de salida enriquecido (por defecto sobrescribe entrada)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    out_path = MapeoCuantico.procesar_json(args.input, output_json=args.output, seed=args.seed)
    print(f"Eventos enriquecidos guardados en: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
