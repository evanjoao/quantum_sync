#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from quantum_sync.audio_quantum_analyzer import AudioQuantumAnalyzer
from quantum_sync.generar_frames_cuanticos import generar_npz_desde_eventos
from quantum_sync.mapeo_cuantico import MapeoCuantico


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline completo: audio -> eventos -> estados cuánticos -> NPZ Blender")
    parser.add_argument("--audio", required=True, help="Ruta absoluta de Sparks.wav")
    parser.add_argument("--workdir", default="/home/evanj/Latex/quantum_sync", help="Directorio de salida")
    parser.add_argument("--points", type=int, default=50_000, help="Puntos por frame en NPZ")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Backend de computo para generar NPZ")
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    events_json = workdir / "secuencia_estados.json"
    enriched_json = workdir / "secuencia_estados_enriquecida.json"
    out_npz = Path("/home/evanj/Latex/atomo_frames_music.npz")

    analyzer = AudioQuantumAnalyzer(audio_path=args.audio)
    events = analyzer.analyze()
    analyzer.save_events_json(events, str(events_json))

    MapeoCuantico.procesar_json(str(events_json), output_json=str(enriched_json), seed=args.seed)

    generar_npz_desde_eventos(
        events_json_path=str(enriched_json),
        output_npz_path=str(out_npz),
        num_points=args.points,
        seed=args.seed,
        device=args.device,
    )

    print("===============================================")
    print(f"Eventos DSP: {events_json}")
    print(f"Eventos cuánticos: {enriched_json}")
    print(f"Frames para Blender: {out_npz}")
    print(f"Backend de computo: {args.device.upper()}")
    print("===============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
