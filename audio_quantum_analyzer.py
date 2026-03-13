#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np


@dataclass
class AudioEvent:
    event_id: int
    timestamp: float
    pitch_dominant_id: int


class AudioQuantumAnalyzer:
    def __init__(self, audio_path: str, sr: int | None = None, hop_length: int = 512) -> None:
        self.audio_path = audio_path
        self.sr_target = sr
        self.hop_length = hop_length

        self.y: np.ndarray | None = None
        self.sr: int | None = None
        self.y_harmonic: np.ndarray | None = None
        self.onset_frames: np.ndarray | None = None
        self.timestamps: np.ndarray | None = None
        self.chroma: np.ndarray | None = None

    def load_audio(self) -> None:
        y, sr = librosa.load(self.audio_path, sr=self.sr_target, mono=True)
        self.y = y
        self.sr = sr

    def isolate_harmonic_content(self) -> None:
        if self.y is None:
            raise RuntimeError("Audio no cargado. Ejecuta load_audio() primero.")
        y_harmonic, _ = librosa.effects.hpss(self.y)
        self.y_harmonic = y_harmonic

    def detect_onsets(self) -> None:
        if self.y_harmonic is None:
            raise RuntimeError("Componente armónica no disponible.")
        self.onset_frames = librosa.onset.onset_detect(
            y=self.y_harmonic,
            sr=self.sr,
            hop_length=self.hop_length,
            units="frames",
            backtrack=False,
        )

    def convert_frames_to_time(self) -> None:
        if self.onset_frames is None:
            raise RuntimeError("No hay onsets detectados.")
        self.timestamps = librosa.frames_to_time(
            self.onset_frames,
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def compute_chroma(self) -> None:
        if self.y_harmonic is None:
            raise RuntimeError("Componente armónica no disponible.")
        self.chroma = librosa.feature.chroma_cqt(
            y=self.y_harmonic,
            sr=self.sr,
            hop_length=self.hop_length,
        )

    def build_events(self) -> List[AudioEvent]:
        if self.timestamps is None or self.chroma is None or self.onset_frames is None:
            raise RuntimeError("Faltan etapas previas para construir eventos.")

        total_frames = self.chroma.shape[1]
        events: List[AudioEvent] = []

        for idx, frame in enumerate(self.onset_frames):
            frame_index = int(np.clip(frame, 0, total_frames - 1))
            pitch_class_zero_based = int(np.argmax(self.chroma[:, frame_index]))
            pitch_dominant_id = pitch_class_zero_based + 1
            events.append(
                AudioEvent(
                    event_id=idx + 1,
                    timestamp=float(self.timestamps[idx]),
                    pitch_dominant_id=pitch_dominant_id,
                )
            )

        return events

    def analyze(self) -> List[AudioEvent]:
        self.load_audio()
        self.isolate_harmonic_content()
        self.detect_onsets()
        self.convert_frames_to_time()
        self.compute_chroma()
        return self.build_events()

    @staticmethod
    def save_events_json(events: List[AudioEvent], output_path: str) -> None:
        payload = [asdict(event) for event in events]
        Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analizador DSP para sincronización cuántica audiovisual")
    parser.add_argument("--audio", required=True, help="Ruta a Sparks.wav (o cualquier audio)")
    parser.add_argument(
        "--output",
        default="secuencia_estados.json",
        help="Ruta de salida JSON para eventos temporales",
    )
    parser.add_argument("--sr", type=int, default=None, help="Sample rate objetivo (None respeta el original)")
    parser.add_argument("--hop", type=int, default=512, help="Hop length para STFT/CQT")
    args = parser.parse_args()

    analyzer = AudioQuantumAnalyzer(audio_path=args.audio, sr=args.sr, hop_length=args.hop)
    events = analyzer.analyze()
    analyzer.save_events_json(events, args.output)

    print(f"Eventos detectados: {len(events)}")
    print(f"JSON generado en: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
