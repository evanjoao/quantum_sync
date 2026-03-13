#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

REQUIRED_PACKAGES = ["librosa", "soundfile", "numpy", "scipy", "pyopenvdb"]


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_checked(cmd: list[str], description: str) -> subprocess.CompletedProcess[str]:
    logging.info("%s", description)
    logging.debug("CMD: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        logging.error("Falló: %s", description)
        logging.error("stdout: %s", completed.stdout.strip())
        logging.error("stderr: %s", completed.stderr.strip())
        raise RuntimeError(f"Error ejecutando: {' '.join(cmd)}")

    if completed.stdout.strip():
        logging.debug("stdout: %s", completed.stdout.strip())
    if completed.stderr.strip():
        logging.debug("stderr: %s", completed.stderr.strip())
    return completed


def discover_blender_bin(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        blender = Path(explicit_path).expanduser().resolve()
        if blender.exists():
            return blender
        raise FileNotFoundError(f"Blender no encontrado en ruta explícita: {blender}")

    candidate = shutil.which("blender")
    if candidate:
        return Path(candidate)

    common = [
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
    ]
    for path in common:
        if path.exists():
            return path

    raise FileNotFoundError("No se encontró el ejecutable de Blender en PATH o rutas comunes.")


def get_blender_python(blender_bin: Path) -> Path:
    expr = "import json,sys;print(json.dumps({'python':sys.executable}))"
    completed = run_checked(
        [str(blender_bin), "--background", "--python-expr", expr],
        "Detectando Python embebido de Blender",
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            py = payload.get("python")
            if py:
                python_path = Path(py)
                if python_path.exists():
                    logging.info("Python de Blender detectado: %s", python_path)
                    return python_path
        except json.JSONDecodeError:
            continue

    raise RuntimeError("No se pudo parsear la ruta de Python embebido desde la salida de Blender.")


def install_packages(blender_python: Path, packages: Iterable[str]) -> None:
    run_checked([str(blender_python), "-m", "ensurepip", "--upgrade"], "Inicializando pip en Blender")
    run_checked(
        [str(blender_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Actualizando pip/setuptools/wheel",
    )
    run_checked(
        [str(blender_python), "-m", "pip", "install", *packages],
        "Instalando dependencias del pipeline audiovisual",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Instala dependencias DSP/cuánticas dentro del Python embebido de Blender."
    )
    parser.add_argument("--blender-bin", help="Ruta al binario de Blender", default=None)
    parser.add_argument(
        "--packages",
        nargs="*",
        default=REQUIRED_PACKAGES,
        help="Lista de paquetes a instalar",
    )
    parser.add_argument("--verbose", action="store_true", help="Habilita logging detallado")
    args = parser.parse_args()

    configure_logging(args.verbose)

    try:
        blender_bin = discover_blender_bin(args.blender_bin)
        logging.info("Blender detectado: %s", blender_bin)

        blender_python = get_blender_python(blender_bin)
        install_packages(blender_python, args.packages)

        logging.info("Dependencias instaladas correctamente en Python de Blender.")
        return 0
    except Exception as exc:
        logging.exception("Configuración fallida: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
