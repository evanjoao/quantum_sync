from __future__ import annotations

import os
from pathlib import Path

import bpy


def import_vdb_sequence(vdb_first_file: str) -> bpy.types.Object:
    path = Path(vdb_first_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo VDB inicial: {path}")

    bpy.ops.object.volume_import(filepath=str(path))
    obj = bpy.context.active_object
    if obj is None or obj.type != 'VOLUME':
        raise RuntimeError("No se pudo importar el volumen VDB")
    return obj


def configure_vdb_sequence(
    volume_obj: bpy.types.Object,
    frame_start: int = 1,
    frame_offset: int = 0,
    sequence_mode: str = 'CLIP',
) -> None:
    volume_data = volume_obj.data

    if hasattr(volume_data, "sequence_mode"):
        volume_data.sequence_mode = sequence_mode
    if hasattr(volume_data, "frame_start"):
        volume_data.frame_start = frame_start
    if hasattr(volume_data, "frame_offset"):
        volume_data.frame_offset = frame_offset


def bind_timestamps_to_vdb(
    volume_obj: bpy.types.Object,
    timestamps_seconds: list[float],
) -> int:
    fps = bpy.context.scene.render.fps
    applied = 0

    for idx, t_sec in enumerate(timestamps_seconds):
        frame_num = int(float(t_sec) * fps)
        if hasattr(volume_obj.data, "frame_offset"):
            volume_obj.data.frame_offset = idx
            volume_obj.data.keyframe_insert(data_path="frame_offset", frame=frame_num)
            applied += 1

    if volume_obj.data.animation_data and volume_obj.data.animation_data.action:
        for fcurve in volume_obj.data.animation_data.action.fcurves:
            for key in fcurve.keyframe_points:
                key.interpolation = 'CONSTANT'

    return applied


def run_openvdb_pipeline(vdb_first_file: str, timestamps_seconds: list[float]) -> bpy.types.Object:
    volume_obj = import_vdb_sequence(vdb_first_file)
    configure_vdb_sequence(volume_obj, frame_start=1, frame_offset=0, sequence_mode='CLIP')
    n = bind_timestamps_to_vdb(volume_obj, timestamps_seconds)

    print("===============================================")
    print(f"VDB importado: {os.path.abspath(vdb_first_file)}")
    print(f"Eventos de offset aplicados: {n}")
    print("Interpolación de keyframes: CONSTANT")
    print("===============================================")

    return volume_obj
