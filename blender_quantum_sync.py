from __future__ import annotations

import json
import os
from pathlib import Path

import bpy


def configure_cycles_gpu(preferred_backend: str = "AUTO") -> bool:
    """Enable Cycles GPU rendering when a compatible device is available."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        return False

    prefs = cycles_addon.preferences
    candidates = ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
    if preferred_backend and preferred_backend != "AUTO":
        candidates = [preferred_backend] + [c for c in candidates if c != preferred_backend]

    backend_set = False
    for backend in candidates:
        try:
            prefs.compute_device_type = backend
            backend_set = True
            break
        except Exception:
            continue

    if not backend_set:
        return False

    try:
        prefs.get_devices()
    except Exception:
        return False

    enabled_any = False
    for device in getattr(prefs, "devices", []):
        try:
            if getattr(device, "type", "CPU") != "CPU":
                device.use = True
                enabled_any = True
        except Exception:
            continue

    if not enabled_any:
        return False

    scene.cycles.device = 'GPU'
    return True


def ensure_quantum_container(name: str = "Volumen_Atomo") -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj

    bpy.ops.mesh.primitive_cube_add(size=8.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def create_quantum_cloud_material(obj: bpy.types.Object, material_name: str = "Quantum_Cloud") -> bpy.types.Material:
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(material_name)

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (500, 0)

    principled_vol = nodes.new(type="ShaderNodeVolumePrincipled")
    principled_vol.location = (220, 0)

    attr_n = nodes.new(type="ShaderNodeAttribute")
    attr_n.attribute_name = "quantum_n"
    attr_n.location = (-700, 250)

    attr_l = nodes.new(type="ShaderNodeAttribute")
    attr_l.attribute_name = "quantum_l"
    attr_l.location = (-700, 50)

    attr_m = nodes.new(type="ShaderNodeAttribute")
    attr_m.attribute_name = "quantum_m"
    attr_m.location = (-700, -150)

    map_n_density = nodes.new(type="ShaderNodeMapRange")
    map_n_density.location = (-420, 220)
    map_n_density.inputs["From Min"].default_value = 1.0
    map_n_density.inputs["From Max"].default_value = 5.0
    map_n_density.inputs["To Min"].default_value = 0.2
    map_n_density.inputs["To Max"].default_value = 1.35
    map_n_density.clamp = True

    map_l_emission = nodes.new(type="ShaderNodeMapRange")
    map_l_emission.location = (-420, 20)
    map_l_emission.inputs["From Min"].default_value = 0.0
    map_l_emission.inputs["From Max"].default_value = 4.0
    map_l_emission.inputs["To Min"].default_value = 0.05
    map_l_emission.inputs["To Max"].default_value = 2.0
    map_l_emission.clamp = True

    abs_m = nodes.new(type="ShaderNodeMath")
    abs_m.operation = 'ABSOLUTE'
    abs_m.location = (-420, -150)

    map_m_aniso = nodes.new(type="ShaderNodeMapRange")
    map_m_aniso.location = (-200, -150)
    map_m_aniso.inputs["From Min"].default_value = 0.0
    map_m_aniso.inputs["From Max"].default_value = 4.0
    map_m_aniso.inputs["To Min"].default_value = -0.5
    map_m_aniso.inputs["To Max"].default_value = 0.65
    map_m_aniso.clamp = True

    links.new(attr_n.outputs["Fac"], map_n_density.inputs["Value"])
    links.new(attr_l.outputs["Fac"], map_l_emission.inputs["Value"])
    links.new(attr_m.outputs["Fac"], abs_m.inputs[0])
    links.new(abs_m.outputs[0], map_m_aniso.inputs["Value"])

    links.new(map_n_density.outputs["Result"], principled_vol.inputs["Density"])
    links.new(map_l_emission.outputs["Result"], principled_vol.inputs["Emission Strength"])
    if "Anisotropy" in principled_vol.inputs:
        links.new(map_m_aniso.outputs["Result"], principled_vol.inputs["Anisotropy"])

    principled_vol.inputs["Color"].default_value = (0.25, 0.6, 1.0, 1.0)
    principled_vol.inputs["Emission Color"].default_value = (0.9, 0.65, 1.0, 1.0)

    links.new(principled_vol.outputs["Volume"], output.inputs["Volume"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat


def ensure_quantum_properties(obj: bpy.types.Object) -> None:
    if "quantum_n" not in obj:
        obj["quantum_n"] = 1.0
    if "quantum_l" not in obj:
        obj["quantum_l"] = 0.0
    if "quantum_m" not in obj:
        obj["quantum_m"] = 0.0


def force_constant_interpolation(obj: bpy.types.Object) -> None:
    anim_data = obj.animation_data
    if anim_data is None or anim_data.action is None:
        return

    for fcurve in anim_data.action.fcurves:
        for key in fcurve.keyframe_points:
            key.interpolation = 'CONSTANT'


def apply_quantum_keyframes_from_json(obj: bpy.types.Object, json_path: str) -> int:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("El JSON de eventos debe ser una lista")

    fps = bpy.context.scene.render.fps
    inserted = 0

    for event in payload:
        timestamp = float(event["timestamp"])
        frame_num = int(timestamp * fps)

        obj["quantum_n"] = float(event["n"])
        obj["quantum_l"] = float(event["l"])
        obj["quantum_m"] = float(event["m"])

        obj.keyframe_insert(data_path='["quantum_n"]', frame=frame_num)
        obj.keyframe_insert(data_path='["quantum_l"]', frame=frame_num)
        obj.keyframe_insert(data_path='["quantum_m"]', frame=frame_num)
        inserted += 1

    force_constant_interpolation(obj)
    return inserted


def setup_vse_audio_sync(audio_path: str, channel: int = 1, frame_start: int = 1) -> bpy.types.SoundSequence:
    scene = bpy.context.scene
    if scene.sequence_editor is None:
        scene.sequence_editor_create()

    seq_editor = scene.sequence_editor
    absolute_audio = os.path.abspath(audio_path)

    existing = next(
        (
            s
            for s in seq_editor.sequences_all
            if s.type == 'SOUND' and bpy.path.abspath(s.sound.filepath) == absolute_audio
        ),
        None,
    )

    if existing is not None:
        strip = existing
    else:
        strip = seq_editor.sequences.new_sound(
            name=Path(absolute_audio).stem,
            filepath=absolute_audio,
            channel=channel,
            frame_start=frame_start,
        )

    scene.sync_mode = 'AUDIO_SYNC'
    return strip


def run_pipeline(events_json_path: str, audio_path: str, use_gpu_render: bool = False, gpu_backend: str = "AUTO") -> None:
    obj = ensure_quantum_container("Volumen_Atomo")
    ensure_quantum_properties(obj)
    create_quantum_cloud_material(obj, "Quantum_Cloud")

    total = apply_quantum_keyframes_from_json(obj, events_json_path)
    setup_vse_audio_sync(audio_path=audio_path, frame_start=1)

    gpu_enabled = False
    if use_gpu_render:
        gpu_enabled = configure_cycles_gpu(preferred_backend=gpu_backend)

    print("===============================================")
    print(f"Eventos aplicados al volumen: {total}")
    print(f"Audio sincronizado: {os.path.abspath(audio_path)}")
    print("Modo de sincronización: AUDIO_SYNC")
    print("Interpolación de keyframes: CONSTANT")
    if use_gpu_render:
        print(f"GPU en Cycles: {'ACTIVADA' if gpu_enabled else 'NO DISPONIBLE'}")
    print("===============================================")


# Uso en Blender (Scripting):
# run_pipeline(
#     events_json_path="/ruta/a/secuencia_estados.json",
#     audio_path="/ruta/a/Sparks.wav",
#     use_gpu_render=True,
#     gpu_backend="AUTO",  # AUTO | OPTIX | CUDA | HIP | METAL | ONEAPI
# )
