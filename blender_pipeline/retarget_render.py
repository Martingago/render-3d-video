"""
Ejecutar solo dentro de Blender:

  blender --background --python blender_pipeline/retarget_render.py --
      <animation.json> <entrada.fbx|glb> <salida.glb> <salida.mp4> <fps>

bpy no está disponible en el intérprete del servidor Flask.
"""

from __future__ import annotations

import json
import os
import sys


def _argv_after_double_dash() -> list:
    if "--" not in sys.argv:
        return []
    i = sys.argv.index("--") + 1
    return sys.argv[i:]


def _pick_armature():
    import bpy

    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        return None
    if len(arms) == 1:
        return arms[0]
    for o in arms:
        if "mixamorig" in o.name.lower() or o.name.lower() == "armature":
            return o
    return max(arms, key=lambda o: len(o.data.bones))


def _resolve_pose_bone(arm, json_bone_name: str):
    """
    Resuelve el hueso del rig: nombre JSON (p. ej. Hips) o con prefijo Mixamo mixamorig:Hips.
    """
    pbs = arm.pose.bones
    name = json_bone_name.strip()
    candidates = [
        name,
        "mixamorig:" + name,
        "Mixamorig:" + name,
    ]
    if name.lower().startswith("mixamorig:"):
        base = name.split(":", 1)[-1]
        candidates.insert(0, name)
        candidates.extend([base, "mixamorig:" + base])

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    for c in ordered:
        if c in pbs:
            return pbs[c]

    for bn, pb in pbs.items():
        if bn == name:
            return pb
        if ":" in bn and bn.split(":", 1)[-1] == name:
            return pb
    return None


def _ensure_armature_action(arm):
    import bpy

    if arm.animation_data is None:
        arm.animation_data_create()
    if arm.animation_data.action is None:
        arm.animation_data.action = bpy.data.actions.new(name="RetargetAction")


def _set_all_bones_quaternion_mode(arm):
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"


def main() -> None:
    import bpy
    from mathutils import Quaternion, Vector

    args = _argv_after_double_dash()
    if len(args) < 5:
        print(
            "Uso: blender -b -P retarget_render.py -- "
            "<json> <in.fbx|glb> <out.glb> <out.mp4> <fps>",
            file=sys.stderr,
        )
        sys.exit(1)

    json_path = os.path.abspath(args[0])
    in_model = os.path.abspath(args[1])
    out_glb = os.path.abspath(args[2])
    out_mp4 = os.path.abspath(args[3])
    fps = float(args[4])

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    frames = doc["frames"]
    if not frames:
        print("JSON sin frames.", file=sys.stderr)
        sys.exit(2)

    ext = os.path.splitext(in_model)[1].lower()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    if ext == ".fbx":
        bpy.ops.import_scene.fbx(
            filepath=in_model,
            automatic_bone_orientation=True,
            use_prepost_rot=True,
        )
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=in_model)
    else:
        print(f"Formato no soportado: {ext} (use .fbx o .glb)", file=sys.stderr)
        sys.exit(4)

    arm = _pick_armature()
    if arm is None:
        print("No se encontró ningún Armature en el modelo importado.", file=sys.stderr)
        sys.exit(3)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    _set_all_bones_quaternion_mode(arm)
    _ensure_armature_action(arm)

    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = len(frames)

    for fi, frame in enumerate(frames):
        fnum = 1 + fi
        scene.frame_set(fnum)

        root_t = frame.get("root_translation")
        if root_t is not None:
            arm.location = Vector(
                (float(root_t[0]), float(root_t[1]), float(root_t[2]))
            )
            arm.keyframe_insert(data_path="location", frame=fnum)

        bones_data = frame.get("bones", {})
        for json_bone_name, qlist in bones_data.items():
            pb = _resolve_pose_bone(arm, json_bone_name)
            if pb is None:
                continue
            pb.rotation_mode = "QUATERNION"
            qw, qx, qy, qz = (
                float(qlist[0]),
                float(qlist[1]),
                float(qlist[2]),
                float(qlist[3]),
            )
            pb.rotation_quaternion = Quaternion((qw, qx, qy, qz))
            pb.keyframe_insert(data_path="rotation_quaternion", frame=fnum)

    bpy.ops.object.mode_set(mode="OBJECT")

    os.makedirs(os.path.dirname(out_glb) or ".", exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    export_kw = dict(
        filepath=out_glb,
        export_format="GLB",
        use_selection=False,
        export_animations=True,
    )
    try:
        bpy.ops.export_scene.gltf(**export_kw, export_force_sampling=True)
    except TypeError:
        bpy.ops.export_scene.gltf(**export_kw)

    bpy.ops.object.light_add(type="SUN", location=(5.0, -3.0, 10.0))
    sun = bpy.context.active_object
    sun.data.energy = 2.5

    bpy.ops.object.camera_add(location=(2.5, -3.2, 1.75))
    cam = bpy.context.active_object
    scene.camera = cam
    tt = cam.constraints.new(type="TRACK_TO")
    tt.target = arm
    tt.track_axis = "TRACK_NEGATIVE_Z"
    tt.up_axis = "UP_Y"

    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        scene.display.shading.color_type = "MATERIAL"
    except Exception:
        pass

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    mp4_path = out_mp4 if out_mp4.lower().endswith(".mp4") else out_mp4 + ".mp4"
    scene.render.image_settings.file_format = "FFMPEG"
    ff = scene.render.ffmpeg
    if hasattr(ff, "format"):
        ff.format = "MPEG4"
    if hasattr(ff, "codec"):
        ff.codec = "H264"
    scene.render.filepath = mp4_path

    bpy.ops.render.render(animation=True)

    print("OK:", out_glb, mp4_path)


if __name__ == "__main__":
    main()
