"""
Puente Pose -> JSON para Blender (Mixamo-like).
Convierte espacio escena (Y arriba) a Blender (Z arriba, -Y delante) y estima cuaterniones por hueso.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 1

# Escena (skeleton_mapper): x derecha, y arriba, z ~profundidad
# Blender típico Mixamo: x derecha, z arriba, -y delante de la cámara
R_SCENE_TO_BLENDER = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def scene_point_to_blender(p: np.ndarray) -> np.ndarray:
    v = np.asarray(p, dtype=np.float64).reshape(3)
    return R_SCENE_TO_BLENDER @ v


def scene_rotation_to_blender(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return R_SCENE_TO_BLENDER @ R @ R_SCENE_TO_BLENDER.T


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def quat_between_vectors(rest: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Cuaternión [w, x, y, z] que rota `rest` hacia `target` (ambos 3D, mismo espacio).
    """
    a = _normalize(np.asarray(rest, dtype=np.float64).reshape(3))
    b = _normalize(np.asarray(target, dtype=np.float64).reshape(3))
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        orth = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            orth = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, orth))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
    w = 1.0 + dot
    x, y, z = cross
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotación 3x3 -> cuaternión [w,x,y,z] (Shepperd)."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# Direcciones "reposo" aproximadas (T-pose Mixamo en espacio Blender del puente).
# Ajustables si el rig base difiere.
REST_BONE_DIRECTIONS: Dict[str, np.ndarray] = {
    "Hips": np.array([0.0, 0.0, 1.0]),
    "Spine": np.array([0.0, 0.0, 1.0]),
    "Spine1": np.array([0.0, 0.0, 1.0]),
    "Spine2": np.array([0.0, 0.0, 1.0]),
    "Neck": np.array([0.0, 0.0, 1.0]),
    "Head": np.array([0.0, 0.0, 1.0]),
    "LeftArm": np.array([-1.0, 0.0, 0.0]),
    "LeftForeArm": np.array([-1.0, 0.0, 0.0]),
    "LeftHand": np.array([-1.0, 0.0, 0.0]),
    "RightArm": np.array([1.0, 0.0, 0.0]),
    "RightForeArm": np.array([1.0, 0.0, 0.0]),
    "RightHand": np.array([1.0, 0.0, 0.0]),
    "LeftUpLeg": np.array([-0.25, 0.0, -0.97]),
    "LeftLeg": np.array([0.0, 0.0, -1.0]),
    "LeftFoot": np.array([0.05, 0.12, -0.99]),
    "RightUpLeg": np.array([0.25, 0.0, -0.97]),
    "RightLeg": np.array([0.0, 0.0, -1.0]),
    "RightFoot": np.array([-0.05, 0.12, -0.99]),
}

# (nombre_hueso, índice_padre_MP, índice_hijo_MP) — dirección padre->hijo en coords Blender
BONE_MP_EDGES: List[Tuple[str, int, int]] = [
    ("LeftArm", 11, 13),
    ("LeftForeArm", 13, 15),
    ("LeftHand", 15, 19),
    ("RightArm", 12, 14),
    ("RightForeArm", 14, 16),
    ("RightHand", 16, 20),
    ("LeftUpLeg", 23, 25),
    ("LeftLeg", 25, 27),
    ("LeftFoot", 27, 31),
    ("RightUpLeg", 24, 26),
    ("RightLeg", 26, 28),
    ("RightFoot", 28, 32),
]


def _hip_mid(j: np.ndarray) -> np.ndarray:
    return 0.5 * (j[23] + j[24])


def _shoulder_mid(j: np.ndarray) -> np.ndarray:
    return 0.5 * (j[11] + j[12])


def build_mixamo_frame(
    joints: np.ndarray,
    pelvis: np.ndarray,
    torso_R: np.ndarray,
) -> Dict[str, Any]:
    j = np.asarray(joints, dtype=np.float64).reshape(33, 3)
    jb = np.stack([scene_point_to_blender(j[i]) for i in range(33)])
    pelvis_b = scene_point_to_blender(np.asarray(pelvis, dtype=np.float64).reshape(3))
    R_b = scene_rotation_to_blender(np.asarray(torso_R, dtype=np.float64).reshape(3, 3))
    q_hips = mat3_to_quat_wxyz(R_b)

    bones: Dict[str, List[float]] = {"Hips": q_hips.tolist()}

    hip_m = _hip_mid(jb)
    sh_m = _shoulder_mid(jb)
    nose = jb[0]
    spine_dir = _normalize(sh_m - hip_m)
    neck_dir = _normalize(nose - sh_m)

    for name in ("Spine", "Spine1", "Spine2"):
        bones[name] = quat_between_vectors(REST_BONE_DIRECTIONS[name], spine_dir).tolist()
    bones["Neck"] = quat_between_vectors(REST_BONE_DIRECTIONS["Neck"], neck_dir).tolist()
    bones["Head"] = quat_between_vectors(REST_BONE_DIRECTIONS["Head"], neck_dir).tolist()

    for bone_name, ia, ib in BONE_MP_EDGES:
        d = jb[ib] - jb[ia]
        bones[bone_name] = quat_between_vectors(
            REST_BONE_DIRECTIONS[bone_name], d
        ).tolist()

    return {
        "root_translation": pelvis_b.tolist(),
        "joints_blender": [jb[i].tolist() for i in range(33)],
        "bones": bones,
    }


def skeleton_sequence_to_document(
    skeleton_sequence: List[Dict[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for fr in skeleton_sequence:
        joints = np.asarray(fr["joints"], dtype=np.float64)
        pelvis = np.asarray(fr["pelvis"], dtype=np.float64)
        torso_R = np.asarray(fr["torso_R"], dtype=np.float64)
        frames.append(build_mixamo_frame(joints, pelvis, torso_R))

    bone_names = sorted(
        set().union(*(f["bones"].keys() for f in frames)) if frames else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": float(fps),
        "frame_count": len(frames),
        "coordinate_space": "blender_z_up",
        "rotation_convention": "quaternion_wxyz_global_hint",
        "joints_field": "joints_blender_33_world_hint",
        "bone_names": bone_names,
        "frames": frames,
    }


def write_animation_json(path: str, document: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)


def export_sequence_to_json_file(
    skeleton_sequence: List[Dict[str, Any]],
    fps: float,
    out_path: str,
) -> str:
    doc = skeleton_sequence_to_document(skeleton_sequence, fps)
    write_animation_json(out_path, doc)
    return out_path
