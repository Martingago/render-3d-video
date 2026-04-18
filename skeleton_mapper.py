import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        POSE_BONES: List[Tuple[int, int]] = list(mp.solutions.pose.POSE_CONNECTIONS)
    else:
        from mediapipe.tasks.python import vision as mp_vision

        POSE_BONES = [
            (c.start, c.end) for c in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        ]
except Exception:  # pragma: no cover - fallback topology
    POSE_BONES = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]

# MediaPipe indices (reference)
IDX_L_SHOULDER, IDX_R_SHOULDER = 11, 12
IDX_L_HIP, IDX_R_HIP = 23, 24

# Menor alpha = más suavizado (sigue menos el frame actual).
SMOOTH_ALPHA = 0.22
SCENE_AMPLITUDE = 2.2
# MediaPipe z es ruidoso; reduce artefactos "doblados" en 3D.
Z_RELATIVE_DAMP = 0.55
# Evita explosión cuando hombros se cruzan / poca separación en imagen.
MIN_TORSO_SCALE_NORM = 0.04
# Máximo desplazamiento por articulación y frame tras suavizado (coords escena).
MAX_JOINT_STEP = 0.11


def pose_dict_to_xyz(pose_data: Dict) -> np.ndarray:
    pts = np.zeros((33, 3), dtype=np.float64)
    for i in range(33):
        key = f"landmark_{i}"
        if key not in pose_data:
            continue
        lm = pose_data[key]
        pts[i, 0] = float(lm["x"])
        pts[i, 1] = float(lm["y"])
        pts[i, 2] = float(lm["z"])
    return pts


def shoulder_width_xy(pts: np.ndarray) -> float:
    d = pts[IDX_R_SHOULDER, :2] - pts[IDX_L_SHOULDER, :2]
    return float(np.linalg.norm(d))


def hip_width_xy(pts: np.ndarray) -> float:
    d = pts[IDX_R_HIP, :2] - pts[IDX_L_HIP, :2]
    return float(np.linalg.norm(d))


def robust_torso_scale_xy(pts: np.ndarray) -> float:
    """Escala de normalización estable: hombros, cadera o mínimo seguro."""
    sw = shoulder_width_xy(pts)
    hw = hip_width_xy(pts)
    return float(max(sw, hw * 0.92, MIN_TORSO_SCALE_NORM))


def hip_midpoint(pts: np.ndarray) -> np.ndarray:
    return 0.5 * (pts[IDX_L_HIP] + pts[IDX_R_HIP])


def shoulder_midpoint(pts: np.ndarray) -> np.ndarray:
    return 0.5 * (pts[IDX_L_SHOULDER] + pts[IDX_R_SHOULDER])


def mediapipe_to_scene_points(pts_centered: np.ndarray) -> np.ndarray:
    """Image x right, y down (MP) -> scene x right, y up, z forward-ish."""
    c = np.asarray(pts_centered, dtype=np.float64).copy()
    c[:, 2] *= Z_RELATIVE_DAMP
    out = np.empty_like(c)
    out[:, 0] = c[:, 0]
    out[:, 1] = -c[:, 1]
    out[:, 2] = c[:, 2]
    return out * SCENE_AMPLITUDE


def torso_rotation_matrix(pts_scene: np.ndarray) -> np.ndarray:
    """
    Columns [right, up, forward] map body axes to scene (right-handed).
    """
    hip = hip_midpoint(pts_scene)
    sh = shoulder_midpoint(pts_scene)
    up = sh - hip
    across = pts_scene[IDX_R_SHOULDER] - pts_scene[IDX_L_SHOULDER]

    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        return np.eye(3, dtype=np.float64)
    up = up / up_norm

    right = across - np.dot(across, up) * up
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, ref)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            right = np.cross(up, ref)
            rn = np.linalg.norm(right)
    right = right / rn

    forward = np.cross(right, up)
    fn = np.linalg.norm(forward)
    if fn < 1e-6:
        return np.eye(3, dtype=np.float64)
    forward = forward / fn

    return np.column_stack([right, up, forward])


def _stabilize_torso_R_forward(R: np.ndarray, R_prev: Optional[np.ndarray]) -> np.ndarray:
    """Evita saltos de 180° en el eje forward del torso entre frames."""
    R = np.asarray(R, dtype=np.float64).copy()
    if R_prev is None:
        return R
    f = R[:, 2]
    fp = R_prev[:, 2]
    if float(np.dot(f, fp)) < 0.0:
        R[:, 0] *= -1.0
        R[:, 2] *= -1.0
    return R


def _velocity_clamp_joint_sequence(seq: List[np.ndarray], max_step: float) -> List[np.ndarray]:
    if not seq:
        return []
    out: List[np.ndarray] = [seq[0].copy()]
    for i in range(1, len(seq)):
        delta = seq[i] - out[-1]
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-8))
        out.append(out[-1] + delta * scale)
    return out


def _forward_fill_poses(poses_list: List[Optional[Dict]]) -> List[Dict]:
    filled: List[Dict] = []
    last: Optional[np.ndarray] = None
    for p in poses_list:
        if p is not None:
            xyz = pose_dict_to_xyz(p)
            last = xyz
            filled.append(p)
        elif last is not None:
            filled.append(
                {
                    f"landmark_{i}": {
                        "x": float(last[i, 0]),
                        "y": float(last[i, 1]),
                        "z": float(last[i, 2]),
                        "visibility": 0.0,
                    }
                    for i in range(33)
                }
            )
        else:
            filled.append(
                {
                    f"landmark_{i}": {
                        "x": 0.5,
                        "y": 0.5,
                        "z": 0.0,
                        "visibility": 0.0,
                    }
                    for i in range(33)
                }
            )
    return filled


def _ema_smooth_sequence(scene_seq: List[np.ndarray]) -> List[np.ndarray]:
    if not scene_seq:
        return []
    prev = scene_seq[0].copy()
    out = [prev.copy()]
    alpha = SMOOTH_ALPHA
    for i in range(1, len(scene_seq)):
        prev = alpha * scene_seq[i] + (1.0 - alpha) * prev
        out.append(prev.copy())
    return out


def map_pose_to_skeleton_frame(pose_data: Dict, scale_proxy: float) -> Dict:
    pts = pose_dict_to_xyz(pose_data)
    hip = hip_midpoint(pts)
    centered = pts - hip
    denom = robust_torso_scale_xy(pts)
    normalized = centered / denom
    scene_pts = mediapipe_to_scene_points(normalized)

    R = torso_rotation_matrix(scene_pts)
    pelvis = hip_midpoint(scene_pts)

    return {
        "joints": scene_pts.astype(np.float32),
        "bones": POSE_BONES,
        "pelvis": pelvis.astype(np.float32),
        "torso_R": R.astype(np.float32),
        "scale_proxy": float(scale_proxy),
    }


def animate_skeleton_sequence(poses_list: List[Optional[Dict]]) -> List[Dict]:
    if not poses_list:
        return []

    filled = _forward_fill_poses(poses_list)

    per_frame_scale: List[float] = []
    centered_norm: List[np.ndarray] = []

    for pose_data in filled:
        pts = pose_dict_to_xyz(pose_data)
        hip = hip_midpoint(pts)
        centered = pts - hip
        denom = robust_torso_scale_xy(pts)
        sw = shoulder_width_xy(pts)
        per_frame_scale.append(sw)
        centered_norm.append(centered / denom)

    scene_raw = [mediapipe_to_scene_points(c) for c in centered_norm]
    scene_smooth = _ema_smooth_sequence(scene_raw)
    scene_smooth = _velocity_clamp_joint_sequence(scene_smooth, MAX_JOINT_STEP)

    animated: List[Dict] = []
    R_prev: Optional[np.ndarray] = None
    for i, scene_pts in enumerate(scene_smooth):
        R = torso_rotation_matrix(scene_pts)
        R = _stabilize_torso_R_forward(R, R_prev)
        R_prev = R
        pelvis = hip_midpoint(scene_pts)
        animated.append(
            {
                "joints": scene_pts.astype(np.float32),
                "bones": POSE_BONES,
                "pelvis": pelvis.astype(np.float32),
                "torso_R": R.astype(np.float32),
                "scale_proxy": float(per_frame_scale[i]),
            }
        )
    return animated


# Kept for compatibility with imports elsewhere
RIG_JOINTS = [
    "root",
    "head",
    "neck",
    "r_shoulder",
    "l_shoulder",
    "spine",
    "r_elbow",
    "l_elbow",
    "r_wrist",
    "l_wrist",
]


def map_pose_to_skeleton(pose_data: Dict) -> Dict:
    sw = shoulder_width_xy(pose_dict_to_xyz(pose_data))
    return map_pose_to_skeleton_frame(pose_data, scale_proxy=sw)
