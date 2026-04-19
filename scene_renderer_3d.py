import os
import math
import numpy as np
import cv2
import pyvista as pv
from typing import List, Dict, Any, Tuple, Optional

pv.OFF_SCREEN = True

# Default framing (override via render_sequence_pyvista kwargs)
DEFAULT_CAMERA_SMOOTHING = 0.32
MIN_BONE_LENGTH = 0.024
BONE_RADIUS = 0.041
HEAD_RADIUS_FRAC = 0.42
SKELETON_TUBE_RADIUS = 0.012
SKELETON_OPACITY = 0.28


def _skeleton_lines_polydata(joints: np.ndarray, bones: List[Tuple[int, int]]) -> pv.PolyData:
    pts = np.asarray(joints, dtype=np.float64)
    lines = []
    for a, b in bones:
        lines.extend([2, int(a), int(b)])
    if not lines:
        return pv.PolyData(pts)
    poly = pv.PolyData(pts)
    poly.lines = np.array(lines, dtype=np.int64)
    return poly


def _torso_focal_point(joints: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    """Look-at point between pelvis and mid-shoulders (stable torso proxy)."""
    j = np.asarray(joints, dtype=np.float64)
    p = np.asarray(pelvis, dtype=np.float64).ravel()
    sh = 0.5 * (j[11] + j[12])
    return 0.5 * (p + sh)


def _body_up_vector(R: np.ndarray) -> np.ndarray:
    u = np.asarray(R, dtype=np.float64)[:, 1]
    n = np.linalg.norm(u)
    return u / n if n > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _forward_vector(R: np.ndarray) -> np.ndarray:
    f = np.asarray(R, dtype=np.float64)[:, 2]
    n = np.linalg.norm(f)
    return f / n if n > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _camera_distance_from_span(joints: np.ndarray, scale: float = 1.22) -> float:
    j = np.asarray(joints, dtype=np.float64)
    span = float(np.linalg.norm(j.max(axis=0) - j.min(axis=0)))
    return float(np.clip(span * scale, 2.0, 5.8))


def _hybrid_camera_position(
    focal: np.ndarray,
    torso_R: np.ndarray,
    joints: np.ndarray,
    distance: Optional[float],
    elevation_frac: float = 0.14,
) -> np.ndarray:
    """
    Mezcla el forward del torso (columna Z de torso_R) con +Z mundo para un encuadre
    tipo frente sin asumir solo un eje fijo. Yaw opcional PYVISTA_CAM_YAW_DEG.
    Peso del cuerpo: PYVISTA_HYBRID_BODY_FORWARD_WEIGHT (0-1, defecto 0.7).
    """
    fpt = np.asarray(focal, dtype=np.float64).ravel()
    fwd_body = _forward_vector(torso_R)
    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    w = float(os.environ.get("PYVISTA_HYBRID_BODY_FORWARD_WEIGHT", "0.7"))
    w = max(0.0, min(1.0, w))
    fwd = w * fwd_body + (1.0 - w) * world_z
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    yaw_deg = float(os.environ.get("PYVISTA_CAM_YAW_DEG", "0"))
    if abs(yaw_deg) > 1e-6:
        yr = math.radians(yaw_deg)
        cy, sy = math.cos(yr), math.sin(yr)
        fwd = np.array(
            [fwd[0] * cy - fwd[2] * sy, fwd[1], fwd[0] * sy + fwd[2] * cy],
            dtype=np.float64,
        )
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up_b = _body_up_vector(torso_R)
    dist = distance if distance is not None else _camera_distance_from_span(joints)
    elev = elevation_frac * dist
    return fpt - dist * fwd + elev * up_b


def _world_front_camera_position(
    focal: np.ndarray,
    joints: np.ndarray,
    distance: Optional[float],
    elevation_frac: float = 0.18,
) -> np.ndarray:
    """
    Cámara fija tipo 'frente': +Z respecto al torso, mirando al personaje (Y arriba).
    Evita giros bruscos cuando el eje forward del torso se invierte.
    """
    fpt = np.asarray(focal, dtype=np.float64).ravel()
    dist = distance if distance is not None else _camera_distance_from_span(joints)
    offset = np.array(
        [0.0, elevation_frac * dist, dist], dtype=np.float64
    )
    return fpt + offset


def _stable_camera_position(
    focal: np.ndarray,
    torso_R: np.ndarray,
    joints: np.ndarray,
    distance: Optional[float],
    elevation_frac: float = 0.14,
) -> np.ndarray:
    """Place camera in front of the torso: along -body_forward + slight body_up."""
    fpt = np.asarray(focal, dtype=np.float64).ravel()
    fwd = _forward_vector(torso_R)
    up_b = _body_up_vector(torso_R)
    dist = distance if distance is not None else _camera_distance_from_span(joints)
    elev = elevation_frac * dist
    return fpt - dist * fwd + elev * up_b


def _smooth_camera(
    raw_pos: np.ndarray,
    prev: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    if prev is None:
        return raw_pos.copy()
    return alpha * raw_pos + (1.0 - alpha) * prev


def _bone_cylinder(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    resolution: int = 10,
) -> Optional[pv.PolyData]:
    a = np.asarray(p0, dtype=np.float64).ravel()
    b = np.asarray(p1, dtype=np.float64).ravel()
    v = b - a
    L = float(np.linalg.norm(v))
    if L < MIN_BONE_LENGTH:
        return None
    direction = v / L
    center = 0.5 * (a + b)
    return pv.Cylinder(
        center=center,
        direction=direction,
        radius=radius,
        height=L,
        resolution=resolution,
    )


def _build_mannequin_mesh(
    joints: np.ndarray,
    bones: List[Tuple[int, int]],
    bone_radius: float,
) -> pv.PolyData:
    j = np.asarray(joints, dtype=np.float64)
    meshes: List[pv.PolyData] = []
    for a, b in bones:
        cyl = _bone_cylinder(j[a], j[b], bone_radius)
        if cyl is not None:
            meshes.append(cyl)
    sw = float(np.linalg.norm(j[12, :2] - j[11, :2]) + 1e-6)
    head_r = max(0.09, min(0.22, sw * HEAD_RADIUS_FRAC))
    head = pv.Sphere(radius=head_r, center=j[0], theta_resolution=14, phi_resolution=14)
    meshes.append(head)
    if not meshes:
        return pv.Sphere(radius=0.1, center=np.zeros(3))
    return pv.merge(meshes)


def render_sequence_pyvista(
    animated_sequence: List[Dict[str, Any]],
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    camera_distance: Optional[float] = None,
    camera_smoothing: float = DEFAULT_CAMERA_SMOOTHING,
    camera_elevation_frac: float = 0.14,
    bone_radius: float = BONE_RADIUS,
    show_skeleton_overlay: bool = True,
    camera_view: Optional[str] = None,
) -> str:
    if not animated_sequence:
        return "Error: Empty animated sequence provided."

    # hybrid: mezcla forward del torso con +Z mundo (recomendado). world: solo +Z. body: solo torso_R.
    view_mode = (camera_view or os.environ.get("PYVISTA_CAMERA_VIEW", "hybrid")).strip().lower()
    if view_mode not in ("body", "world", "world_front", "hybrid"):
        view_mode = "hybrid"

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.abspath(output_path)
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise IOError(f"Could not open VideoWriter for path: {out_path}")

    plotter = pv.Plotter(
        off_screen=True,
        window_size=[width, height],
        lighting="three lights",
    )
    plotter.set_background("#1a1a24")

    smoothed_cam: Optional[np.ndarray] = None

    try:
        for i, frame in enumerate(animated_sequence):
            joints = np.asarray(frame["joints"], dtype=np.float64)
            bones = frame["bones"]
            pelvis = np.asarray(frame["pelvis"], dtype=np.float64).ravel()
            R = np.asarray(frame["torso_R"], dtype=np.float64)

            plotter.clear_actors()

            mannequin = _build_mannequin_mesh(joints, bones, bone_radius)
            plotter.add_mesh(mannequin, color="#c9a87c", smooth_shading=True, opacity=1.0)

            if show_skeleton_overlay:
                skel = _skeleton_lines_polydata(joints, bones)
                if skel.lines is not None and skel.lines.size > 0:
                    thick = skel.tube(radius=SKELETON_TUBE_RADIUS, n_sides=8)
                    plotter.add_mesh(
                        thick,
                        color="#4ecdc4",
                        smooth_shading=True,
                        opacity=SKELETON_OPACITY,
                    )

            focal = _torso_focal_point(joints, pelvis)
            dist_i = camera_distance
            if view_mode == "hybrid":
                raw_cam = _hybrid_camera_position(
                    focal, R, joints, dist_i, elevation_frac=camera_elevation_frac
                )
            elif view_mode in ("world", "world_front"):
                raw_cam = _world_front_camera_position(
                    focal, joints, dist_i, elevation_frac=camera_elevation_frac
                )
            else:
                raw_cam = _stable_camera_position(
                    focal, R, joints, dist_i, elevation_frac=camera_elevation_frac
                )
            smoothed_cam = _smooth_camera(
                raw_cam, smoothed_cam, camera_smoothing
            )
            plotter.camera.position = tuple(float(x) for x in smoothed_cam)
            plotter.camera.focal_point = tuple(float(x) for x in focal)
            plotter.camera.up = (0.0, 1.0, 0.0)
            plotter.camera.clipping_range = (0.01, 100.0)

            img = plotter.screenshot(return_img=True)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"Frame {i + 1}/{len(animated_sequence)}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (240, 240, 245),
                2,
                cv2.LINE_AA,
            )
            out.write(bgr)
    finally:
        out.release()
        plotter.close()

    return out_path
