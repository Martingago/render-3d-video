import os
import numpy as np
import cv2
import pyvista as pv
from typing import List, Dict, Any, Tuple

pv.OFF_SCREEN = True


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


def _transformed_cube(pelvis: np.ndarray, R: np.ndarray) -> pv.PolyData:
    cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=0.38, y_length=0.52, z_length=0.28)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(R, dtype=np.float64)
    tf[:3, 3] = np.asarray(pelvis, dtype=np.float64).ravel()
    return cube.transform(tf, inplace=False)


def render_sequence_pyvista(
    animated_sequence: List[Dict[str, Any]],
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
) -> str:
    if not animated_sequence:
        return "Error: Empty animated sequence provided."

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

    try:
        for i, frame in enumerate(animated_sequence):
            joints = np.asarray(frame["joints"], dtype=np.float64)
            bones = frame["bones"]
            pelvis = np.asarray(frame["pelvis"], dtype=np.float64).ravel()
            R = np.asarray(frame["torso_R"], dtype=np.float64)

            plotter.clear_actors()

            skel = _skeleton_lines_polydata(joints, bones)
            if skel.lines is not None and skel.lines.size > 0:
                thick = skel.tube(radius=0.035, n_sides=12)
                plotter.add_mesh(thick, color="#4ecdc4", smooth_shading=True)

            cube_mesh = _transformed_cube(pelvis, R)
            plotter.add_mesh(cube_mesh, color="#ff6b6b", opacity=0.92, show_edges=True)

            j = joints
            center = 0.5 * (j.min(axis=0) + j.max(axis=0))
            span = float(np.linalg.norm(j.max(axis=0) - j.min(axis=0))) + 0.4
            cam_dist = max(span * 1.35, 1.8)
            plotter.camera.position = (
                float(center[0] + cam_dist * 0.85),
                float(center[1] + cam_dist * 0.45),
                float(center[2] + cam_dist * 0.75),
            )
            plotter.camera.focal_point = tuple(float(x) for x in center)
            plotter.camera.up = (0.0, 1.0, 0.0)

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
