from typing import List, Dict, Any, Optional

from scene_renderer_3d import render_sequence_pyvista


def render_sequence_to_video(
    animated_sequence: List[Dict[str, Any]],
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    camera_distance: Optional[float] = None,
    camera_smoothing: Optional[float] = None,
    bone_radius: Optional[float] = None,
    show_skeleton_overlay: Optional[bool] = None,
) -> str:
    """
    Renders the animated sequence with a procedural mannequin (bone cylinders + head)
    and a stable camera aligned to torso_R; optional semi-transparent skeleton overlay.
    Extra arguments are forwarded only when not None.
    """
    kwargs: Dict[str, Any] = {}
    if camera_distance is not None:
        kwargs["camera_distance"] = camera_distance
    if camera_smoothing is not None:
        kwargs["camera_smoothing"] = camera_smoothing
    if bone_radius is not None:
        kwargs["bone_radius"] = bone_radius
    if show_skeleton_overlay is not None:
        kwargs["show_skeleton_overlay"] = show_skeleton_overlay

    return render_sequence_pyvista(
        animated_sequence=animated_sequence,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        **kwargs,
    )


if __name__ == "__main__":
    print("Video Renderer delegates to scene_renderer_3d (PyVista).")
