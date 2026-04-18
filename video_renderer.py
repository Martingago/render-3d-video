from typing import List, Dict, Any

from scene_renderer_3d import render_sequence_pyvista


def render_sequence_to_video(
    animated_sequence: List[Dict[str, Any]],
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
) -> str:
    """
    Renders the animated skeleton sequence with a basic 3D primitive (cube) using PyVista off-screen.
    """
    return render_sequence_pyvista(
        animated_sequence=animated_sequence,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
    )


if __name__ == "__main__":
    print("Video Renderer delegates to scene_renderer_3d (PyVista).")
