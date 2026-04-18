"""
Invoca Blender en segundo plano para retarget + export GLB + render MP4.
Requiere el ejecutable de Blender (variable de entorno BLENDER_EXECUTABLE o 'blender' en PATH).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Tuple


def resolve_blender_executable() -> Optional[str]:
    exe = os.environ.get("BLENDER_EXECUTABLE")
    if exe and os.path.isfile(exe):
        return exe
    which = shutil.which("blender")
    if which:
        return which
    which_win = shutil.which("blender.exe")
    return which_win


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_base_character() -> str:
    """
    Ruta al personaje en T-Pose (FBX o GLB).
    Prioridad: MIXAMO_BASE_CHARACTER > MIXAMO_BASE_GLB > assets/character.fbx > assets/character.glb
    """
    env_char = os.environ.get("MIXAMO_BASE_CHARACTER")
    if env_char:
        return (
            env_char
            if os.path.isabs(env_char)
            else os.path.normpath(os.path.join(_PROJECT_ROOT, env_char))
        )
    env_glb = os.environ.get("MIXAMO_BASE_GLB")
    if env_glb:
        return (
            env_glb
            if os.path.isabs(env_glb)
            else os.path.normpath(os.path.join(_PROJECT_ROOT, env_glb))
        )
    for name in ("character.fbx", "character.glb"):
        p = os.path.join(_PROJECT_ROOT, "assets", name)
        if os.path.isfile(p):
            return p
    return os.path.join(_PROJECT_ROOT, "assets", "character.fbx")


def resolve_base_character_glb() -> str:
    """Alias retrocompatible; usa resolve_base_character()."""
    return resolve_base_character()


def retarget_script_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "blender_pipeline", "retarget_render.py")


def run_blender_retarget(
    json_path: str,
    input_model: str,
    output_glb: str,
    output_mp4: str,
    fps: float,
    timeout_sec: int = 3600,
) -> Tuple[str, str]:
    blender = resolve_blender_executable()
    if not blender:
        raise RuntimeError(
            "No se encontró Blender. Defina BLENDER_EXECUTABLE o añada 'blender' al PATH."
        )
    script = retarget_script_path()
    if not os.path.isfile(script):
        raise FileNotFoundError(f"No existe el script: {script}")
    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"No existe el modelo base: {input_model}")

    os.makedirs(os.path.dirname(os.path.abspath(output_glb)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)) or ".", exist_ok=True)

    cmd = [
        blender,
        "--background",
        "--python",
        script,
        "--",
        os.path.abspath(json_path),
        os.path.abspath(input_model),
        os.path.abspath(output_glb),
        os.path.abspath(output_mp4),
        str(fps),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or "sin salida"
        raise RuntimeError(f"Blender falló (código {proc.returncode}): {msg[:4000]}")
    return output_glb, output_mp4


def run_blender_pipeline_if_available(
    json_path: str,
    output_glb: str,
    output_mp4: str,
    fps: float,
) -> Optional[Tuple[str, str]]:
    """
    Si existe el GLB base y Blender en PATH / BLENDER_EXECUTABLE, ejecuta el pipeline.
    Si no, devuelve None (sin lanzar excepción).
    """
    base = resolve_base_character()
    if not os.path.isfile(base):
        return None
    if resolve_blender_executable() is None:
        return None
    return run_blender_retarget(json_path, base, output_glb, output_mp4, fps)
