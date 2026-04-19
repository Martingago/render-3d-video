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


def project_abspath(path: str) -> str:
    """Ruta absoluta estable respecto a la raíz del proyecto (no depende del CWD)."""
    p = os.path.expanduser(path.strip())
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p))


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


def _blender_use_gui() -> bool:
    v = os.environ.get("BLENDER_GUI", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _blender_log_to_console() -> bool:
    v = os.environ.get("BLENDER_LOG_OUTPUT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


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

    json_path = project_abspath(json_path)
    output_glb = project_abspath(output_glb)
    output_mp4 = project_abspath(output_mp4)

    os.makedirs(os.path.dirname(output_glb) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_mp4) or ".", exist_ok=True)

    cmd = [blender]
    if not _blender_use_gui():
        cmd.append("--background")
    cmd.extend(
        [
            "--python",
            script,
            "--",
            json_path,
            os.path.abspath(input_model),
            output_glb,
            output_mp4,
            str(fps),
        ]
    )

    use_gui = _blender_use_gui()
    inherit_console = _blender_log_to_console() or use_gui
    if inherit_console:
        run_kw = {"stdout": None, "stderr": None}
    else:
        run_kw = {"capture_output": True, "text": True}

    effective_timeout = None if use_gui else timeout_sec

    proc = subprocess.run(cmd, timeout=effective_timeout, **run_kw)
    if proc.returncode != 0:
        msg = "sin salida"
        if not inherit_console and proc.stderr is not None:
            msg = proc.stderr or proc.stdout or msg
        raise RuntimeError(f"Blender falló (código {proc.returncode}): {msg[:4000]}")
    if not os.path.isfile(output_mp4):
        outd = os.path.dirname(output_mp4)
        mp4_in_dir = (
            [f for f in os.listdir(outd) if f.lower().endswith(".mp4")]
            if os.path.isdir(outd)
            else []
        )
        raise RuntimeError(
            f"Blender terminó (código 0) pero no existe el MP4 esperado: {output_mp4}. "
            f"MP4 en la misma carpeta: {mp4_in_dir}"
        )
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
