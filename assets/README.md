# Assets 3D

Coloca aquí el modelo base en **T-Pose** (FBX o GLB), por convención:

- `character.fbx` — preferido si existe (coincide con muchos descargables de Mixamo).
- `character.glb` — alternativa si no hay FBX.

Si no existen, puedes fijar la ruta con la variable de entorno `MIXAMO_BASE_CHARACTER` (ver `.env.example`).

Rigs tipo **Mixamo / Y-Bot** encajan con el mapeo de huesos del puente JSON y del script `blender_pipeline/retarget_render.py`.

Los archivos `.glb` suelen ser pesados; puedes versionarlos aparte o ignorarlos en git (ver `.gitignore`).
