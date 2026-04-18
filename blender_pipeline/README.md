# Pipeline Blender (headless)

## Requisitos

- Blender 3.6+ o 4.x instalado.
- Variable de entorno `BLENDER_EXECUTABLE` con la ruta absoluta a **`blender.exe`** (no la carpeta del menú Inicio; en el Explorador suele estar bajo `C:\Program Files\Blender Foundation\Blender X.Y\blender.exe`), o `blender` / `blender.exe` en el `PATH`.
- Opcional: archivo `.env` en la raíz del proyecto (copia de `.env.example`); Flask lo carga al arrancar.
- **`BLENDER_GUI=1`**: abre la ventana de Blender (sin `--background`) para ver el import, keyframes y render; hay que cerrar Blender al finalizar.
- **`BLENDER_LOG_OUTPUT=1`**: en modo consola, muestra la salida de Blender en la terminal de Flask.

## Modelo base

Coloca un personaje en T-Pose (por ejemplo Mixamo Y-Bot) en:

- `assets/character.fbx` o `assets/character.glb` (se elige el primero que exista), o
- rutas en `MIXAMO_BASE_CHARACTER` o `MIXAMO_BASE_GLB`.

Los nombres de huesos deben coincidir con Mixamo (`Hips`, `mixamorig:Hips`, etc.); el script `retarget_render.py` prueba alias comunes.

## Comando manual

Desde la raíz del proyecto:

```text
"%BLENDER_EXECUTABLE%" --background --python blender_pipeline\retarget_render.py -- ^
  outputs\mi_anim.json assets\character.glb outputs\mi_salida.glb outputs\mi_salida_blender.mp4 30
```

(Linux/macOS: sustituir `^` por `\`.)

En la API `/upload`, si Blender genera `*_blender.mp4`, ese archivo pasa a ser **`final_output_video`** y la descarga por defecto; el maniquí PyVista queda en **`pyvista_preview_video`** (`*_animated.mp4`).
