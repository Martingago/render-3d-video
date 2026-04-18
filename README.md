# Video-to-3D Animation Service

API de backend que convierte un video de una persona en un MP4 de vista 3D off-screen: se estima la pose frame a frame, se construye un esqueleto coherente en el tiempo (con proxy de escala y suavizado) y se renderiza con **PyVista** un **maniquí procedural** (cilindros por hueso + esfera en la nariz). La **cámara** sigue al torso (punto de mira entre pelvis y hombros, posición delante del cuerpo según `torso_R`, con suavizado temporal), no al centro del bounding box global, para evitar saltos de encuadre cuando se mueven brazos o piernas.

## Prerrequisitos

- Python 3.8+ (en entornos donde el paquete `mediapipe` no incluye `solutions`, se usa la API **Tasks** y se descarga el modelo la primera vez a `models/pose_landmarker_lite.task`; hace falta red en ese primer arranque).
- Dependencias: `pip install -r requirements.txt`
- **Variables locales:** copia [`.env.example`](.env.example) a `.env` en la raíz del proyecto. Ahí defines la ruta al **`blender.exe`** (no uses la carpeta del menú Inicio de Windows; debe ser el ejecutable, p. ej. `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe`) y, si quieres, `MIXAMO_BASE_CHARACTER` (por defecto se busca `assets/character.fbx` y luego `assets/character.glb`). La app carga `.env` al arrancar con `python-dotenv`.

## Ejecución

```bash
python app.py
```

Servidor por defecto en `http://127.0.0.1:5000`.

## Flujo

1. **Upload:** `POST /upload` con `multipart/form-data`, clave `file`.
2. **Validación:** extensiones `.mp4`, `.mov`, `.avi` y límite de tamaño configurado en `app.py`.
3. **Frames:** un resultado de pose por frame de video (si no hay detección en un frame, se rellena hacia adelante en el mapeo esquelético).
4. **Pose:** MediaPipe (Solutions o Tasks según el wheel instalado).
5. **Esqueleto 3D:** centrado en caderas, normalizado por ancho de hombros en imagen (proxy de distancia a cámara), ejes de escena con Y hacia arriba, suavizado temporal (EMA) y matriz `torso_R` para orientación del torso.
6. **Render:** PyVista off-screen — maniquí (malla fusionada de cilindros por conexión MediaPipe + cabeza), tubo esquelético opcional semitransparente, cámara frontal respecto al torso con EMA sobre la posición de cámara.

Un **personaje GLB riggeado con skinning** no forma parte de este pipeline: haría falta malla con pesos por hueso y retargeting; el maniquí es una aproximación volumétrica ligada directamente a las posiciones de las articulaciones.

## API

- **`POST /upload`** — Procesa el video y devuelve JSON con `final_output_video`, `fps` y `download_url`.
- **`POST /upload?download=1`** — Misma petición pero la respuesta es el archivo MP4 (descarga directa).
- **`GET /download/<nombre_archivo.mp4>`** — Descarga un fichero ya generado dentro de `outputs/` (solo nombre base seguro).

## Fidelidad 3D y video monocular

Con **una sola cámara RGB** no hay reconstrucción métrica perfecta: la profundidad y la escala absoluta están subdeterminadas. MediaPipe aporta coordenadas normalizadas en imagen y profundidad relativa respecto a la cadera, no metros del mundo real. El pipeline aplica un **proxy de escala** (tamaño aparente del torso) y una base de orientación del torso para que acercarse/alejarse y giros se reflejen de forma razonable en la escena de prueba; para fidelidad tipo producción harían falta multi-vista, sensores de profundidad o modelos dedicados 2D→3D.

## Estructura del código

| Archivo | Rol |
|--------|-----|
| `app.py` | Flask, orquestación, FPS del contenedor, descarga |
| `video_processor.py` | Lectura de frames, dimensiones, `get_video_fps` |
| `pose_estimator.py` | Pose por frame (Solutions o Tasks + modelo en `models/`) |
| `skeleton_mapper.py` | Huesos, escala, suavizado, `torso_R`, `pelvis` |
| `scene_renderer_3d.py` | PyVista → capturas → OpenCV `VideoWriter` |
| `video_renderer.py` | Delega en `scene_renderer_3d` |
| `animation_bridge.py` | Secuencia esquelética → JSON (cuaterniones wxyz + traslación raíz en espacio Blender Z-up) |
| `blender_runner.py` | Subproceso: `blender --background --python .../retarget_render.py` |
| `blender_pipeline/retarget_render.py` | Solo dentro de Blender (`bpy`): importa GLB, aplica keyframes, exporta GLB animado y MP4 (Workbench + FFmpeg) |

## Pipeline Blender (opcional)

Tras el render PyVista, el backend escribe `outputs/<nombre>_animation.json` y, **si** existen un modelo base y el ejecutable de Blender:

- Modelo por defecto: el primero que exista entre [`assets/character.fbx`](assets/README.md) y `assets/character.glb`, o rutas en **`MIXAMO_BASE_CHARACTER`** / **`MIXAMO_BASE_GLB`** (personaje en T-Pose; convención de huesos tipo Mixamo: `Hips` o `mixamorig:Hips`, etc.). El script de Blender importa **FBX** o **GLB** según la extensión.
- Ejecutable: **`BLENDER_EXECUTABLE`** (ruta absoluta recomendada en Windows) o `blender` / `blender.exe` en el `PATH`.

Salidas adicionales en `outputs/`:

- `<nombre>_blender.glb` — animación horneada en el rig.
- `<nombre>_blender.mp4` — vista previa renderizada en Blender (distinta del MP4 PyVista).

La respuesta JSON de `POST /upload` incluye `animation_json`, `blender_glb`, `blender_video` y `blender_note` (motivo si el paso se omite o falla).

El retarget es **heurístico**: cuaterniones se derivan de direcciones hueso-a-hueso respecto a vectores de reposo aproximados; no sustituye a un retargeter profesional ni a skinning manual. Para un GLB concreto puede ser necesario ajustar `REST_BONE_DIRECTIONS` en `animation_bridge.py` o los alias en `retarget_render.py`.

### Estabilidad del esqueleto y cámara PyVista

- El mapper aplica **escala robusta** (hombros o caderas), **amortiguación del eje z** de MediaPipe, **EMA más fuerte**, **límite de velocidad** por articulación y **corrección de signo** del eje forward del torso para reducir saltos y modelos “doblados”.
- Cámara PyVista por defecto **`world`** (frente estable); con **`PYVISTA_CAMERA_VIEW=body`** en `.env` se vuelve al seguimiento al `torso_R`.

### Ver Blender en vivo / logs

- **`BLENDER_GUI=1`** en `.env`: se lanza Blender **sin** `--background` (se abre la ventana). Debes **cerrar Blender** cuando termine el script para que Flask siga.
- **`BLENDER_LOG_OUTPUT=1`**: la salida estándar de Blender se mezcla con la consola del servidor (útil con `--background`).
