# Video-to-3D Animation Service

API de backend que convierte un video de una persona en un MP4 de vista 3D off-screen: se estima la pose frame a frame, se construye un esqueleto coherente en el tiempo (con proxy de escala y suavizado), se orienta un modelo básico (cubo) al torso y se renderiza con **PyVista**.

## Prerrequisitos

- Python 3.8+ (en entornos donde el paquete `mediapipe` no incluye `solutions`, se usa la API **Tasks** y se descarga el modelo la primera vez a `models/pose_landmarker_lite.task`; hace falta red en ese primer arranque).
- Dependencias: `pip install -r requirements.txt`

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
5. **Esqueleto 3D:** centrado en caderas, normalizado por ancho de hombros en imagen (proxy de distancia a cámara), ejes de escena con Y hacia arriba, suavizado temporal (EMA) y matriz de rotación del torso para el cubo.
6. **Render:** PyVista off-screen — tubo para huesos y cubo en la pelvis.

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
