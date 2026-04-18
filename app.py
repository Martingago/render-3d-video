from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from pose_estimator import analyze_video_poses
from skeleton_mapper import animate_skeleton_sequence
from video_renderer import render_sequence_to_video
from video_processor import get_video_fps

MAX_FILE_SIZE_MB = 20
ALLOWED_VIDEO_EXT = [".mp4", ".mov", ".avi"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return (
            jsonify(
                {"error": "No file part in the request. Must send 'file' key."}
            ),
            400,
        )

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return (
            jsonify(
                {
                    "error": f"Unsupported media format. Only {', '.join(ALLOWED_VIDEO_EXT)} are allowed."
                }
            ),
            415,
        )

    filepath = os.path.join("uploads", original_filename)
    file.save(filepath)

    try:
        print("--- Step 1/4: Estimating poses (one row per frame) ---")
        all_poses = analyze_video_poses(filepath)
        if not all_poses or not any(p for p in all_poses if p):
            return (
                jsonify({"error": "Could not extract any poses from the video."}),
                500,
            )

        print("--- Step 2/4: Mapping to 3D skeleton sequence ---")
        skeleton_sequence = animate_skeleton_sequence(all_poses)

        base_name = os.path.splitext(original_filename)[0]
        output_video_path = f"outputs/{base_name}_animated.mp4"
        os.makedirs("outputs", exist_ok=True)

        fps = get_video_fps(filepath)
        print(f"--- Step 3/4: Rendering 3D video at {fps:.2f} fps ---")
        final_video_path = render_sequence_to_video(
            animated_sequence=skeleton_sequence,
            output_path=output_video_path,
            width=1280,
            height=720,
            fps=fps,
        )

        payload = {
            "message": "Pipeline completed successfully.",
            "uploaded_file": original_filename,
            "final_output_video": final_video_path,
            "fps": fps,
            "download_url": f"/download/{os.path.basename(final_video_path)}",
        }
        if request.args.get("download") == "1":
            return send_file(
                final_video_path,
                as_attachment=True,
                download_name=os.path.basename(final_video_path),
            )
        return jsonify(payload), 201

    except IOError as e:
        return jsonify({"error": f"Video Processing Error: {str(e)}"}), 400
    except Exception as e:
        print(f"Critical error during pipeline run: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_output(filename):
    safe = secure_filename(os.path.basename(filename))
    if not safe:
        return jsonify({"error": "Invalid filename."}), 400
    path = os.path.join("outputs", safe)
    if not os.path.isfile(path):
        return jsonify({"error": "File not found."}), 404
    return send_file(path, as_attachment=True, download_name=safe)


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True)
