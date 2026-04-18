import os
import urllib.request
import cv2
import numpy as np
from typing import List, Dict, Optional

from video_processor import get_frames_generator, get_video_fps

NUM_POSE_LANDMARKS = 33

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _ensure_task_model() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(root, "models")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "pose_landmarker_lite.task")
    if os.path.isfile(path) and os.path.getsize(path) > 100_000:
        return path
    urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _landmarks_dict_from_list(landmarks) -> Dict:
    detected_pose = {}
    for i in range(min(NUM_POSE_LANDMARKS, len(landmarks))):
        lm = landmarks[i]
        vis = getattr(lm, "visibility", None)
        detected_pose[f"landmark_{i}"] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(vis) if vis is not None else 1.0,
        }
    return detected_pose


def _using_legacy_solutions() -> bool:
    import mediapipe as mp

    return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")


def _process_frame_legacy(pose, frame: np.ndarray) -> Optional[Dict]:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    return _landmarks_dict_from_list(landmarks)


def _analyze_legacy(video_path: str) -> List[Optional[Dict]]:
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    all_poses: List[Optional[Dict]] = []
    try:
        for frame in get_frames_generator(video_path):
            all_poses.append(_process_frame_legacy(pose, frame))
    finally:
        pose.close()
    return all_poses


def _analyze_tasks(video_path: str) -> List[Optional[Dict]]:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options as bo
    from mediapipe.tasks.python.vision.core import image as mp_image

    model_path = _ensure_task_model()
    options = vision.PoseLandmarkerOptions(
        base_options=bo.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    all_poses: List[Optional[Dict]] = []
    frame_index = 0
    fps = max(get_video_fps(video_path), 1.0)
    try:
        for frame in get_frames_generator(video_path):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_index * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_img, ts_ms)
            frame_index += 1
            if result.pose_landmarks:
                lm_list = result.pose_landmarks[0]
                all_poses.append(_landmarks_dict_from_list(lm_list))
            else:
                all_poses.append(None)
    finally:
        landmarker.close()
    return all_poses


def analyze_video_poses(video_path: str) -> List[Optional[Dict]]:
    """
    One entry per video frame: pose dict with 33 landmarks, or None if not detected.
    Uses legacy MediaPipe Solutions when available; otherwise Tasks (PoseLandmarker).
    """
    if _using_legacy_solutions():
        all_poses = _analyze_legacy(video_path)
    else:
        all_poses = _analyze_tasks(video_path)

    print(f"Processed {len(all_poses)} frames from {video_path}.")
    return all_poses


if __name__ == "__main__":
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        print("--- Running Pose Estimation Test ---")
        poses_list = analyze_video_poses(test_video)
        detected = sum(1 for p in poses_list if p is not None)
        print(f"Frames: {len(poses_list)}, with pose: {detected}.")
    else:
        print(
            "\nNOTE: To test this module locally, place a video named 'test_video.mp4' in the root directory."
        )
