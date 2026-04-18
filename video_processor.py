import cv2
import numpy as np
from typing import Generator, Tuple

def get_frames_generator(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Reads a video file frame by frame and yields the NumPy array for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()

def get_video_fps(video_path: str, fallback: float = 30.0) -> float:
    """
    Reads the container-reported FPS; falls back if the value is invalid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1.0 or fps > 240.0 or fps != fps:
        return fallback
    return fps


def extract_frame_dimensions(video_path: str) -> Tuple[int, int]:
    """
    Returns the width and height of the frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file for dimension check: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# Placeholder for more advanced processing later
SAMPLE_RATE = 1 # Frames to process per second for resource management

if __name__ == '__main__':
    # Simple test case (assuming a dummy video exists for testing)
    try:
        # This block is just for local testing proof of concept, not for the API execution path
        # Create a dummy video path for testing purposes if no real file is provided
        dummy_video_path = "dummy_input_video.mp4"
        # In a real scenario, you'd check for an uploaded file.
        # For now, we just demonstrate the structure.
        print(f"Structure check complete. Next steps will use {__file__} functions.")
    except Exception as e:
        print(f"Test failed: {e}")