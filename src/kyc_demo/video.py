from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import MAX_FRAME_PIXELS, MAX_SAMPLED_FRAMES, MAX_VIDEO_SECONDS


class VideoInputError(ValueError):
    """Invalid, unsupported, or unusable video input."""


@dataclass
class FaceFrame:
    timestamp_seconds: float
    frame_bgr: object
    face_crop_bgr: object
    face_area_ratio: float
    blur_score: float
    brightness: float


def _face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV face detector could not be loaded: {cascade_path}")
    return detector


def sample_face_frames(
    video_path: str | Path,
    *,
    sample_count: int = 8,
    max_duration_seconds: float = MAX_VIDEO_SECONDS,
    max_frame_pixels: int = MAX_FRAME_PIXELS,
    max_sampled_frames: int = MAX_SAMPLED_FRAMES,
    min_face_area_ratio: float = 0.025,
    min_blur_score: float = 35.0,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
) -> tuple[list[FaceFrame], list[dict]]:
    """Sample a clip and return usable face crops plus per-frame quality details."""
    path = Path(video_path)
    if not path.is_file():
        raise VideoInputError("Video file does not exist.")
    if not 1 <= sample_count <= max_sampled_frames:
        raise VideoInputError(f"sample_count must be between 1 and {max_sampled_frames}.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoInputError("Video could not be opened by OpenCV.")

    usable: list[FaceFrame] = []
    details: list[dict] = []
    detector = _face_detector()
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise VideoInputError("Video metadata is missing or invalid.")

        duration = frame_count / fps
        if duration > max_duration_seconds:
            raise VideoInputError(f"Video exceeds the {max_duration_seconds:g}-second limit.")

        for index in range(sample_count):
            timestamp = duration * (index + 1) / (sample_count + 1)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                details.append({"timestamp_seconds": timestamp, "status": "decode_failed"})
                continue

            height, width = frame.shape[:2]
            if height * width > max_frame_pixels:
                scale = (max_frame_pixels / (height * width)) ** 0.5
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                height, width = frame.shape[:2]

            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            boxes = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            if len(boxes) == 0:
                details.append({"timestamp_seconds": timestamp, "status": "no_face"})
                continue

            x, y, face_width, face_height = max(boxes, key=lambda b: b[2] * b[3])
            area_ratio = (face_width * face_height) / (width * height)
            pad_x, pad_y = int(face_width * 0.2), int(face_height * 0.2)
            left, top = max(0, x - pad_x), max(0, y - pad_y)
            right = min(width, x + face_width + pad_x)
            bottom = min(height, y + face_height + pad_y)
            crop = frame[top:bottom, left:right]

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
            brightness = float(crop_gray.mean())
            reasons = []
            if area_ratio < min_face_area_ratio:
                reasons.append("face_too_small")
            if blur_score < min_blur_score:
                reasons.append("blurry")
            if brightness < min_brightness or brightness > max_brightness:
                reasons.append("poor_lighting")

            details.append({
                "timestamp_seconds": timestamp,
                "status": "usable" if not reasons else "quality_review",
                "reasons": reasons,
                "face_area_ratio": area_ratio,
                "blur_score": blur_score,
                "brightness": brightness,
            })
            if not reasons:
                usable.append(FaceFrame(
                    timestamp_seconds=timestamp,
                    frame_bgr=frame,
                    face_crop_bgr=crop,
                    face_area_ratio=area_ratio,
                    blur_score=blur_score,
                    brightness=brightness,
                ))

        if not details:
            raise VideoInputError("No frames could be decoded from the video.")
        return usable, details
    finally:
        capture.release()

