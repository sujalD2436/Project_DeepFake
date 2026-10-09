from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import (
    DEFAULT_SAMPLED_FRAMES,
    LOWER_RISK_CEILING,
    MIN_FACE_FRAMES,
    MODEL_PATH,
    REVIEW_FLOOR,
)
from .explain import gradcam_overlay_base64
from .model import bgr_crop_to_tensor, image_transform, load_checkpoint
from .video import VideoInputError, sample_face_frames


class VideoScorer:
    def __init__(self, checkpoint_path: str | Path = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_to_idx = load_checkpoint(checkpoint_path, self.device)
        self.fake_index = self.class_to_idx["fake"]
        self.transform = image_transform()

    def score(self, video_path: str | Path, sample_count: int = DEFAULT_SAMPLED_FRAMES) -> dict:
        face_frames, quality = sample_face_frames(video_path, sample_count=sample_count)
        if len(face_frames) < MIN_FACE_FRAMES:
            return {
                "status": "retry_or_manual_review",
                "message": f"Only {len(face_frames)} usable face frames; need at least {MIN_FACE_FRAMES}.",
                "usable_frames": len(face_frames),
                "sampled_frames": len(quality),
                "frame_quality": quality,
                "clip_score": None,
            }

        batch = torch.stack([
            bgr_crop_to_tensor(face.face_crop_bgr, self.transform)
            for face in face_frames
        ]).to(self.device)
        with torch.inference_mode():
            scores = torch.softmax(self.model(batch), dim=1)[:, self.fake_index].cpu().numpy()

        clip_score = float(np.median(scores))
        explanation_index = int(np.argsort(scores)[len(scores) // 2])
        explanation = gradcam_overlay_base64(
            self.model,
            face_frames[explanation_index].face_crop_bgr,
            self.fake_index,
            self.device,
        )
        if clip_score >= REVIEW_FLOOR:
            status = "manual_review"
        elif clip_score <= LOWER_RISK_CEILING:
            status = "lower_score_demo_only"
        else:
            status = "inconclusive_review"

        return {
            "status": status,
            "message": "Model score is uncalibrated and does not verify identity.",
            "clip_score": clip_score,
            "usable_frames": len(face_frames),
            "sampled_frames": len(quality),
            "frame_scores": [
                {"timestamp_seconds": f.timestamp_seconds, "fake_like_score": float(score)}
                for f, score in zip(face_frames, scores)
            ],
            "explanation_png_base64": explanation,
            "explanation_note": "Grad-CAM shows regions influencing this model score; it is not a manipulation mask.",
            "frame_quality": quality,
        }
