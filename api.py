from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.kyc_demo.config import MODEL_PATH
from src.kyc_demo.scoring import VideoScorer
from src.kyc_demo.video import VideoInputError

app = FastAPI(title="Mock KYC Deepfake Risk API", version="0.1.0")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_scorer: VideoScorer | None = None


def scorer() -> VideoScorer:
    global _scorer
    if _scorer is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="Model checkpoint is not trained yet.")
        _scorer = VideoScorer(MODEL_PATH)
    return _scorer


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.is_file()}


@app.post("/score")
async def score_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "input.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=415, detail="Use MP4, MOV, AVI, or MKV video.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds the 50 MB upload limit.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        return scorer().score(temp_path)
    except VideoInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

