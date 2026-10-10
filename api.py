from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.kyc_demo.config import MODEL_PATH
from src.kyc_demo.qr import (
    create_verification_qr_payload,
    extract_token_from_payload,
    generate_qr_base64,
    scan_qr_from_bytes,
)
from src.kyc_demo.report_view import render_verification_report_html
from src.kyc_demo.scoring import VideoScorer
from src.kyc_demo.sessions import SessionManager
from src.kyc_demo.video import VideoInputError

app = FastAPI(title="Mock KYC Deepfake Risk & Verification API", version="0.2.0")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_scorer: VideoScorer | None = None
session_mgr = SessionManager()


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


@app.post("/sessions/create")
def create_session(request: Request, session_id: str | None = None):
    """Generate a unique session ID and an unpredictable expiring verification QR code."""
    session = session_mgr.create_session(session_id=session_id)
    base_url = str(request.base_url).rstrip("/")
    payload = create_verification_qr_payload(session["verification_token"], base_url=base_url)
    qr_b64 = generate_qr_base64(payload)

    return {
        "session_id": session["session_id"],
        "verification_token": session["verification_token"],
        "verification_url": payload,
        "qr_code_base64": qr_b64,
        "created_at": session["created_at"],
        "expires_at": session["expires_at"],
        "status": session["status"],
    }


@app.post("/score")
async def score_video(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
):
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
        result = scorer().score(temp_path)
    except VideoInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    # Manage or create session record in SQLite
    session = session_mgr.get_session(session_id) if session_id else None
    if not session:
        session = session_mgr.create_session(session_id=session_id)

    updated_session = session_mgr.attach_score(session["session_id"], result)

    # Generate unpredictable expiring verification QR code for reviewers
    base_url = str(request.base_url).rstrip("/")
    verification_url = create_verification_qr_payload(
        session["verification_token"], base_url=base_url
    )
    qr_base64 = generate_qr_base64(verification_url)

    result["session_id"] = updated_session["session_id"]
    result["verification_token"] = updated_session["verification_token"]
    result["verification_url"] = verification_url
    result["qr_code_base64"] = qr_base64
    result["expires_at"] = updated_session["expires_at"]
    return result


@app.get("/verify/{token}")
def verify_token(token: str):
    """API endpoint to verify session token and return verification state."""
    clean_token = extract_token_from_payload(token)
    is_valid, message, session = session_mgr.verify_token(clean_token)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found for provided token.")
    return {
        "valid": is_valid,
        "message": message,
        "session": session,
    }


@app.get("/report/{token}", response_class=HTMLResponse)
def reviewer_report(token: str):
    """Render a mobile-friendly reviewer verification report upon scanning the QR code."""
    clean_token = extract_token_from_payload(token)
    is_valid, message, session = session_mgr.verify_token(clean_token)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid or expired verification session.")
    return HTMLResponse(content=render_verification_report_html(session, is_valid, message))


@app.post("/scan-qr")
async def scan_qr_image(file: UploadFile = File(...)):
    """Upload an image to scan for a QR code using OpenCV and verify the token."""
    content = await file.read()
    token, raw_payload = scan_qr_from_bytes(content)
    if not token:
        raise HTTPException(
            status_code=400, detail="No readable QR code found in the uploaded image."
        )

    is_valid, message, session = session_mgr.verify_token(token)
    if not session:
        raise HTTPException(status_code=404, detail="Verification token is not recognized.")

    return {
        "valid": is_valid,
        "message": message,
        "scanned_payload": raw_payload,
        "session": session,
    }


@app.post("/sessions/{session_id}/verdict")
def set_reviewer_verdict(
    session_id: str,
    verdict: str = Form(...),
    notes: str = Form(""),
):
    """Allow an auditor/reviewer to record a manual decision in SQLite."""
    updated = session_mgr.record_reviewer_verdict(session_id, verdict, notes)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "ok", "session": updated}
