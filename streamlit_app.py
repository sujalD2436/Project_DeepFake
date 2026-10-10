from __future__ import annotations

import base64
import threading
from io import BytesIO
from pathlib import Path

import av
import cv2
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from src.kyc_demo.config import DEFAULT_BASE_URL, MODEL_PATH
from src.kyc_demo.model import bgr_crop_to_tensor
from src.kyc_demo.qr import extract_token_from_payload, scan_qr_from_bytes
from src.kyc_demo.scoring import VideoScorer
from src.kyc_demo.sessions import SessionManager

st.set_page_config(
    page_title="Fusion KYC & QR Verification",
    page_icon="🛡️",
    layout="centered",
)
st.title("Identity onboarding & verification")
st.caption("Deepfake risk signal & ephemeral QR code verification demo.")

session_mgr = SessionManager()


@st.cache_resource
def get_scorer(checkpoint_path: str) -> VideoScorer:
    return VideoScorer(checkpoint_path)


def make_live_callback(scorer: VideoScorer):
    """Create a WebRTC callback that scores a face every few camera frames."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV face detector could not be loaded: {cascade_path}")

    state = {"frame": 0, "score": None, "box": None}
    inference_lock = threading.Lock()

    def process(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        state["frame"] += 1

        if state["frame"] % 6 == 0:
            height, width = image.shape[:2]
            scale = min(1.0, 640.0 / max(width, height))
            small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            boxes = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            if len(boxes) == 0:
                state["score"] = None
                state["box"] = None
            else:
                x, y, face_width, face_height = max(boxes, key=lambda box: box[2] * box[3])
                if scale < 1:
                    x, y = int(x / scale), int(y / scale)
                    face_width, face_height = int(face_width / scale), int(face_height / scale)
                pad_x, pad_y = int(face_width * 0.2), int(face_height * 0.2)
                left, top = max(0, x - pad_x), max(0, y - pad_y)
                right = min(width, x + face_width + pad_x)
                bottom = min(height, y + face_height + pad_y)
                crop = image[top:bottom, left:right]
                if crop.size:
                    tensor = bgr_crop_to_tensor(crop, scorer.transform).unsqueeze(0).to(scorer.device)
                    with inference_lock, torch.inference_mode():
                        probability = torch.softmax(scorer.model(tensor), dim=1)[0, scorer.fake_index]
                    state["score"] = float(probability.item())
                    state["box"] = (left, top, right, bottom)

        output = image.copy()
        box = state["box"]
        score = state["score"]
        if box is not None:
            cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (0, 220, 80), 2)
        label = "No face detected" if score is None else f"Demo fake-like score: {score:.1%}"
        cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 390), 44), (20, 20, 20), -1)
        cv2.putText(output, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    return process


live_tab, upload_tab, verify_tab = st.tabs([
    "Live webcam",
    "Upload clip & onboard",
    "Reviewer QR verification",
])

# ----------------- TAB 1: LIVE WEBCAM -----------------
with live_tab:
    st.write("Start the camera and allow browser camera access. The score updates during the feed.")
    checkpoint = st.text_input("Model checkpoint", str(MODEL_PATH), key="live_checkpoint")
    try:
        live_scorer = get_scorer(checkpoint)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        st.error(f"Could not load the model: {exc}")
    else:
        st.caption(f"Inference device: {live_scorer.device}")
        webrtc_streamer(
            key="kyc-live-camera",
            video_frame_callback=make_live_callback(live_scorer),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    st.caption("Live score is an uncalibrated demo signal; it does not verify identity.")

# ----------------- TAB 2: UPLOAD & ONBOARD -----------------
with upload_tab:
    st.write("Upload a short onboarding clip to score deepfake risk and generate a verification QR code.")
    api_url = st.text_input("Inference API", DEFAULT_BASE_URL, key="api_url")
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded is not None:
        st.video(uploaded)
        consent = st.checkbox("I have permission to use this clip for this demo.")
        if st.button("Run onboarding check", type="primary", disabled=not consent):
            with st.spinner("Checking quality, scoring face crops, and generating verification session..."):
                try:
                    response = requests.post(
                        f"{api_url.rstrip('/')}/score",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "video/mp4")},
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                except requests.RequestException as exc:
                    st.error(f"Could not reach the inference service: {exc}")
                else:
                    st.subheader("Mock onboarding result")
                    if result.get("clip_score") is None:
                        st.warning(result.get("message", "Video needs another capture."))
                    else:
                        score_col, status_col = st.columns(2)
                        with score_col:
                            st.metric("Fake-like model score", f"{result['clip_score']:.1%}")
                        with status_col:
                            st.metric("Risk routing band", result["status"].replace("_", " ").title())

                        st.caption(result.get("message", ""))

                        # Grad-CAM explanation
                        encoded_map = result.get("explanation_png_base64")
                        if encoded_map:
                            st.image(
                                Image.open(BytesIO(base64.b64decode(encoded_map))),
                                caption=result.get("explanation_note", "Model attribution map"),
                                use_container_width=True,
                            )

                        # QR Verification Section
                        st.divider()
                        st.subheader("🔒 Reviewer verification QR code")
                        st.caption(
                            "Security rule: Contains an unpredictable, expiring token only. "
                            "No private identity data is stored in the QR code."
                        )

                        qr_col, info_col = st.columns([1, 2])
                        qr_base64 = result.get("qr_code_base64")
                        session_id = result.get("session_id", "N/A")
                        verification_url = result.get("verification_url", "")
                        token = result.get("verification_token", "")

                        if qr_base64:
                            qr_bytes = base64.b64decode(qr_base64)
                            with qr_col:
                                st.image(qr_bytes, caption=f"Session: {session_id}", width=180)
                                st.download_button(
                                    "Download QR Code",
                                    data=qr_bytes,
                                    file_name=f"{session_id}_qr.png",
                                    mime="image/png",
                                )

                            with info_col:
                                st.markdown(f"**Session ID:** `{session_id}`")
                                st.markdown(f"**Verification Token:** `{token[:8]}...{token[-6:]}`")
                                st.markdown(f"**Expires At:** `{result.get('expires_at', 'N/A')}`")
                                if verification_url:
                                    st.markdown(
                                        f"📱 **Mobile Review URL:** [Open Reviewer Report]({verification_url})"
                                    )
                                st.info("Scan with a smartphone camera or the 'Reviewer QR verification' tab.")

                        with st.expander("Frame-level quality details"):
                            st.json(result.get("frame_quality", []))

# ----------------- TAB 3: REVIEWER QR VERIFICATION -----------------
with verify_tab:
    st.subheader("Auditor / reviewer verification")
    st.write("Scan a candidate's QR code using OpenCV or enter a token to verify session authenticity.")

    scan_mode = st.radio(
        "Verification method",
        ["📷 Webcam capture (OpenCV)", "📁 Upload QR code image", "⌨️ Enter token manually"],
        horizontal=True,
    )

    detected_token = ""

    if scan_mode == "📷 Webcam capture (OpenCV)":
        st.write("Point your webcam at the QR code displayed on screen or printed badge:")
        camera_photo = st.camera_input("Capture QR code frame")
        if camera_photo is not None:
            photo_bytes = camera_photo.getvalue()
            with st.spinner("Decoding QR code using OpenCV..."):
                token, raw_payload = scan_qr_from_bytes(photo_bytes)
                if token:
                    st.success(f"QR code detected via OpenCV! Token: `{token[:10]}...`")
                    detected_token = token
                else:
                    st.warning("No readable QR code found in the captured image. Hold it closer or adjust lighting.")

    elif scan_mode == "📁 Upload QR code image":
        qr_file = st.file_uploader("Upload QR code PNG/JPEG", type=["png", "jpg", "jpeg"])
        if qr_file is not None:
            with st.spinner("Decoding QR code using OpenCV..."):
                token, raw_payload = scan_qr_from_bytes(qr_file.getvalue())
                if token:
                    st.success(f"QR code detected! Token: `{token[:10]}...`")
                    detected_token = token
                else:
                    st.error("Could not decode a QR code from this image.")

    else:
        manual_input = st.text_input("Enter verification token or paste full report URL:")
        if manual_input.strip():
            detected_token = extract_token_from_payload(manual_input)

    # Perform Verification & Display Report
    if detected_token:
        st.divider()
        st.subheader("📋 Session verification report")

        # Verify token in backend SQLite
        is_valid, msg, session = session_mgr.verify_token(detected_token)

        if not session:
            st.error("Verification token does not match any known session.")
        else:
            if not is_valid:
                st.error(f"Verification Failed: {msg}")
            else:
                st.success("Session verified successfully!")

            # Metric cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Session ID", session.get("session_id", "N/A"))
            with col2:
                clip_score = session.get("clip_score")
                st.metric(
                    "Fake-like score",
                    f"{clip_score:.1%}" if clip_score is not None else "Pending",
                )
            with col3:
                st.metric(
                    "Status",
                    session.get("status", "PENDING").replace("_", " ").title(),
                )

            # Details
            st.write("---")
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Created At (UTC):** `{session.get('created_at', 'N/A')}`")
                st.markdown(f"**Verified At (UTC):** `{session.get('verified_at', 'N/A')}`")
                st.markdown(f"**Expires At (UTC):** `{session.get('expires_at', 'N/A')}`")
            with d2:
                st.markdown(f"**Usable Face Frames:** `{session.get('usable_frames', 0)}`")
                st.markdown(f"**Token Expired:** `{'Yes' if session.get('is_expired') else 'No'}`")
                st.markdown(f"**Auditor Verdict:** `{session.get('reviewer_verdict') or 'Pending'}`")

            # Explanation heatmap
            if session.get("explanation_png_base64"):
                st.image(
                    Image.open(BytesIO(base64.b64decode(session["explanation_png_base64"]))),
                    caption=session.get("explanation_note", "Model attribution heatmap"),
                    width=320,
                )

            # Reviewer Action Box
            st.write("---")
            st.write("#### Record auditor verdict")
            with st.form("reviewer_verdict_form"):
                verdict = st.selectbox(
                    "Auditor Decision",
                    ["APPROVED", "FLAGGED FOR FRAUD AUDIT", "REJECTED", "NEEDS RE-CAPTURE"],
                )
                notes = st.text_area("Audit Notes", placeholder="e.g. Identity verified against applicant document.")
                submit_verdict = st.form_submit_button("Record Decision in SQLite", type="primary")

                if submit_verdict:
                    updated = session_mgr.record_reviewer_verdict(
                        session["session_id"], verdict, notes
                    )
                    st.success(f"Verdict '{verdict}' successfully recorded for session {session['session_id']}!")
                    st.rerun()