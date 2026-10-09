from __future__ import annotations

import base64
import threading
from io import BytesIO
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
import requests
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from src.kyc_demo.config import MODEL_PATH
from src.kyc_demo.model import bgr_crop_to_tensor
from src.kyc_demo.scoring import VideoScorer


st.set_page_config(page_title="Fusion KYC Demo", page_icon="🛡️", layout="centered")
st.title("Identity onboarding check")
st.caption("Demo risk signal only. It does not verify identity or prove liveness.")


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

        # Run CPU face detection and CUDA classifier inference about 5 times/sec
        # on a typical 30 FPS webcam; annotate the intervening frames cheaply.
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
                    # A lock prevents simultaneous camera callbacks from entering the model.
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


live_tab, upload_tab = st.tabs(["Live webcam", "Upload a clip"])

with live_tab:
    st.write("Start the camera and allow browser camera access. The score updates during the feed.")
    checkpoint = st.text_input("Model checkpoint", str(MODEL_PATH))
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
    st.caption("Face detection runs on the CPU; the classifier uses CUDA when available. This live score is a demo signal, not a liveness verdict.")

with upload_tab:
    st.write("Or upload a short video for the backend to score.")
    api_url = st.text_input("Inference API", "http://127.0.0.1:8000", key="api_url")
    uploaded = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded is not None:
        st.video(uploaded)
        consent = st.checkbox("I have permission to use this clip for this demo.")
        if st.button("Run check", type="primary", disabled=not consent):
            with st.spinner("Checking video quality and scoring sampled face frames..."):
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
                        st.metric("Fake-like model score", f"{result['clip_score']:.1%}")
                        st.info(result["status"].replace("_", " ").title())
                        st.caption(result.get("message", ""))
                        encoded_map = result.get("explanation_png_base64")
                        if encoded_map:
                            st.image(
                                Image.open(BytesIO(base64.b64decode(encoded_map))),
                                caption=result.get("explanation_note", "Model attribution map"),
                            )
                        with st.expander("Frame-level quality details"):
                            st.json(result.get("frame_quality", []))