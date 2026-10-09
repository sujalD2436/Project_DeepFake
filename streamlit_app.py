from __future__ import annotations

import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fusion KYC Demo", page_icon="🛡️", layout="centered")
st.title("Identity onboarding check")
st.caption("Demo risk signal only. It does not verify identity or prove liveness.")

api_url = st.sidebar.text_input("Inference API", "http://127.0.0.1:8000")
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
