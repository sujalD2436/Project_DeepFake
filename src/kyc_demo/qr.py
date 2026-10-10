from __future__ import annotations

import base64
import io
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import qrcode
from PIL import Image

from .config import DEFAULT_BASE_URL


def generate_qr_image(payload: str, box_size: int = 8, border: int = 2) -> Image.Image:
    """Generate a high-contrast PIL QR code image."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="#0F172A", back_color="#FFFFFF").convert("RGB")


def generate_qr_base64(payload: str, box_size: int = 8, border: int = 2) -> str:
    """Generate a base64-encoded PNG string of the QR code."""
    img = generate_qr_image(payload, box_size=box_size, border=border)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def create_verification_qr_payload(
    token: str, base_url: str = DEFAULT_BASE_URL
) -> str:
    """
    Format payload for reviewer verification.
    Security: Only contains the unpredictable expiring token via verification URL.
    No PII or private identity information is encoded into the QR code.
    """
    clean_base = base_url.rstrip("/")
    return f"{clean_base}/report/{token}"


def extract_token_from_payload(payload: str) -> str:
    """
    Extract the clean verification token from either a raw token or a verification URL.
    Examples:
      - 'http://127.0.0.1:8000/report/abcd123' -> 'abcd123'
      - 'http://127.0.0.1:8000/verify?token=abcd123' -> 'abcd123'
      - 'abcd123' -> 'abcd123'
    """
    raw = payload.strip()
    if not raw:
        return ""

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        # Check query params: ?token=...
        query = parse_qs(parsed.query)
        if "token" in query and query["token"]:
            return query["token"][0]

        # Check path: /report/{token} or /verify/{token}
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            return parts[-1]

    # Otherwise return raw trimmed token
    return raw


def scan_qr_from_bgr(bgr_image: np.ndarray) -> tuple[str, bool, Any]:
    """
    Scan a QR code from a BGR OpenCV image using cv2.QRCodeDetector.
    Returns: (decoded_text, success, bbox)
    """
    if bgr_image is None or bgr_image.size == 0:
        return "", False, None

    detector = cv2.QRCodeDetector()
    decoded_text, points, _ = detector.detectAndDecode(bgr_image)
    if decoded_text:
        return decoded_text.strip(), True, points

    # If first pass fails, try equalized grayscale for higher contrast in dim light
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    decoded_text, points, _ = detector.detectAndDecode(equalized)
    if decoded_text:
        return decoded_text.strip(), True, points

    return "", False, None


def scan_qr_from_bytes(image_bytes: bytes) -> tuple[str, str | None]:
    """
    Decode an image from raw file bytes and scan for a verification QR code.
    Returns: (extracted_token, raw_payload_or_None)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return "", None

    raw_payload, found, _ = scan_qr_from_bgr(image)
    if not found or not raw_payload:
        return "", None

    token = extract_token_from_payload(raw_payload)
    return token, raw_payload
