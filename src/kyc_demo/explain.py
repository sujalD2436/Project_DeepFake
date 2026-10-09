from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from .model import bgr_crop_to_tensor


def gradcam_overlay_base64(model, crop_bgr, fake_index: int, device: torch.device) -> str:
    """Return a Grad-CAM overlay PNG as base64; this is attribution, not a mask."""
    activation = {}
    target_layer = model.features[-1]

    def capture(_module, _inputs, output):
        activation["value"] = output

    handle = target_layer.register_forward_hook(capture)
    try:
        tensor = bgr_crop_to_tensor(crop_bgr).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        feature_map = activation["value"]
        gradient = torch.autograd.grad(logits[0, fake_index], feature_map)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=crop_bgr.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop_bgr, 0.58, heatmap, 0.42, 0)
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Could not encode Grad-CAM preview.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    finally:
        handle.remove()

