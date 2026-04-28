from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]  # x1, y1, x2, y2


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO  # lazy import

    # Pretrained YOLOv8 nano; weights auto-download on first run.
    model = YOLO("yolov8n.pt")
    return model


def preprocess_image(img: Image.Image, *, max_size: int = 960) -> np.ndarray:
    """
    Returns BGR uint8 image for OpenCV + ultralytics.
    Resizes (keeping aspect ratio) for faster CPU inference.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = np.array(rgb)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def run_detection(model: Any, bgr: np.ndarray, *, conf: float = 0.25, iou: float = 0.45) -> list[Detection]:
    # Enforce CPU-only inference.
    results = model.predict(bgr, conf=conf, iou=iou, device="cpu", verbose=False)
    if not results:
        return []

    r0 = results[0]
    names: dict[int, str] = getattr(r0, "names", {}) or {}
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

    dets: list[Detection] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        label = names.get(int(k), str(int(k)))
        dets.append(
            Detection(
                label=label,
                confidence=float(c),
                box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return dets


def draw_boxes(bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 230, 118), 2)
        text = f"{d.label} {d.confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x = max(0, x1)
        y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x, y), (x + tw + 8, y + th + baseline + 6), (0, 230, 118), -1)
        cv2.putText(out, text, (x + 4, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def display_results(bgr_with_boxes: np.ndarray, dets: list[Detection]) -> None:
    rgb = cv2.cvtColor(bgr_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Detection result", use_container_width=True)

    if not dets:
        st.info("No objects detected.")
        return

    counts: dict[str, int] = {}
    for d in dets:
        counts[d.label] = counts.get(d.label, 0) + 1

    st.subheader("Object count")
    st.json(dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))))

    st.subheader("Detected objects")
    st.dataframe(
        [
            {
                "label": d.label,
                "confidence": round(d.confidence, 4),
                "x1": d.box_xyxy[0],
                "y1": d.box_xyxy[1],
                "x2": d.box_xyxy[2],
                "y2": d.box_xyxy[3],
            }
            for d in dets
        ],
        use_container_width=True,
        hide_index=True,
    )


st.set_page_config(page_title="Mobile Object Detection (YOLOv8)", layout="wide")
st.title("Mobile Object Detection App")
st.write("Capture with your phone camera or upload an image, then run YOLOv8n object detection on CPU.")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.1, 0.9, 0.45, 0.05)
    max_size = st.select_slider("Max resize (long side)", options=[640, 800, 960, 1120, 1280], value=960)

st.subheader("Input")
cam = st.camera_input("Camera capture")
upload = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "webp"])

img_bytes = None
if cam is not None:
    img_bytes = cam.getvalue()
elif upload is not None:
    img_bytes = upload.getvalue()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Preview")
    if img_bytes:
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_container_width=True)
    else:
        st.caption("Provide an image using the camera or uploader.")

run_btn = st.button("Run detection", type="primary", use_container_width=True, disabled=not bool(img_bytes))

if run_btn and img_bytes:
    img = Image.open(io.BytesIO(img_bytes))
    model = load_model()

    with st.spinner("Preprocessing..."):
        bgr = preprocess_image(img, max_size=max_size)

    with st.spinner("Running YOLOv8 inference (CPU)..."):
        dets = run_detection(model, bgr, conf=conf, iou=iou)

    with st.spinner("Rendering results..."):
        vis = draw_boxes(bgr, dets)

    with col2:
        st.subheader("Result")
        display_results(vis, dets)
