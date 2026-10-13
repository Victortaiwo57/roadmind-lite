"""RoadMind Lite — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from roadmind import ui
from roadmind.config import (
    ACCEPTED_IMAGE_TYPES,
    APP_NAME,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_MODEL,
    MAX_IMAGE_SIDE,
    MODEL_OPTIONS,
)
from roadmind.detector import DetectionResult, InvalidImageError, load_image, run_inference
from roadmind.exports import class_summary, detections_frame, export_stem, to_csv, to_json, to_png
from roadmind.taxonomy import CATEGORY_ORDER

logger = logging.getLogger("roadmind")

st.set_page_config(
    page_title=f"{APP_NAME} · Road-scene detection",
    page_icon="🚦",
    layout="wide",
    menu_items={"About": f"{APP_NAME}: a research prototype for road-scene object detection."},
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading detection model…")
def get_model(weights: str):
    from ultralytics import YOLO  # imported lazily so the UI renders before torch loads

    return YOLO(weights)


@st.cache_data(show_spinner=False, max_entries=16)
def detect(
    image_bytes: bytes, weights: str, confidence: float, iou: float, road_only: bool
) -> DetectionResult:
    image, resized, original_size = load_image(image_bytes)
    result = run_inference(
        get_model(weights),
        image,
        model_name=weights.removesuffix(".pt"),
        confidence=confidence,
        iou=iou,
        road_relevant_only=road_only,
    )
    result.resized = resized
    result.original_size = original_size
    return result


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> tuple[str, float, float, bool]:
    with st.sidebar:
        st.subheader("Model")
        labels = {m.weights: m for m in MODEL_OPTIONS}
        weights = st.selectbox(
            "Detector",
            options=list(labels),
            index=list(labels).index(DEFAULT_MODEL),
            format_func=lambda w: labels[w].label,
            help="Pretrained on COCO (80 classes). Weights download on first use.",
        )
        st.caption(labels[weights].note)

        st.subheader("Thresholds")
        confidence = st.slider(
            "Confidence",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_CONFIDENCE,
            step=0.05,
            help="Minimum score for a detection to be kept. Higher means fewer, more certain detections.",
        )
        iou = st.slider(
            "Overlap (IoU) for NMS",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_IOU,
            step=0.05,
            help="Boxes overlapping more than this are merged. Lower values suppress more duplicates.",
        )

        st.subheader("Scope")
        road_only = st.toggle(
            "Road-relevant classes only",
            value=True,
            help="Restrict detection to road users, vehicles, traffic control, roadside objects and animals.",
        )

        st.divider()
        st.caption(
            "**Research direction** · video tracking, temporal behaviour classification, "
            "domain-shift evaluation (e.g. Lagos vs. BDD100K scenes) and uncertainty analysis."
        )
    return weights, confidence, iou, road_only


# --------------------------------------------------------------------------- #
# Result views
# --------------------------------------------------------------------------- #
def render_metrics(result: DetectionResult) -> None:
    cols = st.columns(5)
    cols[0].metric("Objects detected", result.count)
    cols[1].metric("Road users", result.road_users, help="Vulnerable road users, vehicles and animals.")
    cols[2].metric(
        "Vulnerable road users",
        result.vulnerable_road_users,
        help="Pedestrians, cyclists and motorcyclists.",
    )
    mean = result.mean_confidence
    cols[3].metric("Mean confidence", f"{mean:.0%}" if mean is not None else "—")
    cols[4].metric("Inference time", f"{result.inference_ms:,.0f} ms", help="Model forward pass only, on this server.")


def render_images(original, result: DetectionResult) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Input**")
        st.image(original, width="stretch")
    with right:
        st.markdown("**Detections**")
        st.image(result.annotated, width="stretch")
    note = f"{result.image_width} × {result.image_height} px"
    if result.resized:
        w, h = result.original_size
        note += f" (downscaled from {w} × {h} for analysis, max side {MAX_IMAGE_SIDE}px)"
    st.caption(note)


def render_breakdown(result: DetectionResult) -> None:
    df = detections_frame(result)
    summary = class_summary(df)

    chart_col, table_col = st.columns([1.1, 1], gap="large")
    with chart_col:
        st.markdown("**Detections by class**")
        chart = ui.category_chart(summary)
        st.altair_chart(chart, width="stretch", height=chart.height + 60)
    with table_col:
        st.markdown("**By category**")
        by_cat = (
            df.groupby("category")
            .agg(Count=("label", "size"), Classes=("label", "nunique"), Mean_conf=("confidence", "mean"))
            .reindex([c for c in CATEGORY_ORDER if c in set(df["category"])])
            .reset_index()
            .rename(columns={"category": "Category", "Mean_conf": "Mean confidence"})
        )
        st.dataframe(
            by_cat,
            hide_index=True,
            width="stretch",
            column_config={
                "Mean confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            },
        )


def render_table(result: DetectionResult) -> None:
    df = detections_frame(result)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "label": "Class",
            "category": "Category",
            "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
            "x_min": st.column_config.NumberColumn("x min", format="%.0f"),
            "y_min": st.column_config.NumberColumn("y min", format="%.0f"),
            "x_max": st.column_config.NumberColumn("x max", format="%.0f"),
            "y_max": st.column_config.NumberColumn("y max", format="%.0f"),
            "width": st.column_config.NumberColumn("Width", format="%.0f px"),
            "height": st.column_config.NumberColumn("Height", format="%.0f px"),
        },
    )
    st.caption("Coordinates are in pixels of the analysed image, origin at top-left. Sorted by confidence.")


def render_exports(result: DetectionResult, source_name: str) -> None:
    stem = export_stem(source_name)
    st.markdown("Each export records the model, thresholds and image size so results can be reproduced.")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Detections (CSV)", to_csv(result, source_name), f"{stem}.csv", "text/csv",
        width="stretch", on_click="ignore", icon=":material/table_view:",
    )
    c2.download_button(
        "Detections + metadata (JSON)", to_json(result, source_name), f"{stem}.json", "application/json",
        width="stretch", on_click="ignore", icon=":material/data_object:",
    )
    c3.download_button(
        "Annotated image (PNG)", to_png(result), f"{stem}_annotated.png", "image/png",
        width="stretch", on_click="ignore", icon=":material/image:",
    )


def render_method() -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            """
**What this does**

- Runs a pretrained YOLO11 detector on a single image.
- Maps the 80 COCO classes onto road-scene categories: vulnerable road users,
  vehicles, traffic control, roadside objects and animals.
- Reports counts, confidence and bounding boxes for every detection.

**Open research questions**

1. Which road-user classes are detected reliably across varied scenes?
2. How robust are detections to lighting, weather, viewpoint and geographic domain shift?
3. Does temporal context improve behaviour classification over single-frame features?
"""
        )
    with right:
        st.markdown(
            """
**Limitations**

- Single-frame detection: no tracking, intent, attention or trajectory prediction.
- A detection does not establish a road user's behaviour or future movement.
- COCO has no classes for tricycles (keke), minibuses (danfo) or road signs other than
  stop signs, so local road users may be misclassified or missed.
- Performance varies with camera angle, weather, lighting, image quality and region.
- Results shown here are illustrative. Research claims need a defined dataset,
  ground truth and quantitative evaluation (precision, recall, mAP).
"""
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    ui.inject_styles()
    ui.header()
    weights, confidence, iou, road_only = sidebar()

    uploaded = st.file_uploader(
        "Road-scene image",
        type=list(ACCEPTED_IMAGE_TYPES),
        help="Use an image you have permission to analyse. Images are processed in memory and not stored.",
    )

    if uploaded is None:
        ui.empty_state()
        with st.expander("Method and limitations"):
            render_method()
        ui.footer()
        return

    image_bytes = uploaded.getvalue()
    try:
        original, _, _ = load_image(image_bytes)
        with st.spinner("Analysing scene…"):
            result = detect(image_bytes, weights, confidence, iou, road_only)
    except InvalidImageError:
        st.error("That file couldn't be read as an image. Try a JPG, PNG or WebP file.", icon=":material/broken_image:")
        ui.footer()
        return
    except Exception as exc:  # noqa: BLE001 — surface any model/runtime failure gracefully
        logger.exception("Inference failed")
        st.error(
            "The model couldn't complete the analysis. This is usually a dependency or memory issue on the server.",
            icon=":material/error:",
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        ui.footer()
        return

    render_metrics(result)
    st.write("")
    render_images(original, result)

    if result.count == 0:
        st.warning(
            "No objects were detected above the current confidence threshold. "
            "Try lowering it in the sidebar, or turn off *Road-relevant classes only*.",
            icon=":material/search_off:",
        )
    else:
        st.write("")
        tab_breakdown, tab_table, tab_export, tab_method = st.tabs(
            ["Breakdown", f"Detections ({result.count})", "Export", "Method & limitations"]
        )
        with tab_breakdown:
            render_breakdown(result)
        with tab_table:
            render_table(result)
        with tab_export:
            render_exports(result, uploaded.name)
        with tab_method:
            render_method()

    ui.footer()


main()



'''
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from io import BytesIO

st.set_page_config(
    page_title="RoadMind Lite | Road Scene Understanding",
    page_icon="🚦",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#1f526b); color: white;}
.hero h1 {color: white; margin-bottom: .25rem;}
.hero p {color: #d9e8f0; margin-bottom: 0;}
.small-note {font-size: .88rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚦 RoadMind Lite</h1>
  <p>Interactive baseline for road-scene object detection and visual scene exploration.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write(
    "RoadMind Lite uses a pretrained object detector to identify objects in a road-scene image. "
    "It is a starting point for research into road-user behaviour understanding—not a safety-critical system."
)

@st.cache_resource(show_spinner="Loading pretrained detection model…")
def load_model():
    # The small pretrained model is downloaded on first run.
    return YOLO("yolo11n.pt")

with st.sidebar:
    st.header("Analysis settings")
    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10, max_value=0.90, value=0.35, step=0.05,
        help="Higher thresholds show fewer detections, generally with greater model confidence."
    )
    st.caption("Model: YOLO11n pretrained on COCO classes")
    st.divider()
    st.subheader("Research direction")
    st.write("Next steps: video tracking, temporal behaviour classification, domain-shift evaluation, and uncertainty analysis.")

uploaded = st.file_uploader(
    "Upload a road-scene image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a road or traffic image that you have permission to analyse."
)

if uploaded is None:
    st.info("Upload an image to run detection. For best results, use a clear street scene with visible road users.")
    st.markdown("### Research questions this prototype motivates")
    st.markdown("""
    - Which road-user classes are detected reliably, and which are missed?
    - How does the confidence threshold change the number and type of detections?
    - How might temporal information improve understanding beyond single-frame detection?
    """)
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

try:
    model = load_model()
    with st.spinner("Analysing road scene…"):
        result = model.predict(image, conf=confidence, verbose=False)[0]
    annotated = result.plot()
    # Ultralytics returns a BGR array; convert to RGB for Streamlit.
    import cv2
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    rows = []
    names = result.names
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, clss):
            rows.append({
                "class": names.get(int(cls_id), str(cls_id)),
                "confidence": round(float(score), 4),
                "x_min": round(float(box[0]), 1),
                "y_min": round(float(box[1]), 1),
                "x_max": round(float(box[2]), 1),
                "y_max": round(float(box[3]), 1),
            })

    with right:
        st.subheader("Detected objects")
        st.image(annotated_rgb, use_container_width=True)

    df = pd.DataFrame(rows)
    st.divider()
    st.subheader("Detection summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", len(df))
    c2.metric("Unique classes", int(df["class"].nunique()) if not df.empty else 0)
    c3.metric("Mean confidence", f"{df['confidence'].mean():.1%}" if not df.empty else "—")

    if df.empty:
        st.warning("No objects were detected above this threshold. Try lowering the confidence threshold or using another image.")
    else:
        counts = df["class"].value_counts().rename_axis("Class").reset_index(name="Count")
        chart_col, table_col = st.columns([0.85, 1.15], gap="large")
        with chart_col:
            st.markdown("**Objects by class**")
            st.bar_chart(counts.set_index("Class"))
        with table_col:
            st.markdown("**Detection details**")
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download detections as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="roadmind_detections.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Interpretation and limitations")
    st.markdown("""
    - This is **single-image object detection**, not full road-user behaviour understanding.
    - A detected object does not establish its intention, attention, or future movement.
    - Performance can change with camera angle, weather, lighting, image quality, and geographic context.
    - The pretrained model is a baseline. Research claims require a defined dataset, ground truth, and quantitative evaluation.
    """)
    st.caption("Research prototype by Victor Taiwo · Built with Streamlit and Ultralytics YOLO.")
except Exception as exc:
    st.error("The model could not complete inference. Check the app logs and package versions.")
    st.exception(exc)
'''
