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
