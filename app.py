"""RoadMind Lite — road-scene object detection for road-user research.

A single-file Streamlit app. Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from typing import Final

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

APP_VERSION = "1.0.0"
logger = logging.getLogger("roadmind")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_NAME = "RoadMind Lite"
APP_TAGLINE = "Road-scene object detection for road-user research"
AUTHOR = "Victor Taiwo"


@dataclass(frozen=True)
class ModelOption:
    weights: str
    label: str
    note: str


MODEL_OPTIONS: tuple[ModelOption, ...] = (
    ModelOption("yolo11n.pt", "YOLO11n (nano)", "Fastest. Recommended for hosted demos."),
    ModelOption("yolo11s.pt", "YOLO11s (small)", "Better recall, about 3× slower on CPU."),
    ModelOption("yolo11m.pt", "YOLO11m (medium)", "Most accurate here. Needs more memory."),
)
DEFAULT_MODEL = MODEL_OPTIONS[0].weights

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.50

# Longest image side passed to the model. Larger uploads are downscaled to keep
# memory use and latency predictable on shared hosting.
MAX_IMAGE_SIDE = 1920

ACCEPTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
MAX_UPLOAD_MB = 20


# --------------------------------------------------------------------------- #
# Road-scene taxonomy (COCO class -> road category)
# --------------------------------------------------------------------------- #
VULNERABLE = "Vulnerable road users"
VEHICLES = "Vehicles"
TRAFFIC_CONTROL = "Traffic control"
ROADSIDE = "Roadside objects"
ANIMALS = "Animals"
OTHER = "Other"

CATEGORY_ORDER: Final[tuple[str, ...]] = (
    VULNERABLE,
    VEHICLES,
    TRAFFIC_CONTROL,
    ROADSIDE,
    ANIMALS,
    OTHER,
)

CLASS_TO_CATEGORY: Final[dict[str, str]] = {
    "person": VULNERABLE,
    "bicycle": VULNERABLE,
    "motorcycle": VULNERABLE,
    "car": VEHICLES,
    "bus": VEHICLES,
    "truck": VEHICLES,
    "train": VEHICLES,
    "traffic light": TRAFFIC_CONTROL,
    "stop sign": TRAFFIC_CONTROL,
    "fire hydrant": ROADSIDE,
    "parking meter": ROADSIDE,
    "bench": ROADSIDE,
    "dog": ANIMALS,
    "horse": ANIMALS,
    "sheep": ANIMALS,
    "cow": ANIMALS,
}

# Categories counted as road users (moving participants in traffic).
ROAD_USER_CATEGORIES: Final[frozenset[str]] = frozenset({VULNERABLE, VEHICLES, ANIMALS})

# Colours per category, used consistently in charts.
CATEGORY_COLORS: Final[dict[str, str]] = {
    VULNERABLE: "#D9480F",
    VEHICLES: "#1864AB",
    TRAFFIC_CONTROL: "#2B8A3E",
    ROADSIDE: "#5F3DC4",
    ANIMALS: "#A61E4D",
    OTHER: "#868E96",
}


def category_for(class_name: str) -> str:
    """Return the road-scene category for a COCO class name."""
    return CLASS_TO_CATEGORY.get(class_name.strip().lower(), OTHER)


def is_road_user(class_name: str) -> bool:
    return category_for(class_name) in ROAD_USER_CATEGORIES


def is_vulnerable(class_name: str) -> bool:
    return category_for(class_name) == VULNERABLE


def road_relevant_class_ids(names: dict[int, str]) -> list[int]:
    """IDs of model classes that belong to any category other than ``Other``."""
    return sorted(i for i, name in names.items() if category_for(name) != OTHER)


# --------------------------------------------------------------------------- #
# Image loading, inference and result parsing
# --------------------------------------------------------------------------- #


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


@dataclass(frozen=True)
class Detection:
    label: str
    category: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DetectionResult:
    detections: list[Detection]
    annotated: np.ndarray  # RGB, uint8
    image_width: int
    image_height: int
    inference_ms: float
    model_name: str
    confidence_threshold: float
    iou_threshold: float
    road_relevant_only: bool
    resized: bool = False
    original_size: tuple[int, int] = field(default=(0, 0))

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def road_users(self) -> int:
        return sum(is_road_user(d.label) for d in self.detections)

    @property
    def vulnerable_road_users(self) -> int:
        return sum(is_vulnerable(d.label) for d in self.detections)

    @property
    def mean_confidence(self) -> float | None:
        if not self.detections:
            return None
        return float(np.mean([d.confidence for d in self.detections]))


def load_image(data: bytes, max_side: int = MAX_IMAGE_SIDE) -> tuple[Image.Image, bool, tuple[int, int]]:
    """Decode bytes to an RGB image, honour EXIF rotation and cap the longest side.

    Returns ``(image, was_resized, original_size)``.
    """
    try:
        img = Image.open(BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The file could not be read as an image.") from exc

    original_size = img.size
    resized = False
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        resized = True
    return img, resized, original_size


def parse_boxes(
    names: dict[int, str],
    xyxy: Iterable[Sequence[float]],
    confidences: Iterable[float],
    class_ids: Iterable[int],
) -> list[Detection]:
    """Convert raw model outputs to ``Detection`` objects, sorted by confidence."""
    detections = []
    for box, score, cls_id in zip(xyxy, confidences, class_ids, strict=True):
        label = names.get(int(cls_id), str(int(cls_id)))
        detections.append(
            Detection(
                label=label,
                category=category_for(label),
                confidence=round(float(score), 4),
                x_min=round(float(box[0]), 1),
                y_min=round(float(box[1]), 1),
                x_max=round(float(box[2]), 1),
                y_max=round(float(box[3]), 1),
            )
        )
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def run_inference(
    model,  # ultralytics.YOLO
    image: Image.Image,
    *,
    model_name: str,
    confidence: float,
    iou: float,
    road_relevant_only: bool,
) -> DetectionResult:
    """Run the detector on one image and package the outputs."""
    classes = road_relevant_class_ids(model.names) if road_relevant_only else None

    start = time.perf_counter()
    result = model.predict(image, conf=confidence, iou=iou, classes=classes, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        detections = parse_boxes(
            result.names,
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        )
    else:
        detections = []

    # Ultralytics draws in BGR; flip channels for display.
    annotated = np.ascontiguousarray(result.plot(line_width=2)[..., ::-1])

    return DetectionResult(
        detections=detections,
        annotated=annotated,
        image_width=image.width,
        image_height=image.height,
        inference_ms=elapsed_ms,
        model_name=model_name,
        confidence_threshold=confidence,
        iou_threshold=iou,
        road_relevant_only=road_relevant_only,
    )


# --------------------------------------------------------------------------- #
# Summary tables and exports
# --------------------------------------------------------------------------- #
DETECTION_COLUMNS = [
    "label",
    "category",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
]


def detections_frame(result: DetectionResult) -> pd.DataFrame:
    """One row per detection."""
    rows = [
        {**asdict(d), "width": round(d.width, 1), "height": round(d.height, 1)}
        for d in result.detections
    ]
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and mean confidence per class, ordered by category then count."""
    if df.empty:
        return pd.DataFrame(columns=["category", "label", "count", "mean_confidence"])
    summary = (
        df.groupby(["category", "label"], as_index=False)
        .agg(count=("label", "size"), mean_confidence=("confidence", "mean"))
    )
    summary["category"] = pd.Categorical(summary["category"], CATEGORY_ORDER, ordered=True)
    summary = summary.sort_values(["category", "count"], ascending=[True, False])
    summary["category"] = summary["category"].astype(str)
    return summary.reset_index(drop=True)


def run_metadata(result: DetectionResult, source_name: str) -> dict:
    return {
        "app": "roadmind-lite",
        "app_version": APP_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_image": source_name,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "model": result.model_name,
        "confidence_threshold": result.confidence_threshold,
        "iou_threshold": result.iou_threshold,
        "road_relevant_only": result.road_relevant_only,
        "inference_ms": round(result.inference_ms, 1),
        "objects_detected": result.count,
    }


def to_csv(result: DetectionResult, source_name: str) -> bytes:
    df = detections_frame(result)
    df.insert(0, "source_image", source_name)
    df["model"] = result.model_name
    df["confidence_threshold"] = result.confidence_threshold
    return df.to_csv(index=False).encode("utf-8")


def to_json(result: DetectionResult, source_name: str) -> bytes:
    payload = {
        "metadata": run_metadata(result, source_name),
        "detections": detections_frame(result).to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def to_png(result: DetectionResult) -> bytes:
    buffer = BytesIO()
    Image.fromarray(result.annotated).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def export_stem(source_name: str) -> str:
    stem = source_name.rsplit(".", 1)[0] if "." in source_name else source_name
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
    return f"roadmind_{safe or 'image'}"


# --------------------------------------------------------------------------- #
# Presentation: styles, header, empty state, charts
# --------------------------------------------------------------------------- #
_CSS = """
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1280px;}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}

.rm-header {display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1.5rem; flex-wrap: wrap; padding: .4rem 0 1.1rem;
  border-bottom: 1px solid rgba(128,128,128,.22); margin-bottom: 1.4rem;}
.rm-eyebrow {font-size: .72rem; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--rm-accent); margin: 0 0 .35rem;}
.rm-title {font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 0; letter-spacing: -.01em;}
.rm-sub {margin: .35rem 0 0; opacity: .72; font-size: 1rem;}
.rm-tags {display: flex; gap: .4rem; flex-wrap: wrap;}
.rm-tag {font-size: .75rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid rgba(128,128,128,.3); opacity: .85; white-space: nowrap;}

.rm-step {border: 1px solid rgba(128,128,128,.22); border-radius: 10px;
  padding: 1rem 1.1rem; height: 100%;}
.rm-step-n {font-size: .72rem; font-weight: 700; color: var(--rm-accent);
  letter-spacing: .08em; text-transform: uppercase;}
.rm-step h4 {margin: .3rem 0 .35rem; font-size: 1rem; padding: 0;}
.rm-step p {margin: 0; font-size: .9rem; opacity: .75; line-height: 1.45;}

.rm-footer {margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.22);
  font-size: .8rem; opacity: .6; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem;}

[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 10px;
  padding: .75rem 1rem;}
[data-testid="stMetricLabel"] p {font-size: .8rem; opacity: .75;}

:root {--rm-accent: #E8433F;}
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def header() -> None:
    tags = ["Research prototype", "YOLO11 · COCO", f"v{APP_VERSION}"]
    tag_html = "".join(f'<span class="rm-tag">{html.escape(t)}</span>' for t in tags)
    st.markdown(
        f"""
        <div class="rm-header">
          <div>
            <p class="rm-eyebrow">Road-scene understanding</p>
            <h1 class="rm-title">{html.escape(APP_NAME)}</h1>
            <p class="rm-sub">{html.escape(APP_TAGLINE)}</p>
          </div>
          <div class="rm-tags">{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state() -> None:
    steps = [
        ("Upload", "Add a street or traffic photo (JPG, PNG or WebP) that you have permission to analyse."),
        ("Detect", "A pretrained YOLO11 model locates road users, vehicles and traffic control objects."),
        (
            "Review & export",
            "Inspect counts and confidence by category, then download CSV, JSON or the annotated image.",
        ),
    ]
    cols = st.columns(3, gap="medium")
    for i, (col, (title, body)) in enumerate(zip(cols, steps, strict=True), start=1):
        col.markdown(
            f"""<div class="rm-step"><div class="rm-step-n">Step {i}</div>
            <h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>""",
            unsafe_allow_html=True,
        )
    st.write("")
    st.caption(
        "Tip: clear daytime scenes with visible road users give the most informative results. "
        "Lower the confidence threshold to surface small or distant objects."
    )


def category_chart(summary: pd.DataFrame) -> alt.Chart:
    """Horizontal bar chart of detections per class, coloured by category."""
    present = [c for c in CATEGORY_ORDER if c in set(summary["category"])]
    color = alt.Color(
        "category:N",
        title=None,
        scale=alt.Scale(domain=present, range=[CATEGORY_COLORS[c] for c in present]),
        legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=200),
    )
    height = max(140, 30 * len(summary) + 50)
    return (
        alt.Chart(summary)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("count:Q", title="Detections", axis=alt.Axis(tickMinStep=1, grid=True)),
            y=alt.Y("label:N", title=None, sort=list(dict.fromkeys(summary["label"]))),
            color=color,
            tooltip=[
                alt.Tooltip("label:N", title="Class"),
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Detections"),
                alt.Tooltip("mean_confidence:Q", title="Mean confidence", format=".1%"),
            ],
        )
        .properties(height=height)
    )


def footer() -> None:
    st.markdown(
        f"""<div class="rm-footer">
        <span>{html.escape(APP_NAME)} v{APP_VERSION} · Research prototype by {html.escape(AUTHOR)}</span>
        <span>Not for driving decisions or safety-critical use.</span></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Streamlit app
# --------------------------------------------------------------------------- #

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
    ms = result.inference_ms
    cols[4].metric(
        "Inference time",
        f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms",
        help="Model forward pass only, on this server.",
    )


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
        chart = category_chart(summary)
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
    inject_styles()
    header()
    weights, confidence, iou, road_only = sidebar()

    uploaded = st.file_uploader(
        "Road-scene image",
        type=list(ACCEPTED_IMAGE_TYPES),
        help=(
            f"Up to {MAX_UPLOAD_MB} MB. Use an image you have permission to analyse. "
            "Images are processed in memory and not stored."
        ),
    )

    if uploaded is None:
        empty_state()
        with st.expander("Method and limitations"):
            render_method()
        footer()
        return

    image_bytes = uploaded.getvalue()
    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"That image is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller file.",
            icon=":material/error:",
        )
        footer()
        return
    try:
        original, _, _ = load_image(image_bytes)
        with st.spinner("Analysing scene…"):
            result = detect(image_bytes, weights, confidence, iou, road_only)
    except InvalidImageError:
        st.error("That file couldn't be read as an image. Try a JPG, PNG or WebP file.", icon=":material/broken_image:")
        footer()
        return
    except Exception as exc:  # noqa: BLE001 — surface any model/runtime failure gracefully
        logger.exception("Inference failed")
        st.error(
            "The model couldn't complete the analysis. This is usually a dependency or memory issue on the server.",
            icon=":material/error:",
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        footer()
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

    footer()


main()
