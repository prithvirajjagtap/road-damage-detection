import os
import io
import html
import sqlite3
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Road Damage Intelligence System",
    page_icon="🚧",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "train-5", "weights", "best.pt"
)

DB_PATH = os.path.join(BASE_DIR, "inspection_history.db")

# --------------------------------------------------
# DARK AI COMMAND CENTER THEME
# --------------------------------------------------

st.markdown("""
<style>
/* Main application */
.stApp {
    background: linear-gradient(135deg, #080F1C 0%, #101A2B 100%);
    color: #E8F0FC;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Main header */
.title-box {
    background: linear-gradient(110deg, #10243A, #12364A);
    color: #F4F9FF;
    border: 1px solid #24516B;
    border-left: 5px solid #38D9F5;
    text-align: left;
    font-size: 30px;
    font-weight: 750;
    padding: 24px 28px;
    border-radius: 14px;
    margin-bottom: 16px;
    letter-spacing: 0.3px;
}

/* Supporting introduction */
.intro-text {
    color: #A8B8CC;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 24px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0A1322;
    border-right: 1px solid #22334A;
}

[data-testid="stSidebar"] * {
    color: #DCE8F7;
}

/* Headings */
h1, h2, h3 {
    color: #F0F6FF !important;
    letter-spacing: 0.2px;
}

h2 {
    padding-bottom: 8px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #087FA5, #0BA6C7);
    color: #FFFFFF;
    border: 1px solid #24C5E5;
    border-radius: 9px;
    padding: 0.65rem 1rem;
    font-weight: 650;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #13B9D8;
    border-color: #7AE8FA;
    color: #06111E;
}

/* Inputs and upload areas */
[data-testid="stFileUploader"] {
    background: #111E30;
    border: 1px dashed #2E5875;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #111E30;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111E30;
    border: 1px solid #263C55;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #9EB4CC;
}

[data-testid="stMetricValue"] {
    color: #45DDF5;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #AFC2D8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #43D9F4;
}

/* Data tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid #26364C;
    border-radius: 10px;
    overflow: hidden;
}

/* Alerts and messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Horizontal separators */
hr {
    border-color: #26364C;
}

/* Images */
[data-testid="stImage"] img {
    border-radius: 10px;
}

/* Footer */
.dashboard-footer {
    color: #8296AF;
    font-size: 12px;
    text-align: center;
    padding: 22px 0 8px 0;
    border-top: 1px solid #26364C;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title-box">🛣️ RoadVision AI '
    '<span style="color:#45DDF5;">|</span> '
    'Road Damage Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="intro-text">'
    'AI-powered road inspection using YOLO11n. Detect potholes, '
    'cracks, and manholes, review preliminary severity, and '
    'generate maintenance guidance from road images.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_date TEXT,
                image_name TEXT,
                damage_type TEXT,
                confidence REAL,
                severity TEXT,
                urgency TEXT,
                recommendation TEXT
            )
        """)

        # Add dimension columns to existing databases if needed.
        existing_columns = {
            row[1] for row in
            conn.execute("PRAGMA table_info(inspections)").fetchall()
        }

        if "width_px" not in existing_columns:
            conn.execute(
                "ALTER TABLE inspections ADD COLUMN width_px REAL"
            )

        if "height_px" not in existing_columns:
            conn.execute(
                "ALTER TABLE inspections ADD COLUMN height_px REAL"
            )


def save_inspection(
    image_name,
    damage_type,
    confidence,
    severity,
    urgency,
    recommendation,
    width_px,
    height_px
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO inspections (
                inspection_date,
                image_name,
                damage_type,
                confidence,
                severity,
                urgency,
                recommendation,
                width_px,
                height_px
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            damage_type,
            confidence,
            severity,
            urgency,
            recommendation,
            width_px,
            height_px
        ))


def load_history():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM inspections ORDER BY id DESC",
            conn
        )


initialize_database()

# --------------------------------------------------
# LOAD EXISTING TRAINED YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# REPAIR RECOMMENDATION ENGINE
# Recommendations are shown as short bullet points.
# --------------------------------------------------

REPAIR_GUIDANCE = {
    "pothole": {
        "repair": [
            "Inspect pothole depth and size.",
            "Remove loose material and apply suitable patching.",
            "Compact the repair material properly.",
            "Check drainage and surrounding pavement."
        ],
        "inspection": [
            "Check for water accumulation.",
            "Inspect nearby pavement for damage."
        ]
    },
    "crack": {
        "repair": [
            "Inspect crack width and pattern.",
            "Clean and seal suitable surface cracks.",
            "Monitor cracks for further expansion.",
            "Arrange further assessment for wide or recurring cracks."
        ],
        "inspection": [
            "Check crack depth and surrounding pavement.",
            "Look for interconnected cracks or deformation."
        ]
    },
    "manhole": {
        "repair": [
            "Check whether the cover is loose or broken.",
            "Inspect the surrounding road surface.",
            "Adjust uneven levels if required.",
            "Repair or replace a damaged cover."
        ],
        "inspection": [
            "Check cover stability and alignment.",
            "Look for settlement around the manhole."
        ]
    },
    "unknown": {
        "repair": [
            "Confirm the type of damage.",
            "Arrange manual inspection.",
            "Select a repair method after assessment."
        ],
        "inspection": [
            "Verify the damage on site."
        ]
    }
}


def assess_damage(damage_type, box_area_ratio):
    """
    Preliminary image-based assessment.
    Bounding-box area is not a direct measurement of severity.
    """

    damage = damage_type.lower().strip()

    if damage not in REPAIR_GUIDANCE:
        damage = "unknown"

    if damage == "pothole":
        if box_area_ratio >= 0.12:
            severity = "High"
            urgency = "Prompt on-site inspection recommended"
        elif box_area_ratio >= 0.04:
            severity = "Medium"
            urgency = "Timely inspection recommended"
        else:
            severity = "Low"
            urgency = "Routine inspection recommended"

    elif damage == "crack":
        if box_area_ratio >= 0.12:
            severity = "High"
            urgency = "Prompt crack inspection recommended"
        elif box_area_ratio >= 0.04:
            severity = "Medium"
            urgency = "Timely crack inspection recommended"
        else:
            severity = "Low"
            urgency = "Routine crack inspection recommended"

    elif damage == "manhole":
        if box_area_ratio >= 0.12:
            severity = "High"
            urgency = "Prompt cover and surrounding road inspection"
        elif box_area_ratio >= 0.04:
            severity = "Medium"
            urgency = "Timely cover and surface inspection"
        else:
            severity = "Low"
            urgency = "Routine cover and surface inspection"

    else:
        severity = "Unclassified"
        urgency = "Manual inspection required"

    repair_points = REPAIR_GUIDANCE[damage]["repair"]
    inspection_points = REPAIR_GUIDANCE[damage]["inspection"]

    recommendation = (
        "**Recommended Actions**\n"
        + "\n".join(f"- {point}" for point in repair_points)
        + "\n\n**Inspection Checklist**\n"
        + "\n".join(f"- {point}" for point in inspection_points)
        + f"\n\n**Preliminary Severity:** {severity}"
        + "\n- Final repair decisions require on-site inspection."
    )

    return severity, urgency, recommendation


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Damage Detection",
        "Inspection Dashboard",
        "Maintenance Reports"
    ]
)

# --------------------------------------------------
# PAGE 1: DAMAGE DETECTION
# --------------------------------------------------

if page == "Damage Detection":

    # Compact upload header
    st.markdown("""
    <div style="
        background: linear-gradient(110deg, #10243A, #12364A);
        border: 1px solid #24516B;
        border-left: 5px solid #38D9F5;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;">
        <div>
            <div style="font-size:24px;font-weight:750;color:#F4F9FF;">
                Road Damage Analysis
            </div>
            <div style="font-size:13px;color:#A8B8CC;margin-top:4px;">
                Intelligent road inspection and damage assessment
            </div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#12364A;color:#45DDF5;
                border:1px solid #24516B;border-radius:20px;
                padding:6px 12px;font-size:12px;font-weight:700;">
                YOLO11n
            </span>
            <span style="background:#123A30;color:#58D6A2;
                border:1px solid #22634D;border-radius:20px;
                padding:6px 12px;font-size:12px;font-weight:700;">
                ● Model Ready
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pre-upload KPI cards
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Supported Classes", "03")
        st.caption("Potholes · Cracks · Manholes")

    with k2:
        st.metric("Model Engine", "YOLO11n")
        st.caption("Custom-trained detector")

    with k3:
        st.metric("Inference Speed", "Measured live")
        st.caption("Depends on image and CPU")

    with k4:
        st.metric("System Status", "Ready")
        st.caption("Upload an image to begin")

    st.markdown("### 📤 Upload Road Image")
    st.caption(
        "Choose a road image or try one from your existing dataset. "
        "Supported formats: JPG, JPEG, PNG."
    )

    # Sample images from the existing dataset
    sample_dir = os.path.join(BASE_DIR, "dataset", "data", "images")
    sample_images = []

    if os.path.isdir(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_images.append(filename)

    selected_sample = None

    if sample_images:
        sample_col1, sample_col2 = st.columns([3, 1])

        with sample_col1:
            selected_sample = st.selectbox(
                "Quick sample image",
                options=["None — upload my own"] + sample_images,
                key="road_sample_image"
            )

        with sample_col2:
            st.write("")
            st.write("")
            st.caption(f"{len(sample_images)}+ dataset images available")

    uploaded_file = st.file_uploader(
        "Drag and drop your image here",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear road photo for AI-based damage detection."
    )

    image = None
    image_name = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name

    elif selected_sample and selected_sample != "None — upload my own":
        sample_path = os.path.join(sample_dir, selected_sample)
        image = Image.open(sample_path).convert("RGB")
        image_name = selected_sample

    if image is not None:
        # Center the uploaded image without changing the rest of the layout.
        _, preview_col, _ = st.columns([1, 2, 1])
        with preview_col:
            st.image(
                image,
                caption=f"Selected image: {image_name}",
                width=600
            )

        # Center the Analyze button and keep it medium-sized.
        _, analyze_col, _ = st.columns([1, 1, 1])
        with analyze_col:
            analyze_clicked = st.button(
                "🔍 Analyze Road Damage",
                use_container_width=True,
                type="primary"
            )

        if analyze_clicked:
            with st.spinner("YOLO11n is analyzing the road image..."):
                start_time = time.perf_counter()

                results = model.predict(
                    source=image,
                    conf=0.20,
                    verbose=False
                )

                inference_ms = (time.perf_counter() - start_time) * 1000

                result = results[0]
                annotated_image = result.plot()

                st.session_state["analysis"] = {
                    "image_name": image_name,
                    "original_image": image.copy(),
                    "annotated_image": annotated_image,
                    "detections": [],
                    "inference_ms": round(inference_ms, 1)
                }

                image_width, image_height = image.size
                total_area = image_width * image_height

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

                    box_width_px = max(0, x2 - x1)
                    box_height_px = max(0, y2 - y1)
                    box_area = box_width_px * box_height_px
                    area_ratio = box_area / total_area

                    damage_type = model.names.get(
                        class_id, str(class_id)
                    )

                    severity, urgency, recommendation = assess_damage(
                        damage_type, area_ratio
                    )

                    st.session_state["analysis"]["detections"].append({
                        "damage_type": damage_type,
                        "confidence": confidence,
                        "width_px": round(box_width_px, 1),
                        "height_px": round(box_height_px, 1),
                        "severity": severity,
                        "urgency": urgency,
                        "recommendation": recommendation,
                        "area_ratio": area_ratio,
                        # Keep original-image coordinates for the zoomed crop.
                        "bbox": (
                            max(0, int(x1)),
                            max(0, int(y1)),
                            min(image_width, int(x2)),
                            min(image_height, int(y2))
                        )
                    })

            st.success("Analysis completed successfully.")

    # Show analysis after button click, including on reruns
    analysis = st.session_state.get("analysis")

    if analysis is not None:

        # Balanced side-by-side comparison with a compact image width.
        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Original Image")
            st.image(
                analysis["original_image"],
                width=600
            )

        with right:
            st.subheader("Detection Result")
            st.image(
                analysis["annotated_image"],
                channels="BGR",
                width=600
            )

        detections = analysis["detections"]

        st.subheader("Inspection Summary")

        if detections:
            # One row per detected instance, ready for CSV and PDF export.
            summary_rows = []
            for number, item in enumerate(detections, start=1):
                summary_rows.append({
                    "Instance": number,
                    "Class": item["damage_type"],
                    "Confidence (%)": round(item["confidence"] * 100, 2),
                    "Dimensions (px)": (
                        f"W {item['width_px']:.1f} × H {item['height_px']:.1f}"
                    ),
                    "Bounding-Box Width (px)": item["width_px"],
                    "Bounding-Box Height (px)": item["height_px"],
                    "Severity Grade": item["severity"],
                    "Recommended Action": item["recommendation"],
                    "Recommended Urgency": item["urgency"]
                })

            summary_df = pd.DataFrame(summary_rows)
            export_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
            export_col, pdf_col = st.columns([1, 1])
            with export_col:
                st.download_button(
                    "📥 Export Summary Report (CSV)",
                    data=export_csv,
                    file_name="road_damage_inspection_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="export_damage_summary_csv"
                )
            with pdf_col:
                try:
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import A4, landscape
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.lib.units import mm
                    from reportlab.platypus import (
                        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    )

                    pdf_buffer = io.BytesIO()
                    pdf_doc = SimpleDocTemplate(
                        pdf_buffer, pagesize=landscape(A4),
                        rightMargin=10 * mm, leftMargin=10 * mm,
                        topMargin=10 * mm, bottomMargin=10 * mm
                    )
                    pdf_styles = getSampleStyleSheet()
                    pdf_story = [
                        Paragraph("Road Damage Inspection Summary", pdf_styles["Title"]),
                        Spacer(1, 6),
                        Paragraph(
                            "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pdf_styles["Normal"]
                        ),
                        Spacer(1, 10)
                    ]
                    table_data = [[
                        "#", "Class", "Confidence", "Dimensions (px)",
                        "Severity", "Recommended Action"
                    ]]
                    for row in summary_rows:
                        action = str(row["Recommended Action"]).replace("\n", " ")
                        table_data.append([
                            str(row["Instance"]), str(row["Class"]),
                            f"{row['Confidence (%)']:.2f}%",
                            str(row["Dimensions (px)"]),
                            str(row["Severity Grade"]), action[:180]
                        ])
                    pdf_table = Table(
                        table_data, repeatRows=1,
                        colWidths=[10*mm, 28*mm, 24*mm, 35*mm, 24*mm, 130*mm]
                    )
                    pdf_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12364A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
                    ]))
                    pdf_story.extend([
                        pdf_table, Spacer(1, 8),
                        Paragraph(
                            "Severity and recommended actions are preliminary image-based estimates. "
                            "Confirm findings through on-site inspection.",
                            pdf_styles["Normal"]
                        )
                    ])
                    pdf_doc.build(pdf_story)
                    with pdf_col:
                        st.download_button(
                            "📄 Export Summary Report (PDF)",
                            data=pdf_buffer.getvalue(),
                            file_name="road_damage_inspection_summary.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="export_damage_summary_pdf"
                        )
                except ImportError:
                    with pdf_col:
                        st.caption("PDF export requires reportlab. Install it with: pip install reportlab")

        c1, c2, c3 = st.columns(3)

        c1.metric("Damage Instances", len(detections))

        c2.metric(
            "High Severity",
            sum(d["severity"] == "High" for d in detections)
        )

        c3.metric(
            "Classes Detected",
            len(set(d["damage_type"] for d in detections))
        )

        if not detections:
            st.info(
                "No damage was detected at the current confidence "
                "threshold. This does not guarantee that the road "
                "is damage-free."
            )

        for index, detection in enumerate(detections, start=1):
            severity = str(detection["severity"]).strip().title()
            if severity == "High":
                severity_badge = "🔴 High Severity"
            elif severity == "Low":
                severity_badge = "🟢 Low Severity"
            else:
                severity_badge = f"🟡 {severity} Severity"

            expander_title = (
                f"{severity_badge}  |  Damage {index}: {detection['damage_type']}  |  "
                f"Confidence {detection['confidence']:.1%}"
            )

            with st.expander(expander_title, expanded=False):
                # Show the exact detected region next to its metrics.
                preview_col, metrics_col = st.columns([1, 2], gap="medium")
                with preview_col:
                    try:
                        original = analysis["original_image"]
                        x1, y1, x2, y2 = detection.get(
                            "bbox", (0, 0, original.width, original.height)
                        )
                        x1 = max(0, min(int(x1), original.width))
                        y1 = max(0, min(int(y1), original.height))
                        x2 = max(x1 + 1, min(int(x2), original.width))
                        y2 = max(y1 + 1, min(int(y2), original.height))
                        cropped_damage = original.crop((x1, y1, x2, y2))
                        st.image(
                            cropped_damage,
                            caption=f"Cropped {detection['damage_type']} region",
                            use_container_width=True
                        )
                    except (ValueError, TypeError, AttributeError):
                        st.info("Cropped preview is unavailable for this detection.")

                with metrics_col:
                    a, b, c = st.columns(3)
                    a.metric("Model Confidence", f"{detection['confidence']:.1%}")
                    b.metric("Estimated Severity", detection["severity"])

                    # A compact custom Dimensions card avoids st.metric's
                    # automatic ellipsis on longer width/height values.
                    dimensions_text = html.escape(
                        f"W {detection['width_px']:.1f} × H {detection['height_px']:.1f} px"
                    )
                    with c:
                        st.markdown(
                            "<div style='box-sizing:border-box; min-height:105px; "
                            "height:100%; padding:14px 10px; border:1px solid #263C55; "
                            "border-radius:12px; background:#111E30; overflow:visible;'>"
                            "<div style='font-size:0.82rem; line-height:1.25; "
                            "color:#9EB4CC; margin-bottom:12px;'>Dimensions</div>"
                            f"<div style='font-size:clamp(0.68rem, 1vw, 0.88rem); "
                            "font-weight:600; line-height:1.45; color:#45DDF5; "
                            "white-space:normal; overflow-wrap:anywhere; "
                            f"word-break:normal;'>{dimensions_text}</div>"
                            "</div>",
                            unsafe_allow_html=True
                        )

                # Use a wrapping text card instead of st.metric, which can
                # truncate long urgency descriptions with an ellipsis.
                urgency_text = html.escape(str(detection["urgency"]))
                st.markdown(
                    "<div style='margin: 0.5rem 0 1rem; padding: 0.8rem 1rem; "
                    "border: 1px solid #2C526B; border-radius: 10px; "
                    "background: rgba(13, 35, 54, 0.75); "
                    "white-space: normal; overflow-wrap: anywhere; "
                    "word-break: break-word;'>"
                    "<div style='font-size: 0.82rem; color: #A8B8CC; "
                    "margin-bottom: 0.3rem;'>Recommended Urgency</div>"
                    f"<div style='font-size: 1rem; font-weight: 650; "
                    f"color: #F0F6FF; line-height: 1.5;'>{urgency_text}</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

                st.markdown("**🛠️ Smart Repair Recommendations**")
                st.markdown(detection["recommendation"])

        if detections:
            st.caption(
                "Dimensions are measured in image pixels, not actual road "
                "measurements in centimetres. Depth cannot be reliably "
                "measured from an ordinary single image alone; physical "
                "measurement or suitable depth/camera calibration data is needed."
            )

        st.warning(
            "Severity is estimated from bounding-box area, not actual "
            "depth, structural damage, or road safety. Confirm findings "
            "through physical inspection."
        )

        if detections:

            if st.button(
                "Save Inspection to History",
                use_container_width=True
            ):

                if (
                    st.session_state.get("saved_image_name")
                    == analysis["image_name"]
                    and st.session_state.get("saved_detections")
                    == len(detections)
                ):
                    st.info(
                        "This analysis has already been saved in this session."
                    )
                else:
                    for detection in detections:
                        save_inspection(
                            analysis["image_name"],
                            detection["damage_type"],
                            detection["confidence"],
                            detection["severity"],
                            detection["urgency"],
                            detection["recommendation"],
                            detection["width_px"],
                            detection["height_px"]
                        )

                    st.session_state["saved_image_name"] = (
                        analysis["image_name"]
                    )
                    st.session_state["saved_detections"] = len(detections)

                    st.success("Inspection saved successfully.")

# --------------------------------------------------
# PAGE 2: INSPECTION DASHBOARD
# --------------------------------------------------

elif page == "Inspection Dashboard":

    st.header("Inspection Analytics Dashboard")

    history = load_history()

    if history.empty:
        st.info(
            "No saved inspections yet. Analyze an image and save "
            "the results first."
        )
    else:
        a, b, c = st.columns(3)

        a.metric("Total Records", len(history))

        b.metric(
            "High Severity Records",
            int((history["severity"] == "High").sum())
        )

        c.metric(
            "Damage Types",
            history["damage_type"].nunique()
        )

        st.subheader("Damage Type Distribution")
        st.bar_chart(history["damage_type"].value_counts())

        st.subheader("Severity Distribution")
        st.bar_chart(history["severity"].value_counts())

        st.subheader("Recent Inspection Records")
        st.dataframe(history, use_container_width=True)

# --------------------------------------------------
# PAGE 3: MAINTENANCE REPORTS
# --------------------------------------------------

elif page == "Maintenance Reports":

    st.header("Maintenance Reports")

    history = load_history()

    if history.empty:
        st.info("No inspection records available for reporting.")
    else:
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="road_maintenance_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Prepare PDF Report"):

            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import (
                    SimpleDocTemplate,
                    Paragraph,
                    Spacer,
                    Table,
                    TableStyle
                )
                from reportlab.lib.units import mm

                buffer = io.BytesIO()

                document = SimpleDocTemplate(
                    buffer,
                    pagesize=landscape(A4),
                    rightMargin=12 * mm,
                    leftMargin=12 * mm,
                    topMargin=12 * mm,
                    bottomMargin=12 * mm
                )

                styles = getSampleStyleSheet()

                story = [
                    Paragraph(
                        "Road Damage Inspection Report",
                        styles["Title"]
                    ),
                    Spacer(1, 8),
                    Paragraph(
                        "Generated: " + datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        styles["Normal"]
                    ),
                    Spacer(1, 12),
                    Paragraph(
                        "Severity is an image-based estimate. "
                        "Pixel dimensions are bounding-box measurements, "
                        "not physical dimensions. Depth and repair "
                        "recommendations require on-site verification.",
                        styles["Normal"]
                    ),
                    Spacer(1, 12)
                ]

                columns = [
                    "ID",
                    "Date",
                    "Image",
                    "Damage",
                    "Confidence",
                    "Width (px)",
                    "Height (px)",
                    "Severity",
                    "Urgency"
                ]

                table_data = [columns]

                for _, row in history.iterrows():

                    width_value = (
                        f"{float(row['width_px']):.1f}"
                        if pd.notna(row["width_px"]) else "N/A"
                    )

                    height_value = (
                        f"{float(row['height_px']):.1f}"
                        if pd.notna(row["height_px"]) else "N/A"
                    )

                    table_data.append([
                        str(row["id"]),
                        str(row["inspection_date"]),
                        str(row["image_name"])[:20],
                        str(row["damage_type"]),
                        f"{float(row['confidence']):.1%}",
                        width_value,
                        height_value,
                        str(row["severity"]),
                        str(row["urgency"])[:27]
                    ])

                table = Table(
                    table_data,
                    repeatRows=1,
                    colWidths=[
                        10 * mm,
                        29 * mm,
                        30 * mm,
                        19 * mm,
                        19 * mm,
                        18 * mm,
                        18 * mm,
                        19 * mm,
                        48 * mm
                    ]
                )

                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.whitesmoke]),
                ]))

                story.append(table)
                document.build(story)

                st.download_button(
                    "Download PDF Report",
                    data=buffer.getvalue(),
                    file_name="road_maintenance_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            except ImportError:
                st.error(
                    "PDF support is not installed. Run: "
                    "pip install reportlab"
                )
