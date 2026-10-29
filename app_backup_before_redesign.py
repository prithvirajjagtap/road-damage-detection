
import os
import io
import sqlite3
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
# STYLING
# --------------------------------------------------


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
    border: 1px solid #263C55;
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
# --------------------------------------------------

REPAIR_GUIDANCE = {
    "pothole": {
        "repair": (
            "Inspect the pothole depth and underlying pavement. "
            "If suitable, remove loose material, prepare the surface, "
            "apply appropriate patching material, and compact it."
        ),
        "inspection": (
            "Check pothole depth, water accumulation, and surrounding "
            "pavement condition before selecting a repair method."
        )
    },
    "crack": {
        "repair": (
            "Inspect crack width and pattern. Suitable narrow, "
            "non-structural cracks may be cleaned and sealed using "
            "an appropriate crack sealant. Wider or recurring cracks "
            "need further assessment."
        ),
        "inspection": (
            "Check whether the crack is narrow, wide, interconnected, "
            "or associated with pavement deformation."
        )
    },
    "manhole": {
        "repair": (
            "Inspect the cover, frame, surrounding pavement, and level. "
            "If damaged or unstable, arrange repair or replacement by "
            "the responsible maintenance team."
        ),
        "inspection": (
            "Check cover stability, surface level, visible damage, "
            "and whether the surrounding road has settled."
        )
    },
    "unknown": {
        "repair": (
            "The detected class does not match the configured repair "
            "categories. Arrange a manual inspection before deciding "
            "on maintenance."
        ),
        "inspection": (
            "Confirm the damage type during an on-site inspection."
        )
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

    repair = REPAIR_GUIDANCE[damage]["repair"]
    inspection = REPAIR_GUIDANCE[damage]["inspection"]

    recommendation = (
        f"Recommended action: {repair}\n\n"
        f"Inspection checklist: {inspection}\n\n"
        f"Preliminary image-based severity: {severity}.\n"
        "Actual depth, physical dimensions, pavement condition, and "
        "safety risk must be checked on site before deciding the "
        "final repair method and urgency."
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

    st.header("Road Damage Inspection")

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded road image",
            use_container_width=True
        )

        if st.button("Analyze Road Damage", use_container_width=True):

            with st.spinner("Analyzing road image..."):

                results = model.predict(
                    source=image,
                    conf=0.20,
                    verbose=False
                )

                result = results[0]
                annotated_image = result.plot()

                st.session_state["analysis"] = {
                    "image_name": uploaded_file.name,
                    "original_image": image.copy(),
                    "annotated_image": annotated_image,
                    "detections": []
                }

                image_width, image_height = image.size
                total_area = image_width * image_height

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().tolist()
                    )

                    # Bounding-box dimensions in pixels
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
                        "area_ratio": area_ratio
                    })

            st.success("Image analysis completed.")

    # Show analysis after button click, including on reruns
    analysis = st.session_state.get("analysis")

    if analysis is not None:

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(
                analysis["original_image"],
                use_container_width=True
            )

        with right:
            st.subheader("Detection Result")
            st.image(
                analysis["annotated_image"],
                channels="BGR",
                use_container_width=True
            )

        detections = analysis["detections"]

        st.subheader("Inspection Summary")

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

            st.divider()
            st.subheader(
                f"Damage {index}: {detection['damage_type']}"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Model Confidence",
                f"{detection['confidence']:.1%}"
            )

            b.metric(
                "Estimated Severity",
                detection["severity"]
            )

            c.metric(
                "Recommended Urgency",
                detection["urgency"]
            )

            # New image-dimension measurements
            d1, d2 = st.columns(2)

            d1.metric(
                "Bounding-Box Width",
                f"{detection['width_px']:.1f} px"
            )

            d2.metric(
                "Bounding-Box Height",
                f"{detection['height_px']:.1f} px"
            )

            st.caption(
                "These measurements describe the detected rectangle "
                "in image pixels, not actual road dimensions in cm."
            )

            st.info(
                "Depth cannot be reliably measured from this ordinary "
                "single image alone. A physical measurement or suitable "
                "depth/camera calibration data is required."
            )

            st.markdown("**Recommended maintenance action**")
            st.write(detection["recommendation"])

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
