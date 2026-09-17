import streamlit as st
import cv2
import os
from ultralytics import YOLO

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CCTV Distance Analyzer",
    page_icon="📹",
    layout="wide"
)

# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📹 CCTV Distance Analyzer")

st.write(
    "Detect a common object in CCTV footage "
    "and prepare it for distance calculation."
)

st.divider()

# --------------------------------------------------
# 1. UPLOAD VIDEOS
# --------------------------------------------------

st.subheader("1. Upload CCTV Footage")

uploaded_files = st.file_uploader(
    "Select CCTV video files",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} video(s) selected."
    )

    for file in uploaded_files:

        file_path = os.path.join(
            UPLOAD_DIR,
            file.name
        )

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        st.write(f"✅ {file.name}")

st.divider()

# --------------------------------------------------
# 2. COMMON OBJECT
# --------------------------------------------------

st.subheader("2. Select Common Object")

object_name = st.text_input(
    "Object to analyze",
    placeholder="Example: person, chair, car, bottle"
)

st.divider()

# --------------------------------------------------
# 3. ANALYZE
# --------------------------------------------------

st.subheader("3. Analyze Videos")

if st.button("🔍 Analyze Videos"):

    if not uploaded_files:

        st.warning(
            "Please upload at least one video."
        )

    elif not object_name:

        st.warning(
            "Please enter the common object."
        )

    else:

        st.success(
            f"Searching for: **{object_name}**"
        )

        # --------------------------------------------------
        # PROCESS EACH VIDEO
        # --------------------------------------------------

        for file_index, file in enumerate(
            uploaded_files,
            start=1
        ):

            st.write("---")

            st.header(
                f"📹 Camera {file_index}: {file.name}"
            )

            file_path = os.path.join(
                UPLOAD_DIR,
                file.name
            )

            # --------------------------------------------------
            # OPEN VIDEO
            # --------------------------------------------------

            cap = cv2.VideoCapture(
                file_path
            )

            if not cap.isOpened():

                st.error(
                    f"Could not open {file.name}"
                )

                continue

            # --------------------------------------------------
            # VIDEO INFORMATION
            # --------------------------------------------------

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            duration = (
                frame_count / fps
                if fps > 0
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Width",
                f"{width}px"
            )

            col2.metric(
                "Height",
                f"{height}px"
            )

            col3.metric(
                "FPS",
                f"{fps:.2f}"
            )

            col4.metric(
                "Duration",
                f"{duration:.2f}s"
            )

            # --------------------------------------------------
            # SELECT MIDDLE FRAME
            # --------------------------------------------------

            middle_frame = (
                frame_count // 2
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                middle_frame
            )

            success, frame = cap.read()

            if not success:

                st.warning(
                    "Could not read video frame."
                )

                cap.release()
                continue

            # --------------------------------------------------
            # YOLO DETECTION
            # --------------------------------------------------

            results = model(frame)

            detected_objects = []

            # --------------------------------------------------
            # PROCESS DETECTIONS
            # --------------------------------------------------

            for result in results:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    detected_name = model.names[
                        class_id
                    ]

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    # --------------------------------------------------
                    # GROUND CONTACT POINT
                    # --------------------------------------------------

                    ground_x = int(
                        (x1 + x2) / 2
                    )

                    ground_y = y2

                    detected_objects.append({

                        "name": detected_name,

                        "confidence": confidence,

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,

                        "ground_x": ground_x,
                        "ground_y": ground_y
                    })

            # --------------------------------------------------
            # FILTER TARGET CLASS
            # --------------------------------------------------

            matching_objects = [

                obj
                for obj in detected_objects

                if obj["name"].lower()
                == object_name.lower()

            ]

            # --------------------------------------------------
            # DRAW ONLY TARGET OBJECTS
            # --------------------------------------------------

            for index, obj in enumerate(
                matching_objects,
                start=1
            ):

                x1 = obj["x1"]
                y1 = obj["y1"]
                x2 = obj["x2"]
                y2 = obj["y2"]

                ground_x = obj["ground_x"]
                ground_y = obj["ground_y"]

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Ground point
                cv2.circle(
                    frame,
                    (ground_x, ground_y),
                    7,
                    (0, 0, 255),
                    -1
                )

                # Detection number
                cv2.putText(
                    frame,
                    f"OBJECT {index}",
                    (
                        x1,
                        max(y1 - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Draw line to ground point
                cv2.line(
                    frame,
                    (
                        int((x1 + x2) / 2),
                        y2 - 30
                    ),
                    (ground_x, ground_y),
                    (0, 0, 255),
                    2
                )

            # --------------------------------------------------
            # DISPLAY IMAGE
            # --------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                frame_rgb,
                caption=(
                    f"Detected {object_name} "
                    f"objects — {file.name}"
                ),
                use_container_width=True
            )

            # --------------------------------------------------
            # DETECTION RESULTS
            # --------------------------------------------------

            st.subheader(
                "🎯 Detected Target Objects"
            )

            if matching_objects:

                st.success(
                    f"Found "
                    f"{len(matching_objects)} "
                    f"'{object_name}' object(s)."
                )

                for index, obj in enumerate(
                    matching_objects,
                    start=1
                ):

                    st.write(
                        f"**Object {index}**"
                    )

                    st.write(
                        f"Confidence: "
                        f"{obj['confidence']:.2f}"
                    )

                    st.write(
                        f"Ground point: "
                        f"({obj['ground_x']}, "
                        f"{obj['ground_y']})"
                    )

                    st.write("---")

                # --------------------------------------------------
                # OBJECT SELECTION
                # --------------------------------------------------

                selected_index = st.selectbox(
                    "Select the object to measure",
                    options=list(
                        range(
                            1,
                            len(matching_objects) + 1
                        )
                    ),
                    key=f"object_{file_index}"
                )

                selected_object = (
                    matching_objects[
                        selected_index - 1
                    ]
                )

                st.success(
                    f"Selected Object "
                    f"{selected_index}"
                )

                st.write(
                    f"Ground-contact point: "
                    f"**("
                    f"{selected_object['ground_x']}, "
                    f"{selected_object['ground_y']}"
                    f")**"
                )

            else:

                st.warning(
                    f"No '{object_name}' "
                    f"was detected in this frame."
                )

                st.info(
                    "For the real project, we may "
                    "use a different detector if the "
                    "common object is not supported "
                    "by the standard YOLO model."
                )

            cap.release()

        st.divider()

        st.success(
            "Detection stage completed."
        )