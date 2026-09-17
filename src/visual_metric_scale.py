"""
STEP 10F — VISUAL METRIC SCALE RECOVERY

Attempts to recover a metric scale from CCTV footage using visual
metric cues.

Primary cue:
    A person detected in multiple camera views.

The system reconstructs a relative person height and compares it
with a configurable statistical human-height prior.

IMPORTANT:
    This does NOT claim that every scene can produce metric scale.
    If no reliable cue is available, the result remains relative.

No airport-specific dimensions or camera metadata are required.
"""

import json
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


REPORT_PATH = "results/multi_camera_report.json"
OUTPUT_PATH = "results/visual_metric_scale.json"

# Statistical prior, NOT a physical measurement of the environment.
DEFAULT_PERSON_HEIGHT_M = 1.70

SUPPORTED_VIDEO = (".mp4", ".avi", ".mov", ".mkv")


def load_report(path=REPORT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_videos(folder="videos"):
    videos = []

    if not os.path.isdir(folder):
        return videos

    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(SUPPORTED_VIDEO):
                videos.append(os.path.join(root, name))

    return sorted(videos)


def read_middle_frame(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return frame


def detect_people(frame, model):
    """
    Detect people.

    Returns:
        list of dictionaries containing bounding boxes and confidence.
    """

    results = model(frame, verbose=False)

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class 0 = person
            if cls != 0 or conf < 0.50:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "top": [((x1 + x2) / 2.0), y1],
                    "bottom": [((x1 + x2) / 2.0), y2],
                }
            )

    return detections


def estimate_relative_height_from_pair(
    report,
    camera_a,
    camera_b,
    person_height_m=DEFAULT_PERSON_HEIGHT_M,
):
    """
    Look for a successful cross-camera pair.

    At this stage the existing Step 9 report does not store the
    individual correspondence points required to reconstruct a
    person's top and bottom points.

    Therefore this function deliberately refuses to fabricate a
    scale from camera-to-object distances alone.
    """

    pairs = report.get("cross_camera_pairs", [])

    for pair in pairs:
        if (
            pair.get("camera_a") == camera_a
            and pair.get("camera_b") == camera_b
        ):
            tri = pair.get("triangulation", {})

            if not tri.get("success"):
                continue

            return {
                "status": "cue_requires_point_level_reconstruction",
                "camera_a": camera_a,
                "camera_b": camera_b,
                "person_height_prior_m": person_height_m,
                "relative_person_height": None,
                "scale_m_per_relative_unit": None,
                "reason": (
                    "Step 9 contains camera-to-point distances but does "
                    "not retain person top/bottom correspondences. "
                    "A metric scale cannot be derived honestly from "
                    "those distances alone."
                ),
            }

    return {
        "status": "no_successful_pair",
        "camera_a": camera_a,
        "camera_b": camera_b,
        "person_height_prior_m": person_height_m,
        "relative_person_height": None,
        "scale_m_per_relative_unit": None,
    }


def analyze_visual_scale(
    video_folder="videos",
    report_path=REPORT_PATH,
    output_path=OUTPUT_PATH,
):
    print("=" * 70)
    print("STEP 10F — VISUAL METRIC SCALE RECOVERY")
    print("=" * 70)

    report = load_report(report_path)

    videos = discover_videos(video_folder)

    print(f"Videos discovered : {len(videos)}")

    if YOLO is None:
        result = {
            "status": "yolo_unavailable",
            "metric_scale_available": False,
            "scale_m_per_relative_unit": None,
            "reason": "Ultralytics YOLO is not installed.",
        }

        save_result(result, output_path)

        print("YOLO available : NO")
        print("Metric scale   : NOT AVAILABLE")
        print("=" * 70)

        return result

    print("YOLO available : YES")

    # Small pretrained model.
    # Used only as a visual-cue detector.
    model = YOLO("yolo11n.pt")

    video_results = []

    for video in videos:
        frame = read_middle_frame(video)

        if frame is None:
            video_results.append(
                {
                    "video": os.path.basename(video),
                    "frame_read": False,
                    "people_detected": 0,
                }
            )
            continue

        detections = detect_people(frame, model)

        video_results.append(
            {
                "video": os.path.basename(video),
                "frame_read": True,
                "people_detected": len(detections),
                "detections": detections,
            }
        )

        print(
            f"  {os.path.basename(video)} : "
            f"{len(detections)} person(s) detected"
        )

    pairs = report.get("cross_camera_pairs", [])

    successful_pairs = []

    for pair in pairs:
        tri = pair.get("triangulation", {})

        if tri.get("success"):
            successful_pairs.append(
                estimate_relative_height_from_pair(
                    report,
                    pair.get("camera_a"),
                    pair.get("camera_b"),
                )
            )

    # Important:
    # We do NOT generate a fake scale if Step 9 lacks point-level
    # person correspondences.
    metric_available = any(
        x.get("scale_m_per_relative_unit") is not None
        for x in successful_pairs
    )

    if metric_available:
        scale_values = [
            x["scale_m_per_relative_unit"]
            for x in successful_pairs
            if x.get("scale_m_per_relative_unit") is not None
        ]

        scale = float(np.median(scale_values))
        status = "metric_scale_recovered"

    else:
        scale = None
        status = "visual_metric_cue_not_yet_reconstructable"

    result = {
        "status": status,
        "metric_scale_available": metric_available,
        "scale_m_per_relative_unit": scale,
        "person_height_prior_m": DEFAULT_PERSON_HEIGHT_M,
        "video_analysis": video_results,
        "pair_analysis": successful_pairs,
        "method": (
            "Visual person-height cue with multi-view "
            "point-level reconstruction."
        ),
        "important_note": (
            "Metric scale is not fabricated. Step 9 currently "
            "does not retain the point-level correspondences needed "
            "to reconstruct person top/bottom points."
        ),
    }

    save_result(result, output_path)

    print()
    print(f"Metric scale : {'AVAILABLE' if metric_available else 'NOT AVAILABLE'}")

    if metric_available:
        print(f"Scale        : {scale:.6f} m / relative unit")
    else:
        print(
            "Reason       : "
            "Step 9 needs point-level metric-cue reconstruction."
        )

    print()
    print(f"Result saved : {output_path}")
    print("=" * 70)

    return result


def save_result(result, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    analyze_visual_scale()