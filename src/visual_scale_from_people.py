"""
STEP 10F — VISUAL METRIC SCALE FROM PEOPLE

Uses people visible in the CCTV footage as a visual metric cue.

Pipeline:
    1. Load successful multi-camera reconstruction.
    2. Read the best frames used by Step 9.
    3. Detect people in both views.
    4. Match a likely common person.
    5. Use the person's top and bottom image points.
    6. Triangulate top and bottom into the existing relative 3D system.
    7. Estimate relative person height.
    8. Convert relative units to metres using a statistical height prior.
    9. Convert existing camera-object distances to metres.

IMPORTANT:
    The human-height prior is a statistical prior, not a measurement
    of the airport/environment.

    If the visual cue is unreliable, metric scale is NOT reported.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


REPORT_PATH = "results/multi_camera_report.json"
OUTPUT_PATH = "results/visual_scale_result.json"

# Statistical prior used only as a visual scale cue.
DEFAULT_PERSON_HEIGHT_M = 1.70

MIN_PERSON_CONFIDENCE = 0.50
MIN_MATCH_SCORE = 0.35


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return frame


def detect_people(frame, model):
    """
    Returns person detections.

    Each detection:
        bbox
        confidence
        top
        bottom
        center
    """

    results = model(frame, verbose=False)

    people = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO class 0 = person
            if cls != 0:
                continue

            if confidence < MIN_PERSON_CONFIDENCE:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            people.append(
                {
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "confidence": confidence,
                    "top": [
                        float((x1 + x2) / 2.0),
                        float(y1),
                    ],
                    "bottom": [
                        float((x1 + x2) / 2.0),
                        float(y2),
                    ],
                    "center": [
                        float((x1 + x2) / 2.0),
                        float((y1 + y2) / 2.0),
                    ],
                }
            )

    return people


def crop_detection(frame, detection):
    x1, y1, x2, y2 = detection["bbox"]

    h, w = frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def appearance_similarity(crop_a, crop_b):
    """
    Lightweight appearance comparison.

    Uses:
        - color histogram
        - resized grayscale correlation

    This is only used to rank candidate people.
    """

    if crop_a is None or crop_b is None:
        return 0.0

    try:
        a = cv2.resize(crop_a, (128, 256))
        b = cv2.resize(crop_b, (128, 256))

        hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

        hist_a = cv2.calcHist(
            [hsv_a],
            [0, 1],
            None,
            [32, 32],
            [0, 180, 0, 256],
        )

        hist_b = cv2.calcHist(
            [hsv_b],
            [0, 1],
            None,
            [32, 32],
            [0, 180, 0, 256],
        )

        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)

        histogram_score = cv2.compareHist(
            hist_a,
            hist_b,
            cv2.HISTCMP_CORREL,
        )

        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

        correlation = np.corrcoef(
            gray_a.astype(np.float64).ravel(),
            gray_b.astype(np.float64).ravel(),
        )[0, 1]

        if not np.isfinite(correlation):
            correlation = 0.0

        histogram_score = max(
            0.0,
            min(1.0, (histogram_score + 1.0) / 2.0),
        )

        correlation = max(
            0.0,
            min(1.0, (correlation + 1.0) / 2.0),
        )

        return float(
            0.65 * histogram_score
            + 0.35 * correlation
        )

    except Exception:
        return 0.0


def match_people(
    frame_a,
    people_a,
    frame_b,
    people_b,
):
    """
    Find the most plausible same-person pair.
    """

    candidates = []

    for index_a, person_a in enumerate(people_a):
        crop_a = crop_detection(frame_a, person_a)

        for index_b, person_b in enumerate(people_b):
            crop_b = crop_detection(frame_b, person_b)

            appearance = appearance_similarity(
                crop_a,
                crop_b,
            )

            # Prefer detections with reasonably similar relative
            # appearance and plausible body proportions.
            height_a = (
                person_a["bbox"][3]
                - person_a["bbox"][1]
            )

            height_b = (
                person_b["bbox"][3]
                - person_b["bbox"][1]
            )

            if height_a <= 0 or height_b <= 0:
                continue

            ratio = min(height_a, height_b) / max(
                height_a,
                height_b,
            )

            score = (
                0.75 * appearance
                + 0.25 * ratio
            )

            candidates.append(
                {
                    "index_a": index_a,
                    "index_b": index_b,
                    "appearance_score": appearance,
                    "size_similarity": ratio,
                    "score": score,
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    if best["score"] < MIN_MATCH_SCORE:
        return None

    return best


def triangulate_point(P_a, P_b, point_a, point_b):
    """
    Standard linear triangulation.
    """

    pa = np.asarray(point_a, dtype=np.float64).reshape(2)
    pb = np.asarray(point_b, dtype=np.float64).reshape(2)

    P1 = np.asarray(P_a, dtype=np.float64)
    P2 = np.asarray(P_b, dtype=np.float64)

    A = np.vstack(
        [
            pa[0] * P1[2] - P1[0],
            pa[1] * P1[2] - P1[1],
            pb[0] * P2[2] - P2[0],
            pb[1] * P2[2] - P2[1],
        ]
    )

    _, _, Vt = np.linalg.svd(A)

    X = Vt[-1]

    if abs(X[3]) < 1e-10:
        return None

    X = X / X[3]

    return X[:3]


def point_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return float(np.linalg.norm(a - b))


def extract_projection_matrices(pair):
    projections = pair.get(
        "projection_matrices",
        {},
    )

    P_a = projections.get("camera_a")
    P_b = projections.get("camera_b")

    if P_a is None or P_b is None:
        return None, None

    return (
        np.asarray(P_a, dtype=np.float64),
        np.asarray(P_b, dtype=np.float64),
    )


def estimate_scale_for_pair(
    pair,
    video_folder,
    model,
    person_height_m,
):
    camera_a = pair.get("camera_a")
    camera_b = pair.get("camera_b")

    frame_a_index = pair.get("best_frame_a")
    frame_b_index = pair.get("best_frame_b")

    if (
        camera_a is None
        or camera_b is None
        or frame_a_index is None
        or frame_b_index is None
    ):
        return {
            "success": False,
            "reason": "Missing camera/frame information.",
        }

    video_a = Path(video_folder) / camera_a
    video_b = Path(video_folder) / camera_b

    # Search recursively if videos are inside subdirectories.
    if not video_a.exists():
        matches = list(
            Path(video_folder).rglob(camera_a)
        )
        if matches:
            video_a = matches[0]

    if not video_b.exists():
        matches = list(
            Path(video_folder).rglob(camera_b)
        )
        if matches:
            video_b = matches[0]

    frame_a = read_frame(
        video_a,
        frame_a_index,
    )

    frame_b = read_frame(
        video_b,
        frame_b_index,
    )

    if frame_a is None or frame_b is None:
        return {
            "success": False,
            "reason": "Could not read Step 9 best frames.",
        }

    people_a = detect_people(
        frame_a,
        model,
    )

    people_b = detect_people(
        frame_b,
        model,
    )

    print(
        f"  {camera_a}: {len(people_a)} person(s)"
    )

    print(
        f"  {camera_b}: {len(people_b)} person(s)"
    )

    if not people_a or not people_b:
        return {
            "success": False,
            "reason": (
                "A person was not detected in both "
                "camera views."
            ),
        }

    person_match = match_people(
        frame_a,
        people_a,
        frame_b,
        people_b,
    )

    if person_match is None:
        return {
            "success": False,
            "reason": (
                "No sufficiently reliable common-person "
                "match was found."
            ),
        }

    person_a = people_a[
        person_match["index_a"]
    ]

    person_b = people_b[
        person_match["index_b"]
    ]

    P_a, P_b = extract_projection_matrices(pair)

    if P_a is None or P_b is None:
        return {
            "success": False,
            "reason": (
                "Step 9 projection matrices unavailable."
            ),
        }

    top_3d = triangulate_point(
        P_a,
        P_b,
        person_a["top"],
        person_b["top"],
    )

    bottom_3d = triangulate_point(
        P_a,
        P_b,
        person_a["bottom"],
        person_b["bottom"],
    )

    if top_3d is None or bottom_3d is None:
        return {
            "success": False,
            "reason": "Person top/bottom triangulation failed.",
        }

    relative_height = point_distance(
        top_3d,
        bottom_3d,
    )

    if not np.isfinite(relative_height):
        return {
            "success": False,
            "reason": "Invalid reconstructed person height.",
        }

    if relative_height < 1e-6:
        return {
            "success": False,
            "reason": "Reconstructed person height is degenerate.",
        }

    scale = (
        float(person_height_m)
        / relative_height
    )

    return {
        "success": True,
        "camera_a": camera_a,
        "camera_b": camera_b,
        "best_frame_a": int(frame_a_index),
        "best_frame_b": int(frame_b_index),
        "person_match_score": float(
            person_match["score"]
        ),
        "person_match_appearance": float(
            person_match["appearance_score"]
        ),
        "person_height_prior_m": float(
            person_height_m
        ),
        "person_top_3d_relative": top_3d.tolist(),
        "person_bottom_3d_relative": bottom_3d.tolist(),
        "person_height_relative": float(
            relative_height
        ),
        "scale_m_per_relative_unit": float(
            scale
        ),
    }


def convert_distances(report, scale):
    converted = []

    for pair in report.get(
        "cross_camera_pairs",
        [],
    ):
        tri = pair.get(
            "triangulation",
            {},
        )

        da = tri.get(
            "camera_a_distance_relative"
        )

        db = tri.get(
            "camera_b_distance_relative"
        )

        if da is not None:
            converted.append(
                {
                    "camera": pair.get("camera_a"),
                    "distance_relative": float(da),
                    "distance_m": float(
                        da * scale
                    ),
                }
            )

        if db is not None:
            converted.append(
                {
                    "camera": pair.get("camera_b"),
                    "distance_relative": float(db),
                    "distance_m": float(
                        db * scale
                    ),
                }
            )

    return converted


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recover visual metric scale from "
            "people in multi-camera CCTV footage."
        )
    )

    parser.add_argument(
        "--report",
        default=REPORT_PATH,
    )

    parser.add_argument(
        "--videos",
        default="videos",
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
    )

    parser.add_argument(
        "--person-height",
        type=float,
        default=DEFAULT_PERSON_HEIGHT_M,
        help=(
            "Statistical human-height prior in metres."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STEP 10F — VISUAL METRIC SCALE RECOVERY")
    print("=" * 70)

    if YOLO is None:
        print(
            "ERROR: Ultralytics YOLO is not installed."
        )
        return

    if not os.path.exists(args.report):
        print(
            f"ERROR: Report not found: {args.report}"
        )
        return

    report = load_json(args.report)

    successful_pairs = [
        pair
        for pair in report.get(
            "cross_camera_pairs",
            [],
        )
        if pair.get(
            "triangulation",
            {},
        ).get("success", False)
    ]

    print(
        f"Successful 3D pairs : "
        f"{len(successful_pairs)}"
    )

    if not successful_pairs:
        output = {
            "status": "no_successful_3d_pair",
            "metric_scale_available": False,
            "scale_m_per_relative_unit": None,
            "scale_cue": "person_height_prior",
            "pair_results": [],
            "note": (
                "No successful multi-view 3D pair was available, so no metric "
                "scale could be recovered. Per-camera single-view estimates "
                "remain available from the visual metric layer."
            ),
        }

        save_json(
            output,
            args.output,
        )

        print(
            "No successful 3D pair is available."
        )
        print(
            f"Result saved : {args.output}"
        )
        return

    print()
    print("Loading visual detector...")

    model = YOLO("yolo11n.pt")

    scale_results = []

    print()
    print("-" * 70)
    print("VISUAL SCALE CUE")
    print("-" * 70)

    for pair in successful_pairs:

        print(
            f"\nAnalyzing "
            f"{pair.get('camera_a')} <-> "
            f"{pair.get('camera_b')}"
        )

        result = estimate_scale_for_pair(
            pair,
            args.videos,
            model,
            args.person_height,
        )

        scale_results.append(result)

        if result.get("success"):
            print(
                "  Common person : FOUND"
            )

            print(
                f"  Relative height : "
                f"{result['person_height_relative']:.6f}"
            )

            print(
                f"  Scale : "
                f"{result['scale_m_per_relative_unit']:.6f} "
                f"m / relative unit"
            )

        else:
            print(
                "  Metric cue : NOT RELIABLE"
            )

            print(
                f"  Reason : "
                f"{result.get('reason', 'unknown')}"
            )

    valid_scales = [
        r["scale_m_per_relative_unit"]
        for r in scale_results
        if r.get("success")
        and r.get("scale_m_per_relative_unit")
        is not None
    ]

    if not valid_scales:

        output = {
            "status": "metric_scale_not_recovered",
            "metric_scale_available": False,
            "scale_m_per_relative_unit": None,
            "scale_cue": "person_height_prior",
            "pair_results": scale_results,
            "note": (
                "No reliable common-person visual cue "
                "was reconstructed. Relative distances "
                "remain the valid output."
            ),
        }

        save_json(
            output,
            args.output,
        )

        print()
        print(
            "METRIC SCALE : NOT RECOVERED"
        )
        print(
            "Relative distances remain valid."
        )
        print(
            f"Result saved : {args.output}"
        )
        print("=" * 70)

        return

    # Robust aggregation across successful pairs.
    scale = float(
        np.median(
            np.asarray(
                valid_scales,
                dtype=np.float64,
            )
        )
    )

    metric_distances = convert_distances(
        report,
        scale,
    )

    output = {
        "status": "metric_scale_recovered_with_visual_prior",
        "metric_scale_available": True,
        "scale_m_per_relative_unit": scale,
        "scale_cue": "person_height_prior",
        "person_height_prior_m": float(
            args.person_height
        ),
        "pair_results": scale_results,
        "metric_distances": metric_distances,
        "important_note": (
            "Metric scale depends on the statistical "
            "human-height prior and therefore carries "
            "uncertainty. It is not a direct measured "
            "environment dimension."
        ),
    }

    save_json(
        output,
        args.output,
    )

    print()
    print("=" * 70)
    print("METRIC SCALE RECOVERED")
    print("=" * 70)

    print(
        f"Scale : {scale:.6f} m / relative unit"
    )

    print()
    print("Metric distances:")

    for item in metric_distances:
        print(
            f"  {item['camera']} : "
            f"{item['distance_m']:.3f} m"
        )

    print()
    print(
        "STATUS: Visual metric conversion available."
    )

    print(
        f"Result saved : {args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()