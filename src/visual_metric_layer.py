"""
Visual Metric Layer
-------------------

Per-camera metric estimates that do NOT depend on two-view reconstruction:
- camera -> ground (camera mounting height) from the image horizon
- camera -> object distance from pinhole projection

Geometry used
-------------
For a camera at height h looking at a ground plane, points on the plane at
distance d project to

    v = y_ground - y_horizon   (vertical pixel offset above the horizon)

with

    v = f * h / d      =>      d = f * h / v

The horizon line is recovered per camera from ground-plane vanishing points
(src/robust_geometry.analyze), the focal length from self-calibration or a
dimension fallback, and the height h from a human-height prior (YOLO person
detection): a person standing on the ground with real height H_p projects to

    d = f * H_p / person_height_px

and h then follows from the same plane projection. The human-height prior is a
statistical assumption and every metric value is reported as an estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from robust_geometry import analyze as analyze_ground_geometry
    from self_calibration import self_calibrate
except ImportError:
    from robust_geometry import analyze as analyze_ground_geometry
    from self_calibration import self_calibrate

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

DEFAULT_PERSON_HEIGHT_M = 1.70
PERSON_HEIGHT_SIGMA_M = 0.10
MIN_PERSON_HEIGHT_PX = 55

# A person smaller than this is too far away for the height prior to give a
# trustworthy scale cue (one pixel of error then shifts metres).
MIN_SCALE_PERSON_PX = 80

# Physically plausible mounting heights for a surveillance camera (metres).
# Estimates outside this range are numerical artefacts (e.g. a person detected
# barely above the estimated horizon) and are reported as unavailable instead
# of showing a nonsense number.
CAMERA_HEIGHT_RANGE_M = (0.4, 15.0)

# Points this close to the horizon produce diverging, meaningless distances.
HORIZON_GUARD_PX = 8.0


def discover_videos(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def camera_intrinsics(width, height, frame=None):
    focal = None
    confidence = 0.2
    source = "dimension_fallback"

    if frame is not None:
        try:
            result = self_calibrate(frame)
            if getattr(result, "success", False) and getattr(result, "focal_length", None):
                focal = float(result.focal_length)
                confidence = float(min(1.0, getattr(result, "confidence", 0.3)))
                source = "vanishing_point_self_calibration"
        except Exception:
            focal = None

    if focal is None or focal <= 0:
        focal = float(max(width, height))

    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K, source, confidence


def detect_people(frame, model, confidence_threshold=0.5):
    results = model(frame, verbose=False)
    people = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) != 0:  # COCO class 0 = person
                continue
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            height_px = float(y2 - y1)
            if height_px < MIN_PERSON_HEIGHT_PX:
                continue
            people.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "bottom": [float((x1 + x2) / 2.0), float(y2)],
                "height_px": height_px,
            })

    return people


def plane_distance_from_pixel_offset(focal, height_m, vertical_offset_px):
    """d = f*h / v for a point on the ground plane, v = y - y_horizon."""
    if vertical_offset_px is None or vertical_offset_px <= 0:
        return None
    return float(focal * height_m / vertical_offset_px)


def analyze_camera(video_path, model, person_height_m=DEFAULT_PERSON_HEIGHT_M):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"camera": Path(video_path).name, "status": "unreadable_video"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if frame_count <= 0:
        return {"camera": Path(video_path).name, "status": "empty_video"}

    # Best-effort geometry frame: the sharp middle frame.
    frame_index = frame_count // 2
    frame = read_frame(video_path, frame_index)
    if frame is None:
        frame_index = 0
        frame = read_frame(video_path, 0)
    if frame is None:
        return {"camera": Path(video_path).name, "status": "unreadable_frame"}

    height_px, width_px = frame.shape[:2]

    # ---- Horizon / ground geometry (single view) --------------------------
    # Estimate on several frames and take the median. The vanishing-point
    # RANSAC is random, so each frame is analysed with a fixed seed: results
    # are deterministic across runs while the median dampens per-frame noise.
    horizon_candidates = []
    sample_indices = sorted({
        frame_count // 2,
        max(0, frame_count // 3),
        max(0, (2 * frame_count) // 3),
        max(0, frame_count // 5),
        max(0, (4 * frame_count) // 5),
    })

    for idx in sample_indices:
        geo_frame = read_frame(video_path, idx)
        if geo_frame is None:
            continue
        try:
            np.random.seed(1000 + int(idx))  # deterministic VP RANSAC per frame
            geo = analyze_ground_geometry(geo_frame)
            candidate = geo.get("horizon_y")
            if (
                geo.get("horizon_valid")
                and candidate is not None
                and 0.0 <= float(candidate) < height_px
            ):
                horizon_candidates.append(float(candidate))
        except Exception:
            continue

    # ---- Intrinsics (needed before horizon selection) ----------------------
    K, intrinsic_source, intrinsic_confidence = camera_intrinsics(width_px, height_px, frame)
    focal = float(K[0, 0])

    # ---- Metric scale cue: people -----------------------------------------
    people = []
    if model is not None:
        try:
            people = detect_people(frame, model)
        except Exception:
            people = []

    usable_people = [p for p in people if p["height_px"] >= MIN_SCALE_PERSON_PX]

    def _per_person_heights(horizon_candidate):
        """Implied camera height per detected person for a given horizon."""
        out = []
        for p in usable_people:
            feet_offset = float(p["bottom"][1]) - float(horizon_candidate)
            if feet_offset > 1.0:
                out.append(feet_offset * float(person_height_m) / p["height_px"])
        return out

    if horizon_candidates and len(usable_people) >= 2:
        # The horizon is chosen by cross-person consensus: every person
        # standing on the same ground must imply the SAME camera height, so
        # the correct horizon is the one where the implied heights agree.
        # The search covers the vanishing-point candidates plus a dense scan
        # of the plausible range, which rescues panning/translating footage
        # where the raw vanishing-point estimate is unstable.
        scan = list(np.arange(0.05 * height_px, 0.75 * height_px, 8.0))
        cands = sorted(set([float(c) for c in horizon_candidates] + [float(s) for s in scan]))
        best_h, best_score, best_median = None, None, None
        for cand in cands:
            hs = _per_person_heights(cand)
            if len(hs) < 2:
                continue
            med = float(np.median(hs))
            if not (CAMERA_HEIGHT_RANGE_M[0] <= med <= CAMERA_HEIGHT_RANGE_M[1]):
                continue
            deviations = np.abs(np.asarray(hs) - med)
            mad = float(np.median(deviations))
            score = 1.4826 * mad / med  # robust relative spread
            if best_score is None or score < best_score:
                best_h, best_score, best_median = float(cand), score, med
        if best_h is not None and best_score <= 0.5:
            horizon_y = best_h
            horizon_source = "person_consensus_horizon"
        else:
            horizon_y = float(np.median(horizon_candidates))
            horizon_source = "ground_plane_vanishing_points"
            spread = float(np.max(horizon_candidates) - np.min(horizon_candidates))
            if spread > 0.25 * height_px:
                horizon_source = "ground_plane_vanishing_points_high_spread"
    elif horizon_candidates:
        horizon_y = float(np.median(horizon_candidates))
        horizon_source = "ground_plane_vanishing_points"
        spread = float(np.max(horizon_candidates) - np.min(horizon_candidates))
        if spread > 0.25 * height_px:
            horizon_source = "ground_plane_vanishing_points_high_spread"
    else:
        # Conservative fallback: the horizon sits near the image centre.
        horizon_y = height_px * 0.5
        horizon_source = "image_centre_fallback"

    scale = None
    scale_source = None
    scale_confidence = 0.0
    scale_person = None
    if people:
        # Only nearby people (tall in the image) give a reliable prior; a
        # 60-px-tall person far away makes every derived metre meaningless.
        candidates = [p for p in people if p["height_px"] >= MIN_SCALE_PERSON_PX]
        if candidates:
            best = max(candidates, key=lambda p: p["height_px"])
            scale = float(person_height_m) / best["height_px"]
            scale_source = "person_height_prior"
            scale_confidence = 0.45
            scale_person = best

    # ---- Camera height (camera -> ground) ---------------------------------
    # Every detected person standing on the ground satisfies
    #   h = (y_feet - horizon) * H_person / person_height_px
    # so all of them must yield the SAME camera height. Taking the median
    # across persons is robust against single bad detections (cropped boxes,
    # people on steps, partially occluded feet).
    camera_height_m = None
    camera_height_status = "unavailable"
    height_note = None

    if scale is not None and scale_person is not None:
        if horizon_source != "image_centre_fallback":
            per_person = []
            for p in people:
                if p["height_px"] < MIN_SCALE_PERSON_PX:
                    continue
                feet_offset = float(p["bottom"][1]) - horizon_y
                if feet_offset > 1.0:
                    per_person.append(feet_offset * float(person_height_m) / p["height_px"])
            if per_person:
                camera_height_m = float(np.median(per_person))
                camera_height_status = "estimated_from_person_prior"
            else:
                camera_height_status = "degenerate_person_position"
                height_note = (
                    "The detected people's feet are at or above the horizon, so "
                    "no physical camera height can be derived."
                )
        else:
            camera_height_status = "needs_reliable_horizon"
            height_note = "Person scale found, but the image horizon was not reliable."
    else:
        camera_height_status = "needs_person_in_view"
        height_note = (
            "No upright person close enough to the camera was detected (persons "
            f"nearer than {MIN_SCALE_PERSON_PX} px in height are too far away for a "
            "reliable prior), so no metric cue was available for the "
            "camera-to-ground height."
        )

    raw_camera_height_m = camera_height_m
    if camera_height_m is not None and not (
        CAMERA_HEIGHT_RANGE_M[0] <= camera_height_m <= CAMERA_HEIGHT_RANGE_M[1]
    ):
        camera_height_m = None
        camera_height_status = "implausible_height_estimate"
        height_note = (
            "The person-prior estimate fell outside the physically plausible "
            f"mounting-height range {CAMERA_HEIGHT_RANGE_M[0]:.1f}\u2013{CAMERA_HEIGHT_RANGE_M[1]:.1f} m "
            f"(raw estimate {raw_camera_height_m:.2f} m \u2014 usually the detected "
            "person stands nearly at the horizon, which makes this method "
            "unstable). No camera-to-ground value is reported rather than a "
            "nonsense number."
        )

    # ---- Object distance ----------------------------------------------------
    # The object pixel position is supplied later by the UI; distances are
    # computed in compute_object_distances().

    return {
        "camera": Path(video_path).name,
        "status": "analyzed",
        "analyzed_frame_index": int(frame_index),
        "image_size": [int(width_px), int(height_px)],
        "horizon_y": horizon_y,
        "horizon_source": horizon_source,
        "intrinsics": {
            "focal_px": focal,
            "principal_point": [float(K[0, 2]), float(K[1, 2])],
            "source": intrinsic_source,
            "confidence": intrinsic_confidence,
        },
        "person_detections": [
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "height_px": p["height_px"],
            }
            for p in people
        ],
        "metric_scale": {
            "available": scale is not None,
            "m_per_px": scale,
            "at_person_depth": True,
            "source": scale_source,
            "confidence": scale_confidence,
            "prior_person_height_m": float(person_height_m),
        },
        "camera_to_ground": {
            "metric_height_m": camera_height_m,
            "raw_metric_height_m": raw_camera_height_m,
            "status": camera_height_status,
            "note": height_note,
            "method": "person_feet_offset_above_horizon_x_height_prior",
        },
    }


def compute_object_distances(camera_result, object_xy, person_height_m=DEFAULT_PERSON_HEIGHT_M):
    """
    Camera-to-object distance from the pinhole ground-plane projection.

    d = f * h / (y_object - y_horizon)

    This is the honest single-camera estimate; it assumes the selected point
    sits on (or near) the ground plane. Points within HORIZON_GUARD_PX of the
    horizon are rejected: the distance formula diverges there, so any value
    would be numerically meaningless.
    """
    if camera_result.get("status") != "analyzed":
        return None

    if object_xy is None:
        return None

    x, y = float(object_xy[0]), float(object_xy[1])
    horizon_y = camera_result["horizon_y"]
    focal = camera_result["intrinsics"]["focal_px"]
    scale = camera_result["metric_scale"]
    height_info = camera_result["camera_to_ground"]

    vertical_offset = float(y) - float(horizon_y)

    if vertical_offset <= HORIZON_GUARD_PX:
        return {
            "object_point": [x, y],
            "status": "at_or_above_horizon",
            "vertical_offset_px": vertical_offset,
            "note": (
                "Selected point is at/above (or within a few pixels of) the "
                "horizon; a ground-plane distance is undefined or unstable there."
            ),
        }

    if not scale.get("available") or not scale.get("m_per_px"):
        return {
            "object_point": [x, y],
            "status": "no_metric_scale",
            "vertical_offset_px": vertical_offset,
            "note": "No person-derived metric cue was available in this camera.",
        }

    h = height_info.get("metric_height_m")
    if h is None or h <= 0:
        return {
            "object_point": [x, y],
            "status": "no_camera_height",
            "vertical_offset_px": vertical_offset,
        }

    distance_m = plane_distance_from_pixel_offset(focal, h, vertical_offset)

    return {
        "object_point": [x, y],
        "status": "estimated",
        "vertical_offset_px": vertical_offset,
        "camera_height_m_used": float(h),
        "distance_m": float(distance_m),
        "method": "pinhole_ground_plane_projection_d_eq_fh_over_v",
        "note": (
            "Single-camera estimate assuming the selected point lies on the "
            "ground plane; conditional on the person-height prior."
        ),
    }


def run(input_folder, output_path, person_height_m=DEFAULT_PERSON_HEIGHT_M):
    videos = discover_videos(input_folder)
    model = None

    if YOLO is not None:
        try:
            model = YOLO("yolo11n.pt")
        except Exception:
            model = None

    cameras = []
    for video in videos:
        print(f"Analyzing camera: {video.name}")
        result = analyze_camera(video, model, person_height_m)
        cameras.append(result)

        height = result.get("camera_to_ground", {})
        if height.get("metric_height_m") is not None:
            print(f"  camera->ground (height): {height['metric_height_m']:.3f} m "
                  f"({height['status']})")
        else:
            print(f"  camera->ground: unavailable ({height.get('status')})")

    report = {
        "stage": "visual_metric_layer",
        "input_folder": str(Path(input_folder).resolve()),
        "camera_count": len(videos),
        "cameras": cameras,
        "person_height_prior_m": float(person_height_m),
        "person_height_sigma_m": PERSON_HEIGHT_SIGMA_M,
        "notes": [
            "Camera-to-ground and camera-to-object distances are single-view estimates.",
            "The metric cue is the statistical human-height prior; values are estimates.",
            "Multi-view triangulation, when valid, is the more accurate geometric source.",
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {output}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Per-camera visual metric layer")
    parser.add_argument("input_folder")
    parser.add_argument("--output", default="results/visual_metric_layer.json")
    parser.add_argument("--person-height", type=float, default=DEFAULT_PERSON_HEIGHT_M)
    args = parser.parse_args()

    run(args.input_folder, args.output, args.person_height)


if __name__ == "__main__":
    main()
