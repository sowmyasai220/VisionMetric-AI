"""
Automatic CCTV Geometry Pipeline
---------------------------------

Environment-independent pipeline for:
1. Discovering CCTV videos
2. Sampling frames automatically
3. Selecting geometrically useful frames
4. Running vanishing-point analysis
5. Running ground geometry analysis
6. Running self-calibration
7. Aggregating results per video

No environment-specific measurements are assumed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

try:
    from .self_calibration import self_calibrate
except ImportError:
    from self_calibration import self_calibrate

try:
    from .vanishing_points import analyze_frame as analyze_vanishing_frame
except ImportError:
    analyze_vanishing_frame = None

try:
    from .ground_geometry import analyze_frame as analyze_ground_frame
except ImportError:
    analyze_ground_frame = None


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v",
}


@dataclass
class FrameCandidate:
    frame_index: int
    timestamp_seconds: float
    score: float
    sharpness: float
    brightness: float
    edge_density: float
    line_density: float


@dataclass
class VideoGeometryResult:
    video_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    sampled_frames: int
    selected_frames: List[Dict[str, Any]]

    calibration_successes: int
    focal_length_estimates: List[float]

    vanishing_point_successes: int
    ground_geometry_successes: int

    horizon_estimates: List[float]

    notes: List[str]


# ---------------------------------------------------------------------
# VIDEO DISCOVERY
# ---------------------------------------------------------------------

def discover_videos(input_folder: str) -> List[Path]:
    """
    Recursively discover supported video files.
    """
    folder = Path(input_folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}"
        )

    videos = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


# ---------------------------------------------------------------------
# VIDEO INFORMATION
# ---------------------------------------------------------------------

def get_video_info(video_path: Path) -> Dict[str, Any]:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0:
        fps = 1.0

    duration = frame_count / fps

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


# ---------------------------------------------------------------------
# FRAME QUALITY
# ---------------------------------------------------------------------

def sharpness_score(gray: np.ndarray) -> float:
    """
    Variance of Laplacian.
    Higher generally means sharper imagery.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(gray: np.ndarray) -> float:
    """
    Score around a useful mid-range brightness.
    """
    mean_value = float(np.mean(gray))

    # Maximum score near the middle of the intensity range.
    score = 1.0 - abs(mean_value - 128.0) / 128.0

    return max(0.0, min(1.0, score))


def edge_density_score(gray: np.ndarray) -> float:
    """
    Percentage of pixels belonging to Canny edges.
    """
    edges = cv2.Canny(gray, 50, 150)

    density = float(np.count_nonzero(edges)) / float(edges.size)

    return density


def line_density_score(gray: np.ndarray) -> float:
    """
    Detect structural lines using HoughLinesP.
    """
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=max(20, min(gray.shape) // 12),
        maxLineGap=10,
    )

    if lines is None:
        return 0.0

    return min(1.0, len(lines) / 100.0)


def score_frame(frame: np.ndarray) -> FrameCandidate:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharp = sharpness_score(gray)
    bright = brightness_score(gray)
    edges = edge_density_score(gray)
    lines = line_density_score(gray)

    # Log compression prevents very sharp frames from dominating
    # everything else.
    sharp_normalized = math.log1p(sharp) / math.log1p(1000.0)
    sharp_normalized = max(
        0.0,
        min(1.0, sharp_normalized)
    )

    edge_normalized = max(
        0.0,
        min(1.0, edges * 10.0)
    )

    score = (
        0.40 * sharp_normalized
        + 0.20 * bright
        + 0.20 * edge_normalized
        + 0.20 * lines
    )

    return FrameCandidate(
        frame_index=-1,
        timestamp_seconds=0.0,
        score=float(score),
        sharpness=sharp,
        brightness=bright,
        edge_density=edges,
        line_density=lines,
    )


# ---------------------------------------------------------------------
# FRAME SAMPLING
# ---------------------------------------------------------------------

def sample_video_frames(
    video_path: Path,
    candidate_count: int = 12,
) -> List[FrameCandidate]:

    info = get_video_info(video_path)

    frame_count = info["frame_count"]
    fps = info["fps"]

    if frame_count <= 0:
        return []

    candidate_count = min(
        candidate_count,
        frame_count
    )

    # Uniform sampling across the complete video.
    indices = np.linspace(
        0,
        frame_count - 1,
        candidate_count,
        dtype=int,
    )

    cap = cv2.VideoCapture(str(video_path))

    candidates = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success or frame is None:
            continue

        candidate = score_frame(frame)

        candidate.frame_index = int(index)
        candidate.timestamp_seconds = float(index / fps)

        candidates.append(candidate)

    cap.release()

    return candidates


# ---------------------------------------------------------------------
# FRAME SELECTION
# ---------------------------------------------------------------------

def select_best_frames(
    candidates: List[FrameCandidate],
    keep_count: int = 4,
) -> List[FrameCandidate]:

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: x.score,
        reverse=True,
    )

    # Avoid selecting frames that are immediately adjacent.
    selected = []

    for candidate in candidates:

        too_close = False

        for existing in selected:
            if abs(
                candidate.frame_index
                - existing.frame_index
            ) < 5:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= keep_count:
            break

    return selected


# ---------------------------------------------------------------------
# READ SELECTED FRAME
# ---------------------------------------------------------------------

def read_frame(
    video_path: Path,
    frame_index: int,
) -> Optional[np.ndarray]:

    cap = cv2.VideoCapture(str(video_path))

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index)
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


# ---------------------------------------------------------------------
# SAFE RESULT HELPERS
# ---------------------------------------------------------------------

def extract_focal_length(result: Any) -> Optional[float]:

    if result is None:
        return None

    if isinstance(result, dict):

        for key in (
            "focal_length",
            "focal_length_estimate",
            "focal",
        ):
            value = result.get(key)

            if isinstance(value, (int, float)):
                return float(value)

    for key in (
        "focal_length",
        "focal_length_estimate",
        "focal",
    ):
        if hasattr(result, key):

            value = getattr(result, key)

            if isinstance(value, (int, float)):
                return float(value)

    return None


def extract_horizon(result: Any) -> Optional[float]:

    if result is None:
        return None

    if isinstance(result, dict):

        for key in (
            "horizon_y",
            "horizon",
        ):
            value = result.get(key)

            if isinstance(value, (int, float)):
                return float(value)

    for key in (
        "horizon_y",
        "horizon",
    ):
        if hasattr(result, key):

            value = getattr(result, key)

            if isinstance(value, (int, float)):
                return float(value)

    return None


# ---------------------------------------------------------------------
# GEOMETRY ANALYSIS
# ---------------------------------------------------------------------

def analyze_selected_frame(
    frame: np.ndarray,
) -> Dict[str, Any]:

    result = {
        "calibration": None,
        "vanishing_points": None,
        "ground_geometry": None,
    }

    # -------------------------------------------------------------
    # SELF CALIBRATION
    # -------------------------------------------------------------

    try:

        calibration = self_calibrate(frame)

        result["calibration"] = calibration

    except Exception as exc:

        result["calibration"] = {
            "error": str(exc)
        }

    # -------------------------------------------------------------
    # VANISHING POINTS
    # -------------------------------------------------------------

    if analyze_vanishing_frame is not None:

        try:

            result["vanishing_points"] = (
                analyze_vanishing_frame(frame)
            )

        except Exception as exc:

            result["vanishing_points"] = {
                "error": str(exc)
            }

    # -------------------------------------------------------------
    # GROUND GEOMETRY
    # -------------------------------------------------------------

    if analyze_ground_frame is not None:

        try:

            result["ground_geometry"] = (
                analyze_ground_frame(frame)
            )

        except Exception as exc:

            result["ground_geometry"] = {
                "error": str(exc)
            }

    return result


# ---------------------------------------------------------------------
# VIDEO ANALYSIS
# ---------------------------------------------------------------------

def analyze_video(
    video_path: Path,
    candidate_count: int = 12,
    keep_count: int = 4,
) -> VideoGeometryResult:

    info = get_video_info(video_path)

    candidates = sample_video_frames(
        video_path,
        candidate_count=candidate_count,
    )

    selected = select_best_frames(
        candidates,
        keep_count=keep_count,
    )

    focal_estimates = []
    horizon_estimates = []

    calibration_successes = 0
    vp_successes = 0
    ground_successes = 0

    selected_results = []

    for candidate in selected:

        frame = read_frame(
            video_path,
            candidate.frame_index,
        )

        if frame is None:
            continue

        analysis = analyze_selected_frame(frame)

        calibration = analysis.get("calibration")

        focal = extract_focal_length(calibration)

        if focal is not None and focal > 0:
            focal_estimates.append(focal)
            calibration_successes += 1

        if analysis.get("vanishing_points") is not None:
            if "error" not in (
                analysis["vanishing_points"]
                if isinstance(
                    analysis["vanishing_points"],
                    dict
                )
                else {}
            ):
                vp_successes += 1

        ground = analysis.get("ground_geometry")

        horizon = extract_horizon(ground)

        if horizon is not None:
            horizon_estimates.append(horizon)
            ground_successes += 1

        selected_results.append({
            "frame_index": candidate.frame_index,
            "timestamp_seconds": candidate.timestamp_seconds,
            "quality_score": candidate.score,
            "sharpness": candidate.sharpness,
            "brightness": candidate.brightness,
            "edge_density": candidate.edge_density,
            "line_density": candidate.line_density,
            "focal_length": focal,
            "horizon_y": horizon,
        })

    notes = []

    if not selected:
        notes.append(
            "No usable frames were selected."
        )

    if not focal_estimates:
        notes.append(
            "No reliable focal-length estimate was obtained."
        )

    if not horizon_estimates:
        notes.append(
            "No reliable ground horizon estimate was obtained."
        )

    notes.append(
        "Absolute metric scale is not assumed by this pipeline."
    )

    return VideoGeometryResult(
        video_path=str(video_path),
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        frame_count=info["frame_count"],
        duration_seconds=info["duration_seconds"],
        sampled_frames=len(candidates),
        selected_frames=selected_results,
        calibration_successes=calibration_successes,
        focal_length_estimates=focal_estimates,
        vanishing_point_successes=vp_successes,
        ground_geometry_successes=ground_successes,
        horizon_estimates=horizon_estimates,
        notes=notes,
    )


# ---------------------------------------------------------------------
# FOLDER ANALYSIS
# ---------------------------------------------------------------------

def analyze_folder(
    input_folder: str,
    output_file: Optional[str] = None,
    candidate_count: int = 12,
    keep_count: int = 4,
) -> Dict[str, Any]:

    videos = discover_videos(input_folder)

    results = []

    for video in videos:

        print()
        print("=" * 60)
        print(f"Analyzing: {video.name}")
        print("=" * 60)

        try:

            result = analyze_video(
                video,
                candidate_count=candidate_count,
                keep_count=keep_count,
            )

            results.append(asdict(result))

            print(
                f"Sampled frames : {result.sampled_frames}"
            )

            print(
                f"Selected frames: {len(result.selected_frames)}"
            )

            print(
                f"Calibration OK : {result.calibration_successes}"
            )

            print(
                f"VP analysis OK : {result.vanishing_point_successes}"
            )

            print(
                f"Ground geometry: {result.ground_geometry_successes}"
            )

        except Exception as exc:

            results.append({
                "video_path": str(video),
                "error": str(exc),
            })

            print(f"ERROR: {exc}")

    output = {
        "input_folder": str(Path(input_folder).resolve()),
        "video_count": len(videos),
        "videos": results,
        "pipeline_notes": [
            "Input videos are discovered automatically.",
            "Frames are sampled automatically.",
            "Useful frames are selected automatically.",
            "No environment-specific metric assumptions are used.",
            "Metric scale remains a separate problem.",
        ],
    }

    if output_file is not None:

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
            )

    return output


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatic environment-independent "
            "CCTV geometry pipeline"
        )
    )

    parser.add_argument(
        "input_folder",
        help="Folder containing CCTV videos",
    )

    parser.add_argument(
        "--output",
        default="results/geometry_pipeline.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=12,
        help="Number of frames sampled per video",
    )

    parser.add_argument(
        "--keep",
        type=int,
        default=4,
        help="Number of best frames retained per video",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATIC CCTV GEOMETRY PIPELINE")
    print("=" * 60)

    print()
    print("Input folder:", args.input_folder)
    print("Output file :", args.output)
    print()

    output = analyze_folder(
        args.input_folder,
        output_file=args.output,
        candidate_count=args.candidates,
        keep_count=args.keep,
    )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"Videos discovered: {output['video_count']}"
    )

    print(
        f"Results saved to: {args.output}"
    )

    print()
    print(
        "Next stage: robust camera/scene geometry "
        "and common-object correspondence."
    )


if __name__ == "__main__":
    main()