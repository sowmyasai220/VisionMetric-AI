"""
Generic CCTV Visual Geometry Analyzer
--------------------------------------

Input:
    A folder containing CCTV videos.

The module automatically:
    - discovers video files
    - reads their dimensions
    - samples frames
    - detects strong line structures
    - looks for repeated parallel structures
    - reports geometric scale cues

NO environment-specific dimensions are assumed.

IMPORTANT:
Geometric repetition does NOT automatically provide metres/feet.
Absolute metric scale requires a trustworthy physical scale cue.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import math


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".m4v"
}


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class GeometryResult:
    video: str
    frames_analyzed: int
    lines_detected: int
    dominant_angles: List[float]
    repeated_structure: bool
    spacing_pixels: List[float]


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def discover_videos(folder):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}"
        )

    videos = []

    for path in folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(path):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        fps = 1.0

    duration = frame_count / fps

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration
    )


# ============================================================
# FRAME SAMPLING
# ============================================================

def sample_frames(
    path,
    max_frames=12
):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return []

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        cap.release()
        return []

    count = min(
        max_frames,
        total
    )

    positions = np.linspace(
        0,
        total - 1,
        count,
        dtype=int
    )

    frames = []

    for position in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position)
        )

        ok, frame = cap.read()

        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    return frames


# ============================================================
# LINE DETECTION
# ============================================================

def detect_lines(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=45,
        minLineLength=max(
            30,
            int(frame.shape[1] * 0.08)
        ),
        maxLineGap=25
    )

    if lines is None:
        return []

    output = []

    for item in lines:

        x1, y1, x2, y2 = item[0]

        length = math.hypot(
            x2 - x1,
            y2 - y1
        )

        if length < 30:
            continue

        angle = math.degrees(
            math.atan2(
                y2 - y1,
                x2 - x1
            )
        ) % 180

        output.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "angle": angle,
                "length": length
            }
        )

    return output


# ============================================================
# DOMINANT DIRECTIONS
# ============================================================

def dominant_directions(
    lines,
    tolerance=6
):

    if not lines:
        return []

    angles = [
        line["angle"]
        for line in lines
    ]

    groups = []

    for angle in angles:

        found = False

        for group in groups:

            difference = abs(
                angle - group["mean"]
            )

            difference = min(
                difference,
                180 - difference
            )

            if difference <= tolerance:

                group["angles"].append(angle)

                group["mean"] = (
                    sum(group["angles"])
                    / len(group["angles"])
                )

                found = True
                break

        if not found:

            groups.append(
                {
                    "mean": angle,
                    "angles": [angle]
                }
            )

    groups.sort(
        key=lambda g: len(g["angles"]),
        reverse=True
    )

    return [
        round(group["mean"], 2)
        for group in groups[:4]
        if len(group["angles"]) >= 3
    ]


# ============================================================
# REPEATED PARALLEL STRUCTURES
# ============================================================

def line_position(line, angle):

    theta = math.radians(
        angle + 90
    )

    axis_x = math.cos(theta)
    axis_y = math.sin(theta)

    midpoint_x = (
        line["x1"] +
        line["x2"]
    ) / 2

    midpoint_y = (
        line["y1"] +
        line["y2"]
    ) / 2

    return (
        midpoint_x * axis_x +
        midpoint_y * axis_y
    )


def find_repeated_spacing(
    lines,
    angle
):

    selected = []

    for line in lines:

        difference = abs(
            line["angle"] - angle
        )

        difference = min(
            difference,
            180 - difference
        )

        if difference <= 6:
            selected.append(line)

    if len(selected) < 3:
        return []

    positions = sorted(
        line_position(line, angle)
        for line in selected
    )

    differences = []

    for i in range(
        len(positions) - 1
    ):

        d = abs(
            positions[i + 1] -
            positions[i]
        )

        if d > 8:
            differences.append(d)

    if len(differences) < 2:
        return []

    candidates = []

    for d in differences:

        matches = [
            value
            for value in differences
            if abs(value - d)
            <= max(12, d * 0.20)
        ]

        if len(matches) >= 2:

            candidates.append(
                np.median(matches)
            )

    if not candidates:
        return []

    return [
        round(float(x), 2)
        for x in candidates[:3]
    ]


# ============================================================
# ANALYZE ONE VIDEO
# ============================================================

def analyze_video(
    path,
    max_frames=12
):

    frames = sample_frames(
        path,
        max_frames=max_frames
    )

    all_angles = []
    all_spacing = []
    total_lines = 0

    for frame in frames:

        lines = detect_lines(frame)

        total_lines += len(lines)

        directions = dominant_directions(
            lines
        )

        all_angles.extend(
            directions
        )

        for angle in directions:

            spacing = find_repeated_spacing(
                lines,
                angle
            )

            all_spacing.extend(
                spacing
            )

    # Consolidate angle values.
    dominant = []

    if all_angles:

        histogram = {}

        for angle in all_angles:

            bucket = round(
                angle / 5
            ) * 5

            histogram[bucket] = (
                histogram.get(bucket, 0) + 1
            )

        dominant = [
            float(angle)
            for angle, _ in sorted(
                histogram.items(),
                key=lambda x: x[1],
                reverse=True
            )[:4]
        ]

    repeated = len(all_spacing) > 0

    return GeometryResult(
        video=str(path),
        frames_analyzed=len(frames),
        lines_detected=total_lines,
        dominant_angles=dominant,
        repeated_structure=repeated,
        spacing_pixels=all_spacing[:10]
    )


# ============================================================
# ANALYZE INPUT FOLDER
# ============================================================

def analyze_folder(
    folder
):

    videos = discover_videos(
        folder
    )

    print()
    print("=" * 70)
    print("GENERIC CCTV GEOMETRY ANALYZER")
    print("=" * 70)

    print(
        f"Input folder : {Path(folder).resolve()}"
    )

    print(
        f"Videos found : {len(videos)}"
    )

    if not videos:

        print()
        print(
            "No supported video files found."
        )

        print(
            "Supported formats:"
        )

        print(
            ", ".join(
                sorted(VIDEO_EXTENSIONS)
            )
        )

        return []

    results = []

    for index, video in enumerate(
        videos,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(videos)}] "
            f"{video.name}"
        )

        info = get_video_info(
            video
        )

        print(
            f"  Resolution : "
            f"{info.width} x {info.height}"
        )

        print(
            f"  FPS        : "
            f"{info.fps:.2f}"
        )

        print(
            f"  Frames     : "
            f"{info.frame_count}"
        )

        print(
            f"  Duration   : "
            f"{info.duration_seconds:.2f} sec"
        )

        result = analyze_video(
            video
        )

        results.append(result)

        print(
            f"  Frames analyzed : "
            f"{result.frames_analyzed}"
        )

        print(
            f"  Lines detected  : "
            f"{result.lines_detected}"
        )

        print(
            f"  Dominant angles : "
            f"{result.dominant_angles}"
        )

        print(
            f"  Repeated geometry: "
            f"{result.repeated_structure}"
        )

        if result.spacing_pixels:

            print(
                f"  Candidate spacing: "
                f"{result.spacing_pixels}"
            )

    print()
    print("=" * 70)
    print("SCALE CONCLUSION")
    print("=" * 70)

    repeated_count = sum(
        result.repeated_structure
        for result in results
    )

    print(
        f"Videos with repeated geometry: "
        f"{repeated_count}/{len(results)}"
    )

    print()
    print(
        "No physical unit has been assumed."
    )

    print(
        "Detected spacing is currently expressed "
        "only in image pixels."
    )

    print(
        "These geometric cues will be used by the "
        "projective geometry stage."
    )

    print("=" * 70)

    return results


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "py src\\scale_cue_detector.py videos"
        )

        print()

        sys.exit(1)

    input_folder = sys.argv[1]

    analyze_folder(
        input_folder
    )