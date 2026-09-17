import cv2
import numpy as np
import os
import math


# ============================================================
# BASIC GEOMETRY
# ============================================================

def line_intersection(line1, line2):
    """
    Find the intersection point of two image lines.

    Each line is represented as:
        (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-8:
        return None

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator

    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return float(px), float(py)


def line_angle(line):
    """
    Return line orientation in degrees.
    """

    x1, y1, x2, y2 = line

    angle = math.degrees(
        math.atan2(
            y2 - y1,
            x2 - x1
        )
    )

    # Convert to [0, 180)
    angle = angle % 180

    return angle


def line_length(line):
    """
    Return Euclidean length of a line segment.
    """

    x1, y1, x2, y2 = line

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ============================================================
# LINE DETECTION
# ============================================================

def detect_lines(
    frame,
    min_line_length=40
):
    """
    Detect strong line segments using Canny + HoughLinesP.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur reduces noisy edges.
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
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=20
    )

    if lines is None:
        return []

    result = []

    for item in lines:

        x1, y1, x2, y2 = item[0]

        line = (
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        )

        length = line_length(line)

        if length >= min_line_length:
            result.append(line)

    return result


# ============================================================
# LINE GROUPING
# ============================================================

def angular_difference(a, b):
    """
    Difference between two undirected line angles.
    """

    difference = abs(a - b)

    return min(
        difference,
        180 - difference
    )


def group_lines_by_direction(
    lines,
    angle_tolerance=12
):
    """
    Group lines that have similar orientations.
    """

    groups = []

    for line in lines:

        angle = line_angle(line)

        placed = False

        for group in groups:

            group_angle = group["angle"]

            if angular_difference(
                angle,
                group_angle
            ) <= angle_tolerance:

                group["lines"].append(line)

                # Weighted average angle.
                lengths = [
                    line_length(x)
                    for x in group["lines"]
                ]

                angles = [
                    line_angle(x)
                    for x in group["lines"]
                ]

                group["angle"] = np.average(
                    angles,
                    weights=lengths
                )

                placed = True
                break

        if not placed:

            groups.append(
                {
                    "angle": angle,
                    "lines": [line]
                }
            )

    groups.sort(
        key=lambda g: sum(
            line_length(x)
            for x in g["lines"]
        ),
        reverse=True
    )

    return groups


# ============================================================
# VANISHING POINT ESTIMATION
# ============================================================

def estimate_vanishing_point(
    lines,
    frame_width,
    frame_height
):
    """
    Estimate a vanishing point from a group of lines.

    Uses intersections of many line pairs and keeps
    geometrically consistent intersections.
    """

    if len(lines) < 2:
        return None

    intersections = []

    # Limit pair count for performance.
    max_lines = min(
        len(lines),
        30
    )

    selected = sorted(
        lines,
        key=line_length,
        reverse=True
    )[:max_lines]

    for i in range(len(selected)):

        for j in range(i + 1, len(selected)):

            intersection = line_intersection(
                selected[i],
                selected[j]
            )

            if intersection is None:
                continue

            x, y = intersection

            # Ignore numerically unstable intersections.
            if not (
                math.isfinite(x)
                and math.isfinite(y)
            ):
                continue

            # Extremely distant intersections are usually
            # caused by nearly parallel image lines.
            limit_x = frame_width * 10
            limit_y = frame_height * 10

            if abs(x) > limit_x:
                continue

            if abs(y) > limit_y:
                continue

            intersections.append(
                (x, y)
            )

    if not intersections:
        return None

    points = np.array(
        intersections,
        dtype=np.float64
    )

    # Robust median estimate.
    vp_x = np.median(
        points[:, 0]
    )

    vp_y = np.median(
        points[:, 1]
    )

    return (
        float(vp_x),
        float(vp_y)
    )


# ============================================================
# FRAME ANALYSIS
# ============================================================

def analyze_frame(frame):
    """
    Analyze one CCTV frame.

    Returns:
        detected lines
        line groups
        candidate vanishing points
    """

    height, width = frame.shape[:2]

    lines = detect_lines(frame)

    groups = group_lines_by_direction(
        lines
    )

    candidates = []

    # Strongest directional groups.
    for group in groups[:6]:

        if len(group["lines"]) < 2:
            continue

        vp = estimate_vanishing_point(
            group["lines"],
            width,
            height
        )

        if vp is not None:

            candidates.append(
                {
                    "angle": group["angle"],
                    "line_count": len(
                        group["lines"]
                    ),
                    "vanishing_point": vp
                }
            )

    return {
        "lines": lines,
        "groups": groups,
        "candidates": candidates
    }


# ============================================================
# MULTI-FRAME ANALYSIS
# ============================================================

def analyze_video(
    video_path,
    sample_count=8
):
    """
    Analyze several frames from a CCTV video.

    The final geometry is based on multiple frames rather
    than one arbitrary frame.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            "Video contains no readable frames."
        )

    # Evenly distribute samples through video.
    sample_indices = np.linspace(
        0,
        total_frames - 1,
        min(
            sample_count,
            total_frames
        ),
        dtype=int
    )

    frame_results = []

    for frame_number in sample_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = cap.read()

        if not success:
            continue

        result = analyze_frame(
            frame
        )

        result["frame_number"] = int(
            frame_number
        )

        frame_results.append(
            result
        )

    cap.release()

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "frames_analyzed": len(
            frame_results
        ),
        "results": frame_results
    }


# ============================================================
# VISUALIZATION
# ============================================================

def draw_analysis(
    frame,
    analysis
):
    """
    Draw detected lines and candidate vanishing points.
    """

    output = frame.copy()

    lines = analysis.get(
        "lines",
        []
    )

    for line in lines:

        x1, y1, x2, y2 = line

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1
        )

    candidates = analysis.get(
        "candidates",
        []
    )

    for candidate in candidates:

        x, y = candidate[
            "vanishing_point"
        ]

        x = int(round(x))
        y = int(round(y))

        # Only draw if reasonably inside/near frame.
        if (
            -output.shape[1] <= x <=
            output.shape[1] * 2
            and
            -output.shape[0] <= y <=
            output.shape[0] * 2
        ):

            cv2.circle(
                output,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

    return output


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    print("=" * 60)
    print("AUTOMATIC CCTV SCENE GEOMETRY")
    print("=" * 60)

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            r"py src\vanishing_points.py "
            r'"videos\camera1\video.mp4"'
        )
        print()
        print(
            "This module analyzes multiple frames and "
            "estimates perspective geometry."
        )

        sys.exit(0)

    video_path = sys.argv[1]

    try:

        result = analyze_video(
            video_path,
            sample_count=8
        )

        print()
        print(
            f"Video: {result['video_path']}"
        )

        print(
            f"Frames: {result['total_frames']}"
        )

        print(
            f"FPS: {result['fps']:.2f}"
        )

        print(
            f"Frames analyzed: "
            f"{result['frames_analyzed']}"
        )

        print()

        for frame_result in result[
            "results"
        ]:

            print(
                f"Frame "
                f"{frame_result['frame_number']}: "
                f"{len(frame_result['lines'])} lines, "
                f"{len(frame_result['candidates'])} "
                f"candidate VPs"
            )

        print()
        print("Scene geometry analysis complete.")

    except Exception as e:

        print()
        print("ERROR:")
        print(str(e))