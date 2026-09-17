import cv2
import numpy as np
import os


def detect_edge_lines(frame):
    """
    Detect strong structural lines in a CCTV frame.
    """

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
        threshold=50,
        minLineLength=max(
            30,
            frame.shape[1] // 12
        ),
        maxLineGap=25
    )

    if lines is None:
        return []

    return [
        tuple(line[0])
        for line in lines
    ]


def line_to_homogeneous(line):
    """
    Convert an image line segment to a homogeneous line.
    """

    x1, y1, x2, y2 = line

    p1 = np.array(
        [x1, y1, 1.0],
        dtype=np.float64
    )

    p2 = np.array(
        [x2, y2, 1.0],
        dtype=np.float64
    )

    return np.cross(p1, p2)


def intersection_of_lines(line1, line2):
    """
    Calculate intersection of two homogeneous lines.
    """

    l1 = line_to_homogeneous(line1)
    l2 = line_to_homogeneous(line2)

    point = np.cross(l1, l2)

    if abs(point[2]) < 1e-9:
        return None

    return (
        point[0] / point[2],
        point[1] / point[2]
    )


def line_angle(line):
    """
    Return orientation of a line in degrees.
    """

    x1, y1, x2, y2 = line

    angle = np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1
        )
    )

    return angle % 180


def line_length(line):
    """
    Return line length.
    """

    x1, y1, x2, y2 = line

    return float(
        np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )
    )


def select_ground_lines(
    lines,
    frame_height
):
    """
    Select lines that are more likely to belong
    to the visible ground plane.

    This is deliberately generic. It does not assume
    tiles, airport architecture, or a particular scene.
    """

    selected = []

    for line in lines:

        x1, y1, x2, y2 = line

        length = line_length(line)

        if length < 40:
            continue

        midpoint_y = (
            y1 + y2
        ) / 2

        # Ground-plane structures are usually visible
        # in the lower/middle portion of surveillance views.
        if midpoint_y < frame_height * 0.25:
            continue

        selected.append(line)

    return selected


def estimate_horizon(
    frame,
    lines
):
    """
    Estimate the ground-plane horizon.

    We search for a dominant vanishing region created
    by structural lines on the ground plane.

    Returns:
        horizon_y
        candidate_points
    """

    height, width = frame.shape[:2]

    ground_lines = select_ground_lines(
        lines,
        height
    )

    if len(ground_lines) < 2:
        return None, []

    intersections = []

    for i in range(
        len(ground_lines)
    ):

        for j in range(
            i + 1,
            len(ground_lines)
        ):

            line_a = ground_lines[i]
            line_b = ground_lines[j]

            angle_a = line_angle(line_a)
            angle_b = line_angle(line_b)

            # Nearly parallel lines do not provide
            # a stable intersection.
            angle_difference = abs(
                angle_a - angle_b
            )

            angle_difference = min(
                angle_difference,
                180 - angle_difference
            )

            if angle_difference < 8:
                continue

            point = intersection_of_lines(
                line_a,
                line_b
            )

            if point is None:
                continue

            x, y = point

            # Keep candidate vanishing points reasonably
            # close to the image.
            if (
                abs(x) > width * 5
                or abs(y) > height * 5
            ):
                continue

            intersections.append(
                (x, y)
            )

    if not intersections:
        return None, []

    points = np.asarray(
        intersections,
        dtype=np.float64
    )

    # Robust median of candidate intersections.
    median_x = np.median(
        points[:, 0]
    )

    median_y = np.median(
        points[:, 1]
    )

    # The horizon is primarily represented by its
    # vertical position for our first stage.
    horizon_y = float(
        median_y
    )

    return horizon_y, intersections


def analyze_frame(frame):
    """
    Analyze one frame for ground geometry.
    """

    lines = detect_edge_lines(
        frame
    )

    horizon_y, intersections = (
        estimate_horizon(
            frame,
            lines
        )
    )

    return {
        "line_count": len(lines),
        "lines": lines,
        "horizon_y": horizon_y,
        "intersections": intersections
    }


def analyze_video(
    video_path,
    sample_count=10
):
    """
    Analyze multiple frames and combine their
    horizon estimates.

    Multiple frames reduce the influence of
    temporary people and other moving objects.
    """

    if not os.path.exists(
        video_path
    ):
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

    if total_frames <= 0:
        cap.release()

        raise RuntimeError(
            "Video contains no readable frames."
        )

    frame_numbers = np.linspace(
        0,
        total_frames - 1,
        min(
            sample_count,
            total_frames
        ),
        dtype=int
    )

    estimates = []
    frame_results = []

    for frame_number in frame_numbers:

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

        result[
            "frame_number"
        ] = int(frame_number)

        frame_results.append(
            result
        )

        if result[
            "horizon_y"
        ] is not None:

            estimates.append(
                result["horizon_y"]
            )

    cap.release()

    if estimates:

        final_horizon = float(
            np.median(
                estimates
            )
        )

    else:

        final_horizon = None

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "frames_analyzed": len(
            frame_results
        ),
        "horizon_y": final_horizon,
        "horizon_estimates": estimates,
        "frames": frame_results
    }


def draw_horizon(
    frame,
    horizon_y
):
    """
    Draw estimated horizon on a frame.
    """

    output = frame.copy()

    if horizon_y is None:
        return output

    y = int(
        round(horizon_y)
    )

    if 0 <= y < output.shape[0]:

        cv2.line(
            output,
            (0, y),
            (
                output.shape[1],
                y
            ),
            (255, 0, 0),
            2
        )

    return output


if __name__ == "__main__":

    import sys

    print("=" * 60)
    print("GROUND-PLANE GEOMETRY ANALYSIS")
    print("=" * 60)

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            r'py src\ground_geometry.py '
            r'"path\to\video.mp4"'
        )
        print()

        sys.exit(0)

    video_path = sys.argv[1]

    try:

        result = analyze_video(
            video_path
        )

        print()
        print(
            f"Video: {result['video_path']}"
        )

        print(
            f"Frames analyzed: "
            f"{result['frames_analyzed']}"
        )

        print(
            f"Horizon Y: "
            f"{result['horizon_y']}"
        )

        print(
            f"Valid horizon estimates: "
            f"{len(result['horizon_estimates'])}"
        )

        print()
        print(
            "Ground geometry analysis complete."
        )

    except Exception as error:

        print()
        print("ERROR:")
        print(str(error))