
import cv2
import numpy as np
import json
from pathlib import Path
import sys
import math


def line_seg(x):
    x = np.asarray(x).reshape(-1)

    if len(x) < 4:
        return None

    x1, y1, x2, y2 = map(float, x[:4])
    return (x1, y1, x2, y2)


def length(x):
    x1, y1, x2, y2 = x
    return math.hypot(x2 - x1, y2 - y1)


def angle(x):
    x1, y1, x2, y2 = x
    return math.atan2(y2 - y1, x2 - x1)


def detect_segments(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=50,
        maxLineGap=20
    )

    if lines is None:
        return []

    segments = []

    for item in lines:
        seg = line_seg(item)

        if seg is not None and length(seg) >= 50:
            segments.append(seg)

    return segments


def homogeneous_line(seg):
    x1, y1, x2, y2 = seg

    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])

    return np.cross(p1, p2)


def intersection(l1, l2):
    p = np.cross(l1, l2)

    if abs(p[2]) < 1e-9:
        return None

    return p / p[2]


def ransac_vp(lines, iterations=500, threshold=5):
    if len(lines) < 2:
        return None

    best_point = None
    best_count = 0

    homogeneous = [
        homogeneous_line(x)
        for x in lines
    ]

    for _ in range(iterations):

        a, b = np.random.choice(
            len(homogeneous),
            2,
            replace=False
        )

        p = np.cross(
            homogeneous[a],
            homogeneous[b]
        )

        if abs(p[2]) < 1e-9:
            continue

        p = p / p[2]

        count = 0

        for line in homogeneous:

            denominator = math.sqrt(
                line[0] ** 2 +
                line[1] ** 2
            )

            if denominator < 1e-9:
                continue

            distance = abs(
                line[0] * p[0] +
                line[1] * p[1] +
                line[2]
            ) / denominator

            if distance < threshold:
                count += 1

        if count > best_count:
            best_count = count
            best_point = p

    return best_point


def groups(segments):

    horizontal = []
    vertical = []
    diagonal = []

    for seg in segments:

        a = abs(
            math.degrees(angle(seg))
        ) % 180

        if a < 15 or a > 165:
            horizontal.append(seg)

        elif 75 < a < 105:
            vertical.append(seg)

        else:
            diagonal.append(seg)

    return horizontal, vertical, diagonal


def split_diagonal(lines):

    if len(lines) < 4:
        return [], []

    family_a = []
    family_b = []

    angles = []

    for seg in lines:

        a = (
            math.degrees(angle(seg))
            % 180
        )

        angles.append(a)

    angles = np.array(angles)

    median = np.median(angles)

    for seg, a in zip(lines, angles):

        if abs(a - median) < 35:
            family_a.append(seg)

        else:
            family_b.append(seg)

    return family_a, family_b


def ground_vps(diagonal):

    a, b = split_diagonal(diagonal)

    vp1 = ransac_vp(a)
    vp2 = ransac_vp(b)

    return vp1, vp2


def horizon(vp1, vp2):

    if vp1 is None or vp2 is None:
        return None

    return np.cross(vp1, vp2)


def horizon_y(hz, x):

    if hz is None:
        return None

    a, b, c = hz

    if abs(b) < 1e-9:
        return None

    return -(a * x + c) / b


def analyze(im):

    h, w = im.shape[:2]

    segments = detect_segments(im)

    horizontal, vertical, diagonal = groups(
        segments
    )

    ground_vp1, ground_vp2 = ground_vps(
        diagonal
    )

    vertical_vp = ransac_vp(vertical)

    hz = horizon(
        ground_vp1,
        ground_vp2
    )

    hy = horizon_y(
        hz,
        w / 2
    )

    normalized_height = None

    if hy is not None:

        # Projective indicator only.
        # NOT a physical height in metres.
        normalized_height = abs(
            (h - hy) / h
        )

    return {
        "image_width": w,
        "image_height": h,

        "line_count": len(segments),

        "horizontal_lines": len(horizontal),
        "vertical_lines": len(vertical),
        "diagonal_lines": len(diagonal),

        "ground_vp_1": (
            ground_vp1.tolist()
            if ground_vp1 is not None
            else None
        ),

        "ground_vp_2": (
            ground_vp2.tolist()
            if ground_vp2 is not None
            else None
        ),

        "vertical_vp": (
            vertical_vp.tolist()
            if vertical_vp is not None
            else None
        ),

        "horizon_y": hy,

        "normalized_camera_height":
            normalized_height,

        "metric_height_available":
            False,

        "note":
            "Normalized projective "
            "camera-height indicator only. "
            "Metric camera height requires "
            "a metric scale cue."
    }


def main():

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "py src\\ground_camera_height.py "
            "frames\\calibration_frame.jpg"
        )

        return

    image_path = sys.argv[1]

    im = cv2.imread(image_path)

    if im is None:

        print(
            "Could not read image:",
            image_path
        )

        return

    result = analyze(im)

    print("=" * 70)
    print("GROUND GEOMETRY / CAMERA HEIGHT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    # --------------------------------------------------
    # SAVE STEP 10A RESULT FOR STEP 10C
    # --------------------------------------------------

    output_path = Path(
        "results/ground_camera_height.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print()
    print(
        f"Saved geometry result to: "
        f"{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

