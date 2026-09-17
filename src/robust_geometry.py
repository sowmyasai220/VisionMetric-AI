
import cv2
import numpy as np
import math
import json
import sys
from pathlib import Path


def segments(image):
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 50, 150)

    ls = cv2.HoughLinesP(
        e,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=max(40, min(g.shape) // 15),
        maxLineGap=25
    )

    out = []

    if ls is None:
        return out

    for item in ls:
        a = np.asarray(item).reshape(-1)

        if len(a) < 4:
            continue

        x1, y1, x2, y2 = map(float, a[:4])

        L = math.hypot(
            x2 - x1,
            y2 - y1
        )

        if L >= 40:
            out.append(
                (x1, y1, x2, y2, L)
            )

    return out


def line(s):
    x1, y1, x2, y2, _ = s

    return np.cross(
        [x1, y1, 1.0],
        [x2, y2, 1.0]
    )


def vp_ransac(ss, trials=600, tol=8):

    if len(ss) < 2:
        return None, 0

    ls = [
        line(s)
        for s in ss
    ]

    best = None
    bestn = 0

    for _ in range(trials):

        i, j = np.random.choice(
            len(ls),
            2,
            replace=False
        )

        p = np.cross(
            ls[i],
            ls[j]
        )

        if abs(p[2]) < 1e-8:
            continue

        p = p / p[2]

        n = 0

        for l in ls:

            den = math.hypot(
                l[0],
                l[1]
            )

            if den == 0:
                continue

            distance = abs(
                l[0] * p[0] +
                l[1] * p[1] +
                l[2]
            ) / den

            if distance < tol:
                n += 1

        if n > bestn:
            best = p
            bestn = n

    return best, bestn


def analyze(image):

    h, w = image.shape[:2]

    ss = segments(image)

    # --------------------------------------------------
    # CLASSIFY LINE DIRECTIONS
    # --------------------------------------------------

    horiz = []
    vert = []
    pos = []
    neg = []

    for s in ss:

        x1, y1, x2, y2, L = s

        deg = math.degrees(
            math.atan2(
                y2 - y1,
                x2 - x1
            )
        )

        ad = abs(deg)

        if ad < 12 or ad > 168:
            horiz.append(s)

        elif 78 < ad < 102:
            vert.append(s)

        elif deg > 0:
            pos.append(s)

        else:
            neg.append(s)

    # --------------------------------------------------
    # GROUND-PLANE VANISHING POINTS
    # --------------------------------------------------

    vp1, n1 = vp_ransac(pos)

    vp2, n2 = vp_ransac(neg)

    # --------------------------------------------------
    # VERTICAL VANISHING POINT
    # --------------------------------------------------

    vertical_vp, nv = vp_ransac(vert)

    # --------------------------------------------------
    # HORIZON
    # --------------------------------------------------

    horizon = None
    horizon_y_value = None

    if (
        vp1 is not None
        and vp2 is not None
    ):

        horizon = np.cross(
            vp1,
            vp2
        )

        a, b, c = horizon

        if abs(b) > 1e-8:

            horizon_y_value = float(
                -(
                    a * (w / 2) +
                    c
                ) / b
            )

    # --------------------------------------------------
    # HORIZON VALIDITY
    # --------------------------------------------------

    horizon_valid = (
        horizon_y_value is not None
        and -0.5 * h
        <= horizon_y_value
        <= 1.5 * h
    )

    # --------------------------------------------------
    # GEOMETRY QUALITY SCORE
    # --------------------------------------------------

    geometry_score = 0.0

    if vp1 is not None:
        geometry_score += 0.25

    if vp2 is not None:
        geometry_score += 0.25

    if vertical_vp is not None:
        geometry_score += 0.20

    if horizon_valid:
        geometry_score += 0.30

    # --------------------------------------------------
    # RELATIVE CAMERA HEIGHT INDICATOR
    # --------------------------------------------------

    normalized_camera_height = None

    if horizon_valid:

        normalized_camera_height = abs(
            (h - horizon_y_value) / h
        )

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    return {

        "image_width": w,

        "image_height": h,

        "line_count": len(ss),

        "ground_direction_1_lines": len(pos),

        "ground_direction_2_lines": len(neg),

        "vertical_lines": len(vert),

        "horizontal_lines": len(horiz),

        "ground_vp_1": (
            vp1.tolist()
            if vp1 is not None
            else None
        ),

        "ground_vp_2": (
            vp2.tolist()
            if vp2 is not None
            else None
        ),

        "vertical_vp": (
            vertical_vp.tolist()
            if vertical_vp is not None
            else None
        ),

        "horizon_y": horizon_y_value,

        "horizon_valid": bool(
            horizon_valid
        ),

        "inlier_counts": {

            "ground_1": n1,

            "ground_2": n2,

            "vertical": nv
        },

        "geometry_score": round(
            float(geometry_score),
            4
        ),

        "normalized_camera_height":
            normalized_camera_height,

        "camera_height_status":
            "relative_geometry_only",

        "metric_scale_status":
            "not_estimated_here",

        "note":
            "Camera height is currently "
            "a relative projective indicator. "
            "Metric height requires a verified "
            "metric scale factor."
    }


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "py src\\robust_geometry.py "
            "<image>"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Could not read image:",
            image_path
        )

        return

    result = analyze(image)

    print("=" * 70)

    print(
        "ROBUST CCTV GEOMETRY"
    )

    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    # --------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------

    output_path = Path(
        "results/robust_geometry.json"
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

