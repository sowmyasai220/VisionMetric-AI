"""
Camera Geometry
---------------

Environment-independent camera geometry estimation.

This module estimates:
- image center
- horizon / vanishing line
- camera focal length when available
- vertical vanishing point
- camera pitch/orientation
- ground-plane direction
- normalized camera height representation

IMPORTANT:
Absolute metric height is NOT invented here.
Without a trustworthy metric cue, the result remains scale-ambiguous.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class CameraGeometryResult:
    width: int
    height: int

    principal_x: float
    principal_y: float

    focal_length: Optional[float]

    horizon_x: Optional[float]
    horizon_y: Optional[float]

    vertical_vanishing_x: Optional[float]
    vertical_vanishing_y: Optional[float]

    pitch_degrees: Optional[float]

    ground_direction_x: Optional[float]
    ground_direction_y: Optional[float]

    normalized_height: Optional[float]

    metric_scale_available: bool

    confidence: float

    notes: list[str]


# ---------------------------------------------------------------------
# BASIC GEOMETRY
# ---------------------------------------------------------------------

def normalize_point(point):
    """Convert homogeneous point to Cartesian coordinates."""

    point = np.asarray(point, dtype=float).reshape(-1)

    if len(point) < 3:
        return None

    if abs(point[2]) < 1e-9:
        return None

    return np.array([
        point[0] / point[2],
        point[1] / point[2],
    ])


def cross(a, b):
    """Homogeneous cross product."""

    return np.cross(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
    )


def line_from_points(p1, p2):
    """Create homogeneous image line from two points."""

    return cross(
        [p1[0], p1[1], 1.0],
        [p2[0], p2[1], 1.0],
    )


def intersect_lines(line1, line2):
    """Intersection of two homogeneous lines."""

    point = cross(line1, line2)

    if abs(point[2]) < 1e-9:
        return None

    return normalize_point(point)


# ---------------------------------------------------------------------
# VANISHING POINT
# ---------------------------------------------------------------------

def estimate_vanishing_point(lines):
    """
    Estimate a vanishing point from image lines.

    Each line is expected as:
        [x1, y1, x2, y2]
    """

    if lines is None or len(lines) < 2:
        return None

    intersections = []

    for i in range(len(lines)):

        x1, y1, x2, y2 = lines[i]

        l1 = line_from_points(
            (x1, y1),
            (x2, y2),
        )

        for j in range(i + 1, len(lines)):

            a1, b1, a2, b2 = lines[j]

            l2 = line_from_points(
                (a1, b1),
                (a2, b2),
            )

            point = intersect_lines(l1, l2)

            if point is None:
                continue

            if np.all(np.isfinite(point)):
                intersections.append(point)

    if len(intersections) < 3:
        return None

    intersections = np.asarray(intersections)

    # Robust median estimate.
    vp = np.median(
        intersections,
        axis=0,
    )

    return vp


# ---------------------------------------------------------------------
# LINE DETECTION
# ---------------------------------------------------------------------

def detect_lines(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    min_line_length = max(
        25,
        int(min(gray.shape) * 0.08),
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=15,
    )

    if lines is None:
        return np.empty(
            (0, 4),
            dtype=float,
        )

    return lines.reshape(
        -1,
        4,
    ).astype(float)


# ---------------------------------------------------------------------
# DIRECTION GROUPING
# ---------------------------------------------------------------------

def line_angle(line):

    x1, y1, x2, y2 = line

    angle = math.atan2(
        y2 - y1,
        x2 - x1,
    )

    angle = math.degrees(angle)

    # Direction is orientation-independent.
    if angle < 0:
        angle += 180

    return angle


def line_length(line):

    x1, y1, x2, y2 = line

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def group_lines(lines):

    groups = {
        "horizontal": [],
        "vertical": [],
        "diagonal_a": [],
        "diagonal_b": [],
    }

    if len(lines) == 0:
        return groups

    for line in lines:

        angle = line_angle(line)

        length = line_length(line)

        if length < 25:
            continue

        # Near horizontal.
        if angle < 15 or angle > 165:
            groups["horizontal"].append(line)

        # Near vertical.
        elif 75 <= angle <= 105:
            groups["vertical"].append(line)

        # Two broad diagonal families.
        elif angle < 90:
            groups["diagonal_a"].append(line)

        else:
            groups["diagonal_b"].append(line)

    return groups


# ---------------------------------------------------------------------
# HORIZON
# ---------------------------------------------------------------------

def estimate_horizon(frame):

    lines = detect_lines(frame)
    groups = group_lines(lines)

    candidates = []

    families = [
        groups["horizontal"],
        groups["diagonal_a"],
        groups["diagonal_b"],
    ]

    for family_a in families:

        if len(family_a) < 2:
            continue

        vp = estimate_vanishing_point(
            family_a
        )

        if vp is not None:
            candidates.append(vp)

    if not candidates:
        return None

    candidates = np.asarray(
        candidates
    )

    # Use the most central stable estimate.
    horizon = np.median(
        candidates,
        axis=0,
    )

    return horizon


# ---------------------------------------------------------------------
# FOCAL LENGTH
# ---------------------------------------------------------------------

def estimate_focal_length_from_vps(
    vp1,
    vp2,
    principal_point,
):
    """
    For orthogonal directions:

        (v1-c)^T(v2-c) + f^2 = 0

    Therefore:

        f^2 = -(v1-c)^T(v2-c)
    """

    if vp1 is None or vp2 is None:
        return None

    v1 = np.asarray(vp1, dtype=float)
    v2 = np.asarray(vp2, dtype=float)
    c = np.asarray(principal_point, dtype=float)

    value = -np.dot(
        v1 - c,
        v2 - c,
    )

    if value <= 0:
        return None

    return float(
        math.sqrt(value)
    )


# ---------------------------------------------------------------------
# CAMERA PITCH
# ---------------------------------------------------------------------

def estimate_pitch(
    vertical_vp,
    principal_point,
    focal_length,
):
    """
    Estimate camera pitch relative to the horizontal plane.

    This is an angular estimate and therefore does not require
    absolute scene scale.
    """

    if vertical_vp is None:
        return None

    if focal_length is None or focal_length <= 0:
        return None

    vx, vy = vertical_vp
    cx, cy = principal_point

    dy = vy - cy

    # Vertical image coordinate relative to principal point.
    angle = math.atan2(
        dy,
        focal_length,
    )

    return float(
        math.degrees(angle)
    )


# ---------------------------------------------------------------------
# NORMALIZED HEIGHT
# ---------------------------------------------------------------------

def normalized_camera_height(
    vertical_vp,
    principal_point,
    focal_length,
):
    """
    Return a normalized height representation.

    This is NOT metres or feet.

    It expresses camera height relative to the unknown scene scale.
    """

    if vertical_vp is None:
        return None

    if focal_length is None or focal_length <= 0:
        return None

    _, vy = vertical_vp
    _, cy = principal_point

    difference = abs(
        vy - cy
    )

    if difference < 1e-9:
        return None

    return float(
        focal_length / difference
    )


# ---------------------------------------------------------------------
# MAIN ESTIMATION
# ---------------------------------------------------------------------

def estimate_camera_geometry(
    frame,
    focal_length: Optional[float] = None,
    vertical_vanishing_point=None,
):
    height, width = frame.shape[:2]

    principal_point = np.array([
        width / 2.0,
        height / 2.0,
    ])

    lines = detect_lines(frame)

    groups = group_lines(lines)

    # -------------------------------------------------------------
    # Ground-plane vanishing geometry
    # -------------------------------------------------------------

    vp_a = estimate_vanishing_point(
        groups["diagonal_a"]
    )

    vp_b = estimate_vanishing_point(
        groups["diagonal_b"]
    )

    # -------------------------------------------------------------
    # Vertical VP
    # -------------------------------------------------------------

    vertical_vp = vertical_vanishing_point

    if vertical_vp is None:

        vertical_vp = estimate_vanishing_point(
            groups["vertical"]
        )

    # -------------------------------------------------------------
    # Horizon
    # -------------------------------------------------------------

    horizon = estimate_horizon(
        frame
    )

    # -------------------------------------------------------------
    # Focal length
    # -------------------------------------------------------------

    if focal_length is None:

        focal_length = (
            estimate_focal_length_from_vps(
                vp_a,
                vp_b,
                principal_point,
            )
        )

    # -------------------------------------------------------------
    # Pitch
    # -------------------------------------------------------------

    pitch = estimate_pitch(
        vertical_vp,
        principal_point,
        focal_length,
    )

    # -------------------------------------------------------------
    # Normalized height
    # -------------------------------------------------------------

    normalized_height = (
        normalized_camera_height(
            vertical_vp,
            principal_point,
            focal_length,
        )
    )

    # -------------------------------------------------------------
    # Ground direction
    # -------------------------------------------------------------

    ground_direction = None

    if vp_a is not None:

        direction = np.asarray(vp_a) - principal_point

        magnitude = np.linalg.norm(
            direction
        )

        if magnitude > 1e-9:

            ground_direction = (
                direction / magnitude
            )

    # -------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------

    confidence_components = []

    if len(lines) >= 10:
        confidence_components.append(0.25)

    if vp_a is not None:
        confidence_components.append(0.20)

    if vp_b is not None:
        confidence_components.append(0.20)

    if vertical_vp is not None:
        confidence_components.append(0.20)

    if horizon is not None:
        confidence_components.append(0.15)

    confidence = sum(
        confidence_components
    )

    notes = [
        "Camera geometry is estimated from image structure.",
        "Absolute metric height is not assumed.",
        "Normalized height is scene-scale dependent.",
    ]

    if focal_length is None:
        notes.append(
            "Focal length could not be reliably estimated."
        )

    if vertical_vp is None:
        notes.append(
            "Vertical vanishing point could not be reliably estimated."
        )

    if horizon is None:
        notes.append(
            "Ground-plane horizon could not be reliably estimated."
        )

    return CameraGeometryResult(
        width=width,
        height=height,

        principal_x=float(
            principal_point[0]
        ),
        principal_y=float(
            principal_point[1]
        ),

        focal_length=focal_length,

        horizon_x=(
            float(horizon[0])
            if horizon is not None
            else None
        ),

        horizon_y=(
            float(horizon[1])
            if horizon is not None
            else None
        ),

        vertical_vanishing_x=(
            float(vertical_vp[0])
            if vertical_vp is not None
            else None
        ),

        vertical_vanishing_y=(
            float(vertical_vp[1])
            if vertical_vp is not None
            else None
        ),

        pitch_degrees=pitch,

        ground_direction_x=(
            float(ground_direction[0])
            if ground_direction is not None
            else None
        ),

        ground_direction_y=(
            float(ground_direction[1])
            if ground_direction is not None
            else None
        ),

        normalized_height=normalized_height,

        metric_scale_available=False,

        confidence=float(
            min(1.0, confidence)
        ),

        notes=notes,
    )


# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------

def print_result(result):

    print()
    print("=" * 60)
    print("CAMERA GEOMETRY")
    print("=" * 60)

    print(
        f"Image size       : "
        f"{result.width} x {result.height}"
    )

    print(
        f"Principal point  : "
        f"({result.principal_x:.2f}, "
        f"{result.principal_y:.2f})"
    )

    print(
        f"Focal length     : "
        f"{result.focal_length}"
    )

    print(
        f"Horizon          : "
        f"({result.horizon_x}, "
        f"{result.horizon_y})"
    )

    print(
        f"Vertical VP      : "
        f"({result.vertical_vanishing_x}, "
        f"{result.vertical_vanishing_y})"
    )

    print(
        f"Camera pitch     : "
        f"{result.pitch_degrees}"
    )

    print(
        f"Normalized height: "
        f"{result.normalized_height}"
    )

    print(
        f"Metric scale     : "
        f"{result.metric_scale_available}"
    )

    print(
        f"Confidence       : "
        f"{result.confidence:.2f}"
    )

    print()
    print("Notes:")

    for note in result.notes:
        print(f"  - {note}")

    print("=" * 60)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estimate environment-independent "
            "camera geometry from a video frame."
        )
    )

    parser.add_argument(
        "image",
        help="Path to an image/frame.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            image_path
        )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise RuntimeError(
            "Could not read image."
        )

    result = estimate_camera_geometry(
        frame
    )

    print_result(
        result
    )

    if args.output:

        output_path = Path(
            args.output
        )

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
                asdict(result),
                file,
                indent=2,
            )

        print(
            f"\nSaved: {output_path}"
        )


if __name__ == "__main__":
    main()