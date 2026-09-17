"""
Cross-Camera 3D Geometry
------------------------

Triangulates corresponding image points from two camera views.

The cameras are represented in a normalized coordinate system.
Absolute metres/feet are NOT assumed.

This module establishes relative 3D geometry first.
Metric scaling is handled separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import cv2
import numpy as np


@dataclass
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class TriangulationResult:
    point_a: list
    point_b: list

    point_3d: list | None

    depth_camera_a: float | None
    depth_camera_b: float | None

    reprojection_error_a: float | None
    reprojection_error_b: float | None

    valid: bool
    confidence: float

    notes: list[str]


# ---------------------------------------------------------------------
# CAMERA MATRIX
# ---------------------------------------------------------------------

def make_projection_matrix(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    P = K [R | t]
    """

    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    translation = np.asarray(
        translation,
        dtype=float,
    ).reshape(3, 1)

    extrinsic = np.hstack([
        rotation,
        translation,
    ])

    return intrinsic @ extrinsic


# ---------------------------------------------------------------------
# POINT TRIANGULATION
# ---------------------------------------------------------------------

def triangulate_point(
    point_a,
    point_b,
    projection_a,
    projection_b,
):
    """
    Triangulate one corresponding image point.

    point_a and point_b:
        (x, y)
    """

    points_a = np.asarray(
        point_a,
        dtype=float,
    ).reshape(2, 1)

    points_b = np.asarray(
        point_b,
        dtype=float,
    ).reshape(2, 1)

    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        points_a,
        points_b,
    )

    if homogeneous.shape != (4, 1):
        return None

    w = homogeneous[3, 0]

    if abs(w) < 1e-10:
        return None

    point_3d = (
        homogeneous[:3, 0] / w
    )

    return point_3d


# ---------------------------------------------------------------------
# REPROJECTION
# ---------------------------------------------------------------------

def project_point(
    point_3d,
    projection,
):
    """
    Project a 3D point into an image.
    """

    point = np.asarray(
        point_3d,
        dtype=float,
    ).reshape(3, 1)

    homogeneous = np.vstack([
        point,
        [[1.0]],
    ])

    projected = projection @ homogeneous

    if abs(projected[2, 0]) < 1e-10:
        return None

    return np.array([
        projected[0, 0] / projected[2, 0],
        projected[1, 0] / projected[2, 0],
    ])


def reprojection_error(
    original_point,
    projected_point,
):
    if projected_point is None:
        return None

    original = np.asarray(
        original_point,
        dtype=float,
    )

    projected = np.asarray(
        projected_point,
        dtype=float,
    )

    return float(
        np.linalg.norm(
            original - projected
        )
    )


# ---------------------------------------------------------------------
# DEPTH
# ---------------------------------------------------------------------

def camera_coordinates(
    point_3d,
    rotation,
    translation,
):
    """
    Convert world/relative coordinates into camera coordinates.
    """

    point = np.asarray(
        point_3d,
        dtype=float,
    )

    rotation = np.asarray(
        rotation,
        dtype=float,
    )

    translation = np.asarray(
        translation,
        dtype=float,
    ).reshape(3)

    return (
        rotation @ point
        + translation
    )


def point_depth(
    point_3d,
    rotation,
    translation,
):
    camera_point = camera_coordinates(
        point_3d,
        rotation,
        translation,
    )

    return float(
        camera_point[2]
    )


# ---------------------------------------------------------------------
# FULL TRIANGULATION
# ---------------------------------------------------------------------

def triangulate_correspondence(
    point_a,
    point_b,
    projection_a,
    projection_b,
    rotation_a=None,
    translation_a=None,
    rotation_b=None,
    translation_b=None,
):
    """
    Full triangulation with reprojection validation.
    """

    point_3d = triangulate_point(
        point_a,
        point_b,
        projection_a,
        projection_b,
    )

    if point_3d is None:

        return TriangulationResult(
            point_a=list(point_a),
            point_b=list(point_b),
            point_3d=None,
            depth_camera_a=None,
            depth_camera_b=None,
            reprojection_error_a=None,
            reprojection_error_b=None,
            valid=False,
            confidence=0.0,
            notes=[
                "Triangulation failed."
            ],
        )

    projected_a = project_point(
        point_3d,
        projection_a,
    )

    projected_b = project_point(
        point_3d,
        projection_b,
    )

    error_a = reprojection_error(
        point_a,
        projected_a,
    )

    error_b = reprojection_error(
        point_b,
        projected_b,
    )

    depth_a = None
    depth_b = None

    if (
        rotation_a is not None
        and translation_a is not None
    ):
        depth_a = point_depth(
            point_3d,
            rotation_a,
            translation_a,
        )

    if (
        rotation_b is not None
        and translation_b is not None
    ):
        depth_b = point_depth(
            point_3d,
            rotation_b,
            translation_b,
        )

    # -------------------------------------------------------------
    # VALIDITY
    # -------------------------------------------------------------

    errors = [
        e for e in (
            error_a,
            error_b,
        )
        if e is not None
    ]

    mean_error = (
        float(np.mean(errors))
        if errors
        else None
    )

    valid = True
    notes = []

    if mean_error is not None:

        if mean_error > 10:
            valid = False

            notes.append(
                "Large reprojection error."
            )

        elif mean_error > 5:

            notes.append(
                "Moderate reprojection error."
            )

        else:

            notes.append(
                "Low reprojection error."
            )

    # -------------------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------------------

    if mean_error is None:

        confidence = 0.5

    else:

        confidence = math_error_confidence(
            mean_error
        )

    if point_3d[2] < 0:

        notes.append(
            "Triangulated point has negative Z in the relative frame."
        )

        valid = False

    notes.append(
        "3D coordinates are relative/projective until metric scale is established."
    )

    return TriangulationResult(
        point_a=list(point_a),
        point_b=list(point_b),

        point_3d=[
            float(x)
            for x in point_3d
        ],

        depth_camera_a=depth_a,
        depth_camera_b=depth_b,

        reprojection_error_a=error_a,
        reprojection_error_b=error_b,

        valid=valid,
        confidence=confidence,

        notes=notes,
    )


def math_error_confidence(
    error,
):
    """
    Convert reprojection error into a conservative confidence score.
    """

    if error <= 1:
        return 1.0

    if error >= 20:
        return 0.0

    return float(
        1.0 - (
            (error - 1.0)
            / 19.0
        )
    )


# ---------------------------------------------------------------------
# MULTIPLE CORRESPONDENCES
# ---------------------------------------------------------------------

def triangulate_multiple(
    points_a,
    points_b,
    projection_a,
    projection_b,
):
    """
    Triangulate many corresponding points.
    """

    if len(points_a) != len(points_b):
        raise ValueError(
            "Both point lists must have the same length."
        )

    results = []

    for point_a, point_b in zip(
        points_a,
        points_b,
    ):

        result = triangulate_correspondence(
            point_a,
            point_b,
            projection_a,
            projection_b,
        )

        results.append(result)

    return results


# ---------------------------------------------------------------------
# RELATIVE DISTANCE
# ---------------------------------------------------------------------

def distance_between_points(
    point_a,
    point_b,
):
    """
    Euclidean distance in the current relative coordinate system.
    """

    a = np.asarray(
        point_a,
        dtype=float,
    )

    b = np.asarray(
        point_b,
        dtype=float,
    )

    return float(
        np.linalg.norm(
            a - b
        )
    )


def camera_to_object_distance(
    camera_position,
    object_position,
):
    """
    Distance from camera center to object point
    in relative 3D coordinates.
    """

    return distance_between_points(
        camera_position,
        object_position,
    )


# ---------------------------------------------------------------------
# DEMO / CLI
# ---------------------------------------------------------------------

def create_demo_camera_matrices():

    intrinsic = np.array([
        [800.0, 0.0, 400.0],
        [0.0, 800.0, 300.0],
        [0.0, 0.0, 1.0],
    ])

    rotation_a = np.eye(3)

    translation_a = np.zeros(3)

    rotation_b = np.eye(3)

    translation_b = np.array([
        -2.0,
        0.0,
        0.0,
    ])

    projection_a = make_projection_matrix(
        intrinsic,
        rotation_a,
        translation_a,
    )

    projection_b = make_projection_matrix(
        intrinsic,
        rotation_b,
        translation_b,
    )

    return (
        projection_a,
        projection_b,
        rotation_a,
        translation_a,
        rotation_b,
        translation_b,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cross-camera 3D triangulation foundation."
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic triangulation test.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CROSS-CAMERA 3D GEOMETRY")
    print("=" * 60)

    if not args.demo:

        print()
        print(
            "Module loaded successfully."
        )

        print(
            "Use --demo for a synthetic triangulation test."
        )

        return

    (
        projection_a,
        projection_b,
        rotation_a,
        translation_a,
        rotation_b,
        translation_b,
    ) = create_demo_camera_matrices()

    # Synthetic corresponding observations.
    point_a = [
        440.0,
        320.0,
    ]

    point_b = [
        400.0,
        320.0,
    ]

    result = triangulate_correspondence(
        point_a,
        point_b,
        projection_a,
        projection_b,
        rotation_a,
        translation_a,
        rotation_b,
        translation_b,
    )

    print()

    print(
        f"Point A: {result.point_a}"
    )

    print(
        f"Point B: {result.point_b}"
    )

    print(
        f"3D point: {result.point_3d}"
    )

    print(
        f"Depth A: {result.depth_camera_a}"
    )

    print(
        f"Depth B: {result.depth_camera_b}"
    )

    print(
        f"Reprojection error A: "
        f"{result.reprojection_error_a}"
    )

    print(
        f"Reprojection error B: "
        f"{result.reprojection_error_b}"
    )

    print(
        f"Valid: {result.valid}"
    )

    print(
        f"Confidence: {result.confidence:.3f}"
    )

    print()

    for note in result.notes:
        print(
            f"- {note}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()