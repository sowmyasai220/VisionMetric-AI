"""
Common Object / Scene Feature Matching
---------------------------------------

Environment-independent feature correspondence between
different CCTV views.

This module does NOT assume:
- object class
- airport layout
- camera coordinates
- camera height
- tile dimensions

It uses visual features and geometric consistency to find
regions that are likely to correspond to the same physical
scene/object across two views.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class MatchResult:
    image_a: str
    image_b: str

    keypoints_a: int
    keypoints_b: int

    raw_matches: int
    good_matches: int
    geometric_matches: int

    homography_found: bool
    confidence: float

    matched_points_a: list
    matched_points_b: list

    notes: list[str]


# ---------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------

def create_detector():
    """
    SIFT is preferred because it is robust to scale and
    moderate viewpoint changes.
    """

    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(
            nfeatures=3000,
            contrastThreshold=0.02,
            edgeThreshold=10,
        )

    # Fallback for OpenCV builds without SIFT.
    return cv2.ORB_create(
        nfeatures=3000
    )


def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    detector = create_detector()

    keypoints, descriptors = detector.detectAndCompute(
        gray,
        None,
    )

    return keypoints, descriptors


# ---------------------------------------------------------------------
# FEATURE MATCHING
# ---------------------------------------------------------------------

def match_descriptors(
    descriptors_a,
    descriptors_b,
):
    """
    Ratio-test feature matching.
    """

    if descriptors_a is None or descriptors_b is None:
        return []

    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return []

    # SIFT descriptors are floating-point.
    # ORB descriptors are binary.
    if descriptors_a.dtype == np.uint8:

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

    else:

        matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False,
        )

    knn_matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        first, second = pair

        # Lowe ratio test.
        if first.distance < 0.72 * second.distance:
            good.append(first)

    return good


# ---------------------------------------------------------------------
# GEOMETRIC CONSISTENCY
# ---------------------------------------------------------------------

def geometric_filter(
    keypoints_a,
    keypoints_b,
    matches,
):
    """
    Use RANSAC homography estimation to remove
    accidental feature matches.
    """

    if len(matches) < 4:
        return None, []

    points_a = np.float32([
        keypoints_a[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    points_b = np.float32([
        keypoints_b[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        points_a,
        points_b,
        cv2.RANSAC,
        4.0,
    )

    if homography is None or mask is None:
        return None, []

    mask = mask.ravel().astype(bool)

    inlier_matches = [
        match
        for match, keep in zip(matches, mask)
        if keep
    ]

    return homography, inlier_matches


# ---------------------------------------------------------------------
# MATCH CLUSTERING
# ---------------------------------------------------------------------

def spatial_clusters(
    keypoints,
    matches,
    image_shape,
):
    """
    Group geometrically consistent matches by their source
    image location.

    A physical object/region normally produces several nearby
    feature correspondences rather than one isolated match.
    """

    if not matches:
        return []

    height, width = image_shape[:2]

    points = np.array([
        keypoints[m.queryIdx].pt
        for m in matches
    ])

    # Normalize image coordinates.
    normalized = np.column_stack([
        points[:, 0] / max(width, 1),
        points[:, 1] / max(height, 1),
    ])

    # Simple grid-based grouping.
    grid_size = 8

    cells = {}

    for index, point in enumerate(normalized):

        cell_x = int(point[0] * grid_size)
        cell_y = int(point[1] * grid_size)

        key = (
            min(cell_x, grid_size - 1),
            min(cell_y, grid_size - 1),
        )

        cells.setdefault(
            key,
            []
        ).append(index)

    clusters = []

    for indices in cells.values():

        if len(indices) >= 2:

            clusters.append(indices)

    return clusters


# ---------------------------------------------------------------------
# MATCH SCORE
# ---------------------------------------------------------------------

def calculate_confidence(
    raw_count,
    good_count,
    geometric_count,
):
    """
    Confidence is intentionally conservative.
    It measures correspondence quality, not metric accuracy.
    """

    if raw_count <= 0:
        return 0.0

    ratio_good = good_count / raw_count

    if good_count > 0:
        ratio_geo = geometric_count / good_count
    else:
        ratio_geo = 0.0

    # More inliers are useful, but cap the contribution.
    quantity_score = min(
        1.0,
        geometric_count / 40.0,
    )

    confidence = (
        0.30 * ratio_good
        + 0.40 * ratio_geo
        + 0.30 * quantity_score
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )
    )


# ---------------------------------------------------------------------
# MAIN IMAGE MATCHING
# ---------------------------------------------------------------------

def match_images(
    image_a,
    image_b,
    path_a="image_a",
    path_b="image_b",
):

    keypoints_a, descriptors_a = extract_features(
        image_a
    )

    keypoints_b, descriptors_b = extract_features(
        image_b
    )

    good_matches = match_descriptors(
        descriptors_a,
        descriptors_b,
    )

    homography, inliers = geometric_filter(
        keypoints_a,
        keypoints_b,
        good_matches,
    )

    matched_points_a = []
    matched_points_b = []

    for match in inliers:

        point_a = keypoints_a[
            match.queryIdx
        ].pt

        point_b = keypoints_b[
            match.trainIdx
        ].pt

        matched_points_a.append([
            float(point_a[0]),
            float(point_a[1]),
        ])

        matched_points_b.append([
            float(point_b[0]),
            float(point_b[1]),
        ])

    confidence = calculate_confidence(
        raw_count=(
            len(descriptors_a)
            if descriptors_a is not None
            else 0
        ),
        good_count=len(good_matches),
        geometric_count=len(inliers),
    )

    notes = []

    if len(inliers) < 4:
        notes.append(
            "Insufficient geometrically consistent matches."
        )

    elif len(inliers) < 10:
        notes.append(
            "Some correspondence exists, but confidence is limited."
        )

    else:
        notes.append(
            "A geometrically consistent correspondence was detected."
        )

    notes.append(
        "Correspondence confidence does not imply metric distance accuracy."
    )

    return MatchResult(
        image_a=str(path_a),
        image_b=str(path_b),

        keypoints_a=len(keypoints_a),
        keypoints_b=len(keypoints_b),

        raw_matches=len(
            good_matches
        ),

        good_matches=len(
            good_matches
        ),

        geometric_matches=len(
            inliers
        ),

        homography_found=(
            homography is not None
        ),

        confidence=confidence,

        matched_points_a=matched_points_a,
        matched_points_b=matched_points_b,

        notes=notes,
    )


# ---------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------

def draw_matches(
    image_a,
    image_b,
    keypoints_a,
    keypoints_b,
    matches,
):

    return cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# ---------------------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------------------

def load_image(path):

    image = cv2.imread(
        str(path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Find visually and geometrically "
            "consistent correspondences between "
            "two CCTV views."
        )
    )

    parser.add_argument(
        "image_a",
        help="First image/frame",
    )

    parser.add_argument(
        "image_b",
        help="Second image/frame",
    )

    parser.add_argument(
        "--output",
        default="results/common_object_match.json",
        help="JSON result path",
    )

    args = parser.parse_args()

    path_a = Path(
        args.image_a
    )

    path_b = Path(
        args.image_b
    )

    image_a = load_image(
        path_a
    )

    image_b = load_image(
        path_b
    )

    result = match_images(
        image_a,
        image_b,
        path_a,
        path_b,
    )

    print()
    print("=" * 60)
    print("COMMON OBJECT / SCENE MATCHING")
    print("=" * 60)

    print(
        f"Keypoints A       : {result.keypoints_a}"
    )

    print(
        f"Keypoints B       : {result.keypoints_b}"
    )

    print(
        f"Good matches      : {result.good_matches}"
    )

    print(
        f"Geometric matches : {result.geometric_matches}"
    )

    print(
        f"Homography found  : {result.homography_found}"
    )

    print(
        f"Confidence        : {result.confidence:.3f}"
    )

    print()

    for note in result.notes:
        print(
            f"- {note}"
        )

    print("=" * 60)

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

    print()
    print(
        f"Result saved to: {output_path}"
    )


if __name__ == "__main__":
    main()