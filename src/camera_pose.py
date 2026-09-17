import cv2
import numpy as np
import math


class CameraPose:
    """
    Stores the estimated geometric pose of a CCTV camera.

    Important:
    This class deliberately separates geometric scale
    from metric scale. We do not invent feet/metres.
    """

    def __init__(
        self,
        camera_height=None,
        pitch=None,
        yaw=None,
        roll=None,
        scale=None
    ):
        self.camera_height = camera_height
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.scale = scale

    def to_dict(self):
        return {
            "camera_height": self.camera_height,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "scale": self.scale
        }


# ============================================================
# VANISHING POINT UTILITIES
# ============================================================

def normalize_point(point):
    """
    Convert a 2D point into homogeneous coordinates.
    """

    x, y = point

    return np.array(
        [float(x), float(y), 1.0],
        dtype=np.float64
    )


def normalize_line(line):
    """
    Normalize a homogeneous image line.
    """

    line = np.asarray(
        line,
        dtype=np.float64
    )

    norm = math.sqrt(
        line[0] ** 2 +
        line[1] ** 2
    )

    if norm < 1e-12:
        return line

    return line / norm


def line_from_points(p1, p2):
    """
    Construct a homogeneous line through two points.
    """

    a = normalize_point(p1)
    b = normalize_point(p2)

    line = np.cross(a, b)

    return normalize_line(line)


def vanishing_point_from_lines(lines):
    """
    Estimate a vanishing point from multiple image lines.

    Uses least-squares intersection in homogeneous form.
    """

    if len(lines) < 2:
        return None

    matrix = []

    for line in lines:

        x1, y1, x2, y2 = line

        l = line_from_points(
            (x1, y1),
            (x2, y2)
        )

        matrix.append(l)

    matrix = np.asarray(
        matrix,
        dtype=np.float64
    )

    try:

        _, _, vh = np.linalg.svd(
            matrix
        )

    except np.linalg.LinAlgError:

        return None

    point = vh[-1]

    if abs(point[2]) < 1e-12:
        return None

    point = point / point[2]

    if not np.all(
        np.isfinite(point)
    ):
        return None

    return (
        float(point[0]),
        float(point[1])
    )


# ============================================================
# CAMERA INTRINSIC ESTIMATION
# ============================================================

def estimate_principal_point(
    width,
    height
):
    """
    Initial principal-point estimate.

    For an uncalibrated CCTV image the optical center
    is usually close to the image center.

    This is an estimate, not a hard-coded physical value.
    """

    return (
        width / 2.0,
        height / 2.0
    )


def build_intrinsic_matrix(
    width,
    height,
    focal_length=None
):
    """
    Construct an approximate camera intrinsic matrix.

    If focal length is unknown, we do NOT claim that
    the result is metric. The matrix is used only for
    projective/pose reasoning.

    A later calibration stage can replace this estimate
    when the footage contains enough information.
    """

    cx, cy = estimate_principal_point(
        width,
        height
    )

    if focal_length is None:

        # Neutral projective initialization.
        focal_length = max(
            width,
            height
        )

    K = np.array(
        [
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ],
        dtype=np.float64
    )

    return K


# ============================================================
# VANISHING-DIRECTION VECTORS
# ============================================================

def image_point_to_camera_ray(
    point,
    K
):
    """
    Convert an image point into a camera-coordinate ray.
    """

    x, y = point

    image_point = np.array(
        [
            [x],
            [y],
            [1.0]
        ],
        dtype=np.float64
    )

    ray = np.linalg.inv(K) @ image_point

    norm = np.linalg.norm(ray)

    if norm < 1e-12:
        return None

    ray /= norm

    return ray.flatten()


def angle_between_vectors(
    vector_a,
    vector_b
):
    """
    Return angle between two vectors in radians.
    """

    a = np.asarray(
        vector_a,
        dtype=np.float64
    )

    b = np.asarray(
        vector_b,
        dtype=np.float64
    )

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if (
        a_norm < 1e-12
        or b_norm < 1e-12
    ):
        return None

    cosine = np.dot(
        a,
        b
    ) / (
        a_norm *
        b_norm
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cosine)
    )


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def estimate_camera_orientation(
    vanishing_points,
    K
):
    """
    Estimate relative camera orientation from
    orthogonal vanishing directions.

    vanishing_points should contain at least two
    approximately orthogonal scene directions.

    This produces orientation information only.
    It does NOT produce a metric camera height.
    """

    if len(vanishing_points) < 2:
        return None

    rays = []

    for point in vanishing_points[:3]:

        ray = image_point_to_camera_ray(
            point,
            K
        )

        if ray is not None:
            rays.append(ray)

    if len(rays) < 2:
        return None

    # Normalize directions.
    directions = []

    for ray in rays:

        ray = ray / np.linalg.norm(
            ray
        )

        directions.append(ray)

    return {
        "directions": directions
    }


# ============================================================
# GROUND NORMAL
# ============================================================

def estimate_ground_normal(
    direction_a,
    direction_b
):
    """
    Estimate the normal of a ground plane from
    two directions lying on that plane.

    The result is a direction, not a metric height.
    """

    a = np.asarray(
        direction_a,
        dtype=np.float64
    )

    b = np.asarray(
        direction_b,
        dtype=np.float64
    )

    normal = np.cross(
        a,
        b
    )

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-12:
        return None

    normal /= norm

    return normal


# ============================================================
# CAMERA HEIGHT IN PROJECTIVE SPACE
# ============================================================

def estimate_relative_camera_height(
    ground_normal,
    camera_direction=None
):
    """
    Return a normalized camera-height representation.

    IMPORTANT:
    Without a metric scale cue this is NOT feet/metres.

    The function therefore returns a relative value that
    can later be converted to metric units once scale
    has been recovered from the scene.
    """

    if ground_normal is None:
        return None

    normal = np.asarray(
        ground_normal,
        dtype=np.float64
    )

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-12:
        return None

    normal /= norm

    if camera_direction is None:

        camera_direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64
        )

    camera_direction = (
        camera_direction /
        np.linalg.norm(
            camera_direction
        )
    )

    alignment = abs(
        np.dot(
            normal,
            camera_direction
        )
    )

    return float(
        alignment
    )


# ============================================================
# COMPLETE POSE ANALYSIS
# ============================================================

def estimate_camera_pose(
    frame,
    vanishing_points
):
    """
    Estimate geometric camera information from
    image vanishing points.

    No airport dimensions, camera coordinates,
    camera height, or tile measurements are used.
    """

    height, width = frame.shape[:2]

    K = build_intrinsic_matrix(
        width,
        height
    )

    orientation = estimate_camera_orientation(
        vanishing_points,
        K
    )

    if orientation is None:

        return CameraPose()

    directions = orientation[
        "directions"
    ]

    ground_normal = None

    if len(directions) >= 2:

        ground_normal = (
            estimate_ground_normal(
                directions[0],
                directions[1]
            )
        )

    relative_height = (
        estimate_relative_camera_height(
            ground_normal
        )
    )

    return CameraPose(
        camera_height=relative_height
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_pose(pose):
    """
    Print pose information.
    """

    print()
    print("=" * 60)
    print("CAMERA GEOMETRIC POSE")
    print("=" * 60)

    print(
        f"Relative camera height : "
        f"{pose.camera_height}"
    )

    print(
        f"Pitch                  : "
        f"{pose.pitch}"
    )

    print(
        f"Yaw                    : "
        f"{pose.yaw}"
    )

    print(
        f"Roll                   : "
        f"{pose.roll}"
    )

    print(
        f"Metric scale           : "
        f"{pose.scale}"
    )

    print()
    print(
        "Note: relative values are not automatically "
        "feet/metres."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAMERA POSE MODULE")
    print("=" * 60)

    print()
    print(
        "Camera pose estimation module ready."
    )

    print(
        "This module separates projective geometry "
        "from metric scale."
    )