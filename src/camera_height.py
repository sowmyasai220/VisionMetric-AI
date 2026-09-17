import cv2
import numpy as np
import math


class CameraHeightEstimator:
    """
    Estimate camera height from a perspective view of a flat floor.

    The method uses:
    - Four or more image points lying on the floor
    - Their corresponding real-world ground coordinates
    - An estimated camera intrinsic matrix

    IMPORTANT:
    The physical scale must come from something visible in the scene.
    If the floor tile size is only approximately known, the result is
    an estimate and should be reported as such.
    """

    def __init__(
        self,
        image_width,
        image_height,
        focal_length_px=None
    ):
        self.image_width = image_width
        self.image_height = image_height

        # Principal point approximation.
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # If focal length is not known, use a reasonable initial
        # perspective estimate. This will later be replaced by
        # focal-length estimation from the actual CCTV geometry.
        if focal_length_px is None:
            focal_length_px = max(image_width, image_height)

        self.focal_length_px = float(focal_length_px)

        self.K = np.array([
            [self.focal_length_px, 0, self.cx],
            [0, self.focal_length_px, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        self.homography = None
        self.rotation = None
        self.translation = None
        self.camera_position = None

    def set_focal_length(self, focal_length_px):
        """
        Update the estimated focal length.
        """
        self.focal_length_px = float(focal_length_px)

        self.K = np.array([
            [self.focal_length_px, 0, self.cx],
            [0, self.focal_length_px, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def calculate_homography(
        self,
        image_points,
        world_points
    ):
        """
        Calculate image -> ground-plane homography.

        image_points:
            [(x1,y1), (x2,y2), ...]

        world_points:
            [(X1,Y1), (X2,Y2), ...]

        At least four correspondences are required.
        """

        if len(image_points) < 4:
            raise ValueError(
                "At least 4 floor-point correspondences are required."
            )

        image_points = np.asarray(
            image_points,
            dtype=np.float64
        )

        world_points = np.asarray(
            world_points,
            dtype=np.float64
        )

        if image_points.shape[0] != world_points.shape[0]:
            raise ValueError(
                "Image and world point counts must match."
            )

        H, mask = cv2.findHomography(
            image_points,
            world_points,
            method=cv2.RANSAC
        )

        if H is None:
            raise ValueError(
                "Could not calculate floor homography."
            )

        self.homography = H

        return H, mask

    def decompose_homography(
        self,
        image_to_world_homography
    ):
        """
        Recover camera pose from a ground-plane homography.

        The homography supplied here should map:

            image coordinates -> world ground coordinates

        We invert it internally to obtain:

            world ground coordinates -> image coordinates
        """

        H_image_to_world = np.asarray(
            image_to_world_homography,
            dtype=np.float64
        )

        H_world_to_image = np.linalg.inv(
            H_image_to_world
        )

        # Normalize H.
        H_world_to_image /= H_world_to_image[2, 2]

        K_inv = np.linalg.inv(self.K)

        h1 = H_world_to_image[:, 0]
        h2 = H_world_to_image[:, 1]
        h3 = H_world_to_image[:, 2]

        r1_temp = K_inv @ h1
        r2_temp = K_inv @ h2

        scale1 = np.linalg.norm(r1_temp)
        scale2 = np.linalg.norm(r2_temp)

        if scale1 < 1e-9 or scale2 < 1e-9:
            raise ValueError(
                "Invalid camera geometry."
            )

        scale = (scale1 + scale2) / 2.0

        r1 = r1_temp / scale
        r2 = r2_temp / scale

        # Third rotation axis.
        r3 = np.cross(r1, r2)

        R_approx = np.column_stack([
            r1,
            r2,
            r3
        ])

        # Enforce a valid rotation matrix using SVD.
        U, _, Vt = np.linalg.svd(R_approx)

        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # Translation.
        t = (K_inv @ h3) / scale

        # Camera center in world coordinates:
        #
        # C = -R^T t
        #
        camera_center = -R.T @ t

        self.rotation = R
        self.translation = t
        self.camera_position = camera_center

        return {
            "rotation": R,
            "translation": t,
            "camera_position": camera_center
        }

    def estimate_height(
        self,
        image_points,
        world_points
    ):
        """
        Estimate camera height above the ground plane.

        World ground plane is assumed to be:

            Z = 0

        The returned height is therefore:

            abs(camera_position[2])
        """

        H, mask = self.calculate_homography(
            image_points,
            world_points
        )

        pose = self.decompose_homography(H)

        camera_position = pose["camera_position"]

        height = abs(float(camera_position[2]))

        return {
            "height": height,
            "camera_x": float(camera_position[0]),
            "camera_y": float(camera_position[1]),
            "homography": H,
            "inliers": mask
        }


def create_tile_world_points(
    rows,
    columns,
    tile_size_ft
):
    """
    Create real-world coordinates for a rectangular
    grid of floor-tile corners.

    Example:

        rows=2
        columns=2
        tile_size_ft=2

    produces:

        (0,0)
        (2,0)
        (4,0)

        (0,2)
        (2,2)
        (4,2)
    """

    points = []

    for r in range(rows + 1):
        for c in range(columns + 1):
            points.append([
                c * tile_size_ft,
                r * tile_size_ft
            ])

    return points


def distance_from_camera_to_ground_point(
    camera_position,
    ground_point
):
    """
    Calculate straight-line distance from camera
    to a ground point.

    camera_position:
        (X, Y, Z)

    ground_point:
        (X, Y)
    """

    cx, cy, cz = camera_position
    gx, gy = ground_point

    horizontal = math.sqrt(
        (gx - cx) ** 2 +
        (gy - cy) ** 2
    )

    distance_3d = math.sqrt(
        horizontal ** 2 +
        cz ** 2
    )

    return {
        "horizontal_distance": horizontal,
        "distance_3d": distance_3d
    }


if __name__ == "__main__":

    print("=" * 60)
    print("CCTV CAMERA HEIGHT ESTIMATOR")
    print("=" * 60)

    print()
    print("Module loaded successfully.")
    print()
    print("This module estimates:")
    print("  1. Camera position")
    print("  2. Camera height above ground")
    print("  3. Camera-to-ground-point distance")
    print()
    print("IMPORTANT:")
    print("Physical scale must come from a known reference.")
    print("An approximate tile size produces an approximate result.")
    print()