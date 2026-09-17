import cv2
import numpy as np
class CameraCalibration:
    """
    Stores the physical calibration information
    for one CCTV camera.
    """

    def __init__(
        self,
        camera_id,
        camera_height_ft,
        camera_x_ft=0.0,
        camera_y_ft=0.0
    ):
        self.camera_id = camera_id
        self.camera_height_ft = camera_height_ft
        self.camera_x_ft = camera_x_ft
        self.camera_y_ft = camera_y_ft

    def display(self):
        print(f"Camera ID      : {self.camera_id}")
        print(f"Camera height  : {self.camera_height_ft:.2f} ft")
        print(f"Ground X       : {self.camera_x_ft:.2f} ft")
        print(f"Ground Y       : {self.camera_y_ft:.2f} ft")

class GroundPlaneCalibrator:
    """
    Converts points from a CCTV image into real-world
    ground-plane coordinates using a perspective transform.
    """

    def __init__(self):
        self.image_points = []
        self.world_points = []
        self.homography = None

    def add_point(self, image_x, image_y, world_x, world_y):
        """
        Add a correspondence between an image point and
        its real-world ground-plane coordinate.
        """

        self.image_points.append([image_x, image_y])
        self.world_points.append([world_x, world_y])

    def calculate_homography(self):
        """
        Calculate the perspective transformation.

        At least four corresponding points are required.
        """

        if len(self.image_points) < 4:
            raise ValueError(
                "At least 4 point correspondences are required."
            )

        image_points = np.float32(self.image_points)
        world_points = np.float32(self.world_points)

        self.homography, _ = cv2.findHomography(
            image_points,
            world_points
        )

        if self.homography is None:
            raise ValueError(
                "Could not calculate homography."
            )

        return self.homography

    def image_to_ground(self, image_x, image_y):
        """
        Convert one image point into real-world ground coordinates.
        """

        if self.homography is None:
            raise ValueError(
                "Calculate homography before converting points."
            )

        point = np.float32(
            [[[image_x, image_y]]]
        )

        ground_point = cv2.perspectiveTransform(
            point,
            self.homography
        )

        world_x, world_y = ground_point[0][0]

        return float(world_x), float(world_y)


if __name__ == "__main__":
    print("Ground-plane calibration module")
    print("Ready for CCTV calibration.")