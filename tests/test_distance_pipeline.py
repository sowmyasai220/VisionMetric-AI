from src.calibration import GroundPlaneCalibrator
from src.distance import calculate_distance_from_image_point


def main():

    # -----------------------------------------
    # 1. Create ground-plane calibration
    # -----------------------------------------

    calibrator = GroundPlaneCalibrator()

    # Simulated floor reference points
    calibrator.add_point(100, 100, 0, 0)
    calibrator.add_point(500, 100, 8, 0)
    calibrator.add_point(500, 400, 8, 6)
    calibrator.add_point(100, 400, 0, 6)

    calibrator.calculate_homography()

    # -----------------------------------------
    # 2. Simulated camera information
    # -----------------------------------------

    camera_x = 0.0
    camera_y = 0.0
    camera_height = 8.0

    # -----------------------------------------
    # 3. Simulated object image position
    # -----------------------------------------

    image_x = 300
    image_y = 250

    # -----------------------------------------
    # 4. Calculate distance
    # -----------------------------------------

    result = calculate_distance_from_image_point(
        calibrator,
        image_x,
        image_y,
        camera_x,
        camera_y,
        camera_height
    )

    # -----------------------------------------
    # 5. Display result
    # -----------------------------------------

    print("\nCOMPLETE DISTANCE PIPELINE TEST")
    print("-" * 45)

    print(
        f"Object image point : "
        f"({image_x}, {image_y})"
    )

    print(
        f"Object ground X    : "
        f"{result['object_x']:.2f} ft"
    )

    print(
        f"Object ground Y    : "
        f"{result['object_y']:.2f} ft"
    )

    print(
        f"Horizontal distance: "
        f"{result['horizontal_distance']:.2f} ft"
    )

    print(
        f"3D camera distance : "
        f"{result['distance_3d']:.2f} ft"
    )


if __name__ == "__main__":
    main()