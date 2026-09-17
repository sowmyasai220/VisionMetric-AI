from src.calibration import GroundPlaneCalibrator


def main():
    calibrator = GroundPlaneCalibrator()

    # Simulated CCTV image points
    # These represent four points on the floor.
    calibrator.add_point(100, 100, 0, 0)
    calibrator.add_point(500, 100, 8, 0)
    calibrator.add_point(500, 400, 8, 6)
    calibrator.add_point(100, 400, 0, 6)

    # Calculate the perspective transformation
    calibrator.calculate_homography()

    # Test an image point
    x, y = calibrator.image_to_ground(300, 250)

    print("Image point : (300, 250)")
    print("Ground point:", round(x, 2), "ft,", round(y, 2), "ft")


if __name__ == "__main__":
    from src.calibration import GroundPlaneCalibrator


def main():
    calibrator = GroundPlaneCalibrator()

    # Four known floor points
    calibrator.add_point(100, 100, 0, 0)
    calibrator.add_point(500, 100, 8, 0)
    calibrator.add_point(500, 400, 8, 6)
    calibrator.add_point(100, 400, 0, 6)

    calibrator.calculate_homography()

    # Test several image points
    test_points = [
        (100, 100),
        (500, 100),
        (500, 400),
        (100, 400),
        (300, 250)
    ]

    print("\nGROUND-PLANE CALIBRATION TEST")
    print("-" * 40)

    for image_x, image_y in test_points:
        x, y = calibrator.image_to_ground(image_x, image_y)

        print(
            f"Image ({image_x}, {image_y})"
            f" -> Ground ({x:.2f}, {y:.2f}) ft"
        )


if __name__ == "__main__":
    main()