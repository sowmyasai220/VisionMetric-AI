import cv2
import numpy as np
import json
import os


IMAGE_PATH = "frames/calibration_frame.jpg"

# Provisional scale only.
# Change this later if we discover the actual tile size.
TILE_SIZE_FT = 2.0

# Number of floor points we want.
REQUIRED_POINTS = 4


clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        print(
            f"Point {len(clicked_points)}: "
            f"image=({x}, {y})"
        )


def create_world_points():
    """
    Four points representing a rectangular ground area.

    We assign:
        top-left     -> (0, 0)
        top-right    -> (tile, 0)
        bottom-right -> (tile, tile)
        bottom-left  -> (0, tile)

    IMPORTANT:
    This assumes the four selected points correspond to
    four corners of a rectangular floor region.
    """

    s = TILE_SIZE_FT

    return np.float32([
        [0, 0],
        [s, 0],
        [s, s],
        [0, s]
    ])


def main():

    if not os.path.exists(IMAGE_PATH):
        print()
        print("ERROR: Calibration frame not found.")
        print()
        print("Expected:")
        print(IMAGE_PATH)
        print()
        print("Create the frame first.")
        return

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("ERROR: Could not read image.")
        return

    display = image.copy()

    cv2.namedWindow("CCTV Calibration")
    cv2.setMouseCallback(
        "CCTV Calibration",
        mouse_callback
    )

    print()
    print("=" * 60)
    print("CCTV FLOOR CALIBRATION")
    print("=" * 60)
    print()
    print("Click FOUR floor points.")
    print()
    print("Choose four corners of a rectangular floor region.")
    print()
    print("Order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print()
    print("Press R to reset.")
    print("Press ENTER when finished.")
    print("Press ESC to cancel.")
    print()
    print(
        f"Current assumed tile/scale = "
        f"{TILE_SIZE_FT} ft"
    )
    print()

    while True:

        display = image.copy()

        for i, (x, y) in enumerate(clicked_points):

            cv2.circle(
                display,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        if len(clicked_points) >= 2:
            for i in range(len(clicked_points) - 1):
                cv2.line(
                    display,
                    clicked_points[i],
                    clicked_points[i + 1],
                    (255, 0, 0),
                    2
                )

        if len(clicked_points) == 4:
            cv2.line(
                display,
                clicked_points[3],
                clicked_points[0],
                (255, 0, 0),
                2
            )

        cv2.imshow(
            "CCTV Calibration",
            display
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("Cancelled.")
            break

        if key == ord("r"):
            clicked_points.clear()
            print("Points reset.")

        if key == 13:

            if len(clicked_points) != REQUIRED_POINTS:
                print(
                    f"You selected {len(clicked_points)} points."
                )
                print("Exactly 4 points are required.")
                continue

            image_points = np.float32(
                clicked_points
            )

            world_points = create_world_points()

            H, mask = cv2.findHomography(
                image_points,
                world_points
            )

            if H is None:
                print(
                    "ERROR: Could not calculate homography."
                )
                continue

            os.makedirs(
                "calibration",
                exist_ok=True
            )

            output = {
                "tile_size_ft": TILE_SIZE_FT,
                "image_points": [
                    list(map(int, p))
                    for p in clicked_points
                ],
                "world_points": [
                    list(map(float, p))
                    for p in world_points
                ],
                "homography": H.tolist()
            }

            with open(
                "calibration/calibration_data.json",
                "w"
            ) as f:
                json.dump(
                    output,
                    f,
                    indent=4
                )

            print()
            print("=" * 60)
            print("CALIBRATION SAVED")
            print("=" * 60)
            print()
            print(
                "File:"
                " calibration/calibration_data.json"
            )
            print()
            print("Image points:")
            print(image_points)
            print()
            print("World points:")
            print(world_points)
            print()
            print("Homography:")
            print(H)
            print()

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()