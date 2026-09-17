import json
import os


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "camera_config.json"
)


def load_camera_config():
    """
    Load camera calibration configuration.
    """

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_camera_config(camera_id):
    """
    Get the configuration for one camera.
    """

    config = load_camera_config()

    cameras = config.get("cameras", {})

    if camera_id not in cameras:
        raise ValueError(
            f"Camera '{camera_id}' not found in configuration."
        )

    return cameras[camera_id]


if __name__ == "__main__":
    config = load_camera_config()

    print("Camera configuration loaded.")
    print("Available cameras:")

    for camera_id in config["cameras"]:
        print("-", camera_id)