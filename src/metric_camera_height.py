
import json
import argparse
from pathlib import Path


class MetricCameraHeight:

    def __init__(self, scale_factor=None):
        self.scale_factor = scale_factor

    def convert(self, relative_height):
        if self.scale_factor is None:
            return None

        if relative_height is None:
            return None

        return (
            float(relative_height)
            * float(self.scale_factor)
        )

    def calculate(
        self,
        relative_height,
        scale_factor=None
    ):

        if scale_factor is not None:
            self.scale_factor = float(scale_factor)

        metric_height = self.convert(
            relative_height
        )

        return {
            "relative_camera_height":
                relative_height,

            "metric_scale_available":
                self.scale_factor is not None,

            "scale_factor_m_per_relative_unit":
                self.scale_factor,

            "camera_height_m":
                metric_height,

            "status":
                (
                    "metric_height_available"
                    if metric_height is not None
                    else "awaiting_verified_metric_scale"
                )
        }


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


def main():

    parser = argparse.ArgumentParser(
        description="Step 10C - Metric Camera Height"
    )

    parser.add_argument(
        "--geometry",
        default="results/robust_geometry.json"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Verified metres per relative unit"
    )

    parser.add_argument(
        "--output",
        default="results/metric_camera_height.json"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STEP 10C — METRIC CAMERA HEIGHT")
    print("=" * 70)

    geometry_path = Path(args.geometry)

    if not geometry_path.exists():

        print(
            f"ERROR: Geometry report not found: "
            f"{geometry_path}"
        )

        return

    geometry = load_json(
        geometry_path
    )

    relative_height = geometry.get(
        "normalized_camera_height"
    )

    if relative_height is None:

        print(
            "ERROR: No normalized camera height "
            "found in geometry report."
        )

        return

    converter = MetricCameraHeight(
        args.scale
    )

    result = converter.calculate(
        relative_height
    )

    result["step"] = "10C"
    result["name"] = "Metric Camera Height"
    result["source_geometry"] = str(
        geometry_path
    )

    if result["camera_height_m"] is None:

        result["note"] = (
            "Relative camera-height geometry "
            "is available. Metric height will "
            "be calculated when a verified "
            "metric scale factor is supplied."
        )

    save_json(
        args.output,
        result
    )

    print(
        f"Relative camera height : "
        f"{relative_height:.6f}"
    )

    if result["camera_height_m"] is not None:

        print(
            f"Metric camera height   : "
            f"{result['camera_height_m']:.4f} m"
        )

    else:

        print(
            "Metric camera height   : "
            "NOT YET AVAILABLE"
        )

    print()
    print(
        f"Result saved to: "
        f"{args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

