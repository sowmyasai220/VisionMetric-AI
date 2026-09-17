"""
Multi-Camera Distance Results

Combines camera/object distance results from multiple cameras
into one structured result.

Environment-independent:
- No fixed number of cameras
- No fixed camera names
- No fixed scene dimensions
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class CameraResult:
    camera_id: str
    horizontal_distance: Optional[float] = None
    vertical_difference: Optional[float] = None
    straight_line_distance: Optional[float] = None
    unit: str = "scene units"
    confidence: float = 0.0
    valid: bool = False
    message: str = ""


class MultiCameraResults:

    def __init__(self):
        self.results: List[CameraResult] = []

    def add_result(
        self,
        camera_id: str,
        horizontal_distance: Optional[float] = None,
        vertical_difference: Optional[float] = None,
        straight_line_distance: Optional[float] = None,
        unit: str = "scene units",
        confidence: float = 0.0,
        valid: bool = True,
        message: str = "",
    ):

        result = CameraResult(
            camera_id=camera_id,
            horizontal_distance=horizontal_distance,
            vertical_difference=vertical_difference,
            straight_line_distance=straight_line_distance,
            unit=unit,
            confidence=confidence,
            valid=valid,
            message=message,
        )

        self.results.append(result)

    def valid_results(self) -> List[CameraResult]:

        return [
            result
            for result in self.results
            if result.valid
        ]

    def closest_camera(
        self,
    ) -> Optional[CameraResult]:

        valid = [
            result
            for result in self.results
            if (
                result.valid
                and
                result.straight_line_distance is not None
            )
        ]

        if not valid:
            return None

        return min(
            valid,
            key=lambda result: result.straight_line_distance
        )

    def sort_by_distance(self) -> List[CameraResult]:

        valid = [
            result
            for result in self.results
            if (
                result.valid
                and
                result.straight_line_distance is not None
            )
        ]

        return sorted(
            valid,
            key=lambda result: result.straight_line_distance
        )

    def summary(self):

        closest = self.closest_camera()

        return {
            "camera_count": len(self.results),
            "valid_camera_count": len(self.valid_results()),
            "closest_camera": (
                closest.camera_id
                if closest is not None
                else None
            ),
        }

    def to_dict(self):

        return {
            "summary": self.summary(),
            "cameras": [
                asdict(result)
                for result in self.results
            ],
        }

    def save_json(self, output_path: str):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.to_dict(),
                file,
                indent=4
            )

    def print_results(self):

        print()
        print("=" * 70)
        print("MULTI-CAMERA DISTANCE RESULTS")
        print("=" * 70)

        for result in self.sort_by_distance():

            print(
                f"{result.camera_id}: "
                f"{result.straight_line_distance:.3f} "
                f"{result.unit} "
                f"(confidence={result.confidence:.2f})"
            )

        print()

        closest = self.closest_camera()

        if closest:

            print(
                f"Closest camera: "
                f"{closest.camera_id}"
            )

            print(
                f"Distance: "
                f"{closest.straight_line_distance:.3f} "
                f"{closest.unit}"
            )

        else:

            print("No valid camera results.")

        print("=" * 70)


def run_demo():

    results = MultiCameraResults()

    # Synthetic test data only.
    # These values are NOT real CCTV measurements.

    results.add_result(
        camera_id="camera_A",
        horizontal_distance=12.0,
        vertical_difference=5.0,
        straight_line_distance=13.0,
        confidence=0.92,
        valid=True,
    )

    results.add_result(
        camera_id="camera_B",
        horizontal_distance=8.0,
        vertical_difference=6.0,
        straight_line_distance=10.0,
        confidence=0.95,
        valid=True,
    )

    results.add_result(
        camera_id="camera_C",
        horizontal_distance=15.0,
        vertical_difference=8.0,
        straight_line_distance=17.0,
        confidence=0.88,
        valid=True,
    )

    results.print_results()

    closest = results.closest_camera()

    print()
    print("Expected closest camera: camera_B")

    if closest and closest.camera_id == "camera_B":

        print("Validation : PASSED")

    else:

        print("Validation : FAILED")

    print()
    print("JSON summary:")
    print(
        json.dumps(
            results.to_dict(),
            indent=4
        )
    )


if __name__ == "__main__":
    run_demo()