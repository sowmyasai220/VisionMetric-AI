"""
Camera-to-Object Distance Calculation

Calculates:
- Horizontal ground distance
- Vertical difference
- Straight-line 3D distance

The module does not assume a particular environment.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DistanceResult:
    horizontal_distance: float
    vertical_difference: float
    straight_line_distance: float
    unit: str
    confidence: float


def euclidean_distance_3d(
    camera_position: Tuple[float, float, float],
    object_position: Tuple[float, float, float],
) -> float:

    dx = object_position[0] - camera_position[0]
    dy = object_position[1] - camera_position[1]
    dz = object_position[2] - camera_position[2]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def horizontal_distance(
    camera_position: Tuple[float, float, float],
    object_position: Tuple[float, float, float],
) -> float:

    dx = object_position[0] - camera_position[0]
    dy = object_position[1] - camera_position[1]

    return math.sqrt(dx * dx + dy * dy)


def calculate_distance(
    camera_position: Tuple[float, float, float],
    object_position: Tuple[float, float, float],
    unit: str = "scene units",
    confidence: float = 0.0,
) -> DistanceResult:

    horizontal = horizontal_distance(
        camera_position,
        object_position
    )

    vertical = abs(
        object_position[2] - camera_position[2]
    )

    straight = euclidean_distance_3d(
        camera_position,
        object_position
    )

    return DistanceResult(
        horizontal_distance=horizontal,
        vertical_difference=vertical,
        straight_line_distance=straight,
        unit=unit,
        confidence=confidence,
    )


def convert_distance(
    distance: float,
    scale: float,
) -> float:

    if scale <= 0:
        raise ValueError("Scale must be greater than zero.")

    return distance * scale


def calculate_metric_distance(
    camera_position: Tuple[float, float, float],
    object_position: Tuple[float, float, float],
    scale: Optional[float] = None,
    target_unit: str = "meters",
    confidence: float = 0.0,
) -> DistanceResult:

    result = calculate_distance(
        camera_position,
        object_position,
        unit="scene units",
        confidence=confidence,
    )

    if scale is not None:

        result.horizontal_distance = convert_distance(
            result.horizontal_distance,
            scale
        )

        result.vertical_difference = convert_distance(
            result.vertical_difference,
            scale
        )

        result.straight_line_distance = convert_distance(
            result.straight_line_distance,
            scale
        )

        result.unit = target_unit

    return result


def print_result(result: DistanceResult):

    print()
    print("=" * 60)
    print("CAMERA -> OBJECT DISTANCE")
    print("=" * 60)

    print(
        f"Horizontal distance : "
        f"{result.horizontal_distance:.4f} {result.unit}"
    )

    print(
        f"Vertical difference : "
        f"{result.vertical_difference:.4f} {result.unit}"
    )

    print(
        f"Straight-line distance : "
        f"{result.straight_line_distance:.4f} {result.unit}"
    )

    print(
        f"Confidence : "
        f"{result.confidence:.3f}"
    )

    print("=" * 60)


def run_demo():

    camera = (0.0, 0.0, 5.0)

    obj = (3.0, 4.0, 0.0)

    result = calculate_distance(
        camera,
        obj,
        confidence=1.0
    )

    print("=" * 60)
    print("DISTANCE MODULE")
    print("=" * 60)

    print("Camera position :", camera)
    print("Object position :", obj)

    print_result(result)

    expected_horizontal = 5.0
    expected_vertical = 5.0
    expected_3d = math.sqrt(50)

    print()
    print("Expected:")
    print(f"Horizontal : {expected_horizontal:.4f}")
    print(f"Vertical   : {expected_vertical:.4f}")
    print(f"3D         : {expected_3d:.4f}")

    tolerance = 1e-6

    valid = (
        abs(
            result.horizontal_distance
            - expected_horizontal
        ) < tolerance
        and
        abs(
            result.vertical_difference
            - expected_vertical
        ) < tolerance
        and
        abs(
            result.straight_line_distance
            - expected_3d
        ) < tolerance
    )

    print()
    print("Validation :", "PASSED" if valid else "FAILED")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()