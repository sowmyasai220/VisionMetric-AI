"""
STEP 10B — METRIC SCALE RECOVERY

Converts relative/projective 3D measurements into metric measurements
when a verified physical reference is available.

No airport dimensions, camera positions, or object sizes are hard-coded.
"""

import json
import math
import argparse
from pathlib import Path


class MetricScaleRecovery:
    def __init__(self):
        self.scale_factor = None
        self.reference_name = None
        self.reference_length_m = None
        self.reference_length_relative = None

    def calculate_scale(
        self,
        reference_length_relative,
        reference_length_m,
        reference_name="reference"
    ):
        """Calculate metres per relative reconstruction unit."""

        if reference_length_relative <= 0:
            raise ValueError("Relative reference length must be greater than 0.")

        if reference_length_m <= 0:
            raise ValueError("Metric reference length must be greater than 0.")

        self.reference_length_relative = float(reference_length_relative)
        self.reference_length_m = float(reference_length_m)
        self.reference_name = reference_name

        self.scale_factor = (
            self.reference_length_m /
            self.reference_length_relative
        )

        return self.scale_factor

    def convert_distance(self, relative_distance):
        """Convert a relative distance to metres."""

        if self.scale_factor is None:
            raise RuntimeError(
                "Metric scale has not been established."
            )

        return float(relative_distance) * self.scale_factor

    def convert_point(self, point):
        """Convert a relative 3D point into metric coordinates."""

        if self.scale_factor is None:
            raise RuntimeError(
                "Metric scale has not been established."
            )

        return [
            float(value) * self.scale_factor
            for value in point
        ]

    def result(self):
        return {
            "metric_scale_available": self.scale_factor is not None,
            "scale_factor_m_per_relative_unit": self.scale_factor,
            "reference_name": self.reference_name,
            "reference_length_relative": self.reference_length_relative,
            "reference_length_m": self.reference_length_m,
        }


def load_step9_report(report_path):
    """Load the existing Step 9 reconstruction report."""

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_relative_distances(report):
    """
    Extract relative camera distances from the Step 9 report.
    Handles the existing report structure without hard-coding
    camera names.
    """

    distances = {}

    def search(obj):
        if isinstance(obj, dict):

            for key, value in obj.items():

                if (
                    "distance" in key.lower()
                    and isinstance(value, (int, float))
                ):
                    if "relative" in key.lower():
                        distances[key] = float(value)

                search(value)

        elif isinstance(obj, list):
            for item in obj:
                search(item)

    search(report)

    return distances


def save_result(output_path, data):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_demo():
    """
    Demonstration using a supplied reference.

    This does NOT represent airport ground truth.
    """

    print("=" * 70)
    print("STEP 10B — METRIC SCALE RECOVERY")
    print("=" * 70)

    recovery = MetricScaleRecovery()

    # Demonstration only.
    # These values are NOT project ground truth.
    relative_reference = 5.0
    physical_reference_m = 2.0

    scale = recovery.calculate_scale(
        relative_reference,
        physical_reference_m,
        "demo_reference"
    )

    print(f"Relative reference : {relative_reference}")
    print(f"Physical reference : {physical_reference_m} m")
    print(f"Scale factor       : {scale:.6f} m / relative unit")

    relative_distance = 16.98082594840898
    metric_distance = recovery.convert_distance(relative_distance)

    print(f"Relative distance  : {relative_distance:.6f}")
    print(f"Metric distance    : {metric_distance:.6f} m")

    print()
    print("STATUS: Scale conversion mechanism working.")
    print("NOTE: Demo values are NOT real CCTV measurements.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Step 10B metric scale recovery"
    )

    parser.add_argument(
        "--report",
        default="results/multi_camera_report.json",
        help="Step 9 report"
    )

    parser.add_argument(
        "--output",
        default="results/metric_scale_recovery.json",
        help="Output JSON"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run scale conversion demonstration"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    print("=" * 70)
    print("STEP 10B — METRIC SCALE RECOVERY")
    print("=" * 70)

    report_path = Path(args.report)

    if not report_path.exists():
        print(f"ERROR: Report not found: {report_path}")
        return

    report = load_step9_report(report_path)
    relative_distances = find_relative_distances(report)

    result = {
        "step": "10B",
        "name": "Metric Scale Recovery",
        "metric_scale_available": False,
        "scale_factor_m_per_relative_unit": None,
        "reference_required": True,
        "relative_distances_found": relative_distances,
        "status": "awaiting_verified_metric_reference",
        "note": (
            "The module is ready to convert relative reconstruction "
            "measurements into metres when a verified physical reference "
            "is available in the deployment environment."
        ),
    }

    save_result(args.output, result)

    print("Step 9 report loaded.")
    print(f"Relative distances found: {len(relative_distances)}")
    print()
    print("Metric scale: NOT YET APPLIED")
    print("Reason: verified metric reference has not been supplied to the module.")
    print()
    print(f"Result saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()