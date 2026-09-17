"""
Metric Scale Recovery
---------------------

Attempts to recover a physical scale from CCTV imagery.

Important:
Pure monocular/projective geometry has an inherent global-scale
ambiguity. Therefore this module NEVER invents a physical scale.

It supports:
1. Visual scale cues supplied from the image
2. Optional known reference measurements
3. Explicit reporting of whether metric scale is reliable

The rest of the geometry pipeline can operate in normalized units
until a trustworthy metric scale is available.
"""

from dataclasses import dataclass
from typing import Optional, List
import math


@dataclass
class ScaleResult:
    scale_available: bool
    scale_m_per_unit: Optional[float]
    source: str
    confidence: float
    uncertainty_percent: Optional[float]
    message: str

    def to_dict(self):
        return {
            "scale_available": self.scale_available,
            "scale_m_per_unit": self.scale_m_per_unit,
            "source": self.source,
            "confidence": self.confidence,
            "uncertainty_percent": self.uncertainty_percent,
            "message": self.message,
        }


class MetricScaleRecovery:
    """
    Handles conversion from normalized/projective scene units
    to physical metric units.

    The class deliberately refuses to create a fake metric scale.
    """

    def __init__(self):
        self.scale_sources = []

    # ---------------------------------------------------------
    # ADD A VISUAL REFERENCE
    # ---------------------------------------------------------

    def add_reference(
        self,
        scene_distance: float,
        physical_distance_m: float,
        source: str = "visual_reference",
        uncertainty_percent: float = 10.0,
    ):
        """
        Add a known relationship between scene units and meters.

        Example:

            scene_distance = 5.2
            physical_distance_m = 2.0

        means:

            5.2 scene units = 2 meters
        """

        if scene_distance <= 0:
            raise ValueError("Scene distance must be positive.")

        if physical_distance_m <= 0:
            raise ValueError("Physical distance must be positive.")

        scale = physical_distance_m / scene_distance

        confidence = max(
            0.0,
            min(
                1.0,
                1.0 - uncertainty_percent / 100.0
            )
        )

        self.scale_sources.append({
            "scale": scale,
            "source": source,
            "confidence": confidence,
            "uncertainty_percent": uncertainty_percent,
        })

    # ---------------------------------------------------------
    # COMBINE MULTIPLE REFERENCES
    # ---------------------------------------------------------

    def estimate_from_references(self) -> ScaleResult:

        if not self.scale_sources:
            return ScaleResult(
                scale_available=False,
                scale_m_per_unit=None,
                source="none",
                confidence=0.0,
                uncertainty_percent=None,
                message=(
                    "No trustworthy metric reference was found. "
                    "Results must remain in normalized scene units."
                ),
            )

        weighted_sum = 0.0
        weight_total = 0.0

        for item in self.scale_sources:

            weight = max(item["confidence"], 0.01)

            weighted_sum += item["scale"] * weight
            weight_total += weight

        scale = weighted_sum / weight_total

        confidence = sum(
            item["confidence"]
            for item in self.scale_sources
        ) / len(self.scale_sources)

        uncertainty = (
            sum(
                item["uncertainty_percent"]
                for item in self.scale_sources
            )
            / len(self.scale_sources)
        )

        sources = ", ".join(
            item["source"]
            for item in self.scale_sources
        )

        return ScaleResult(
            scale_available=True,
            scale_m_per_unit=scale,
            source=sources,
            confidence=confidence,
            uncertainty_percent=uncertainty,
            message=(
                "Metric scale recovered from visual/reference "
                "measurements."
            ),
        )

    # ---------------------------------------------------------
    # SCALE A NORMALIZED DISTANCE
    # ---------------------------------------------------------

    def convert_to_meters(
        self,
        normalized_distance: float
    ) -> Optional[float]:

        result = self.estimate_from_references()

        if not result.scale_available:
            return None

        return normalized_distance * result.scale_m_per_unit

    # ---------------------------------------------------------
    # SCALE A LIST OF DISTANCES
    # ---------------------------------------------------------

    def convert_distances(
        self,
        distances: List[float]
    ):

        result = self.estimate_from_references()

        if not result.scale_available:
            return None

        return [
            d * result.scale_m_per_unit
            for d in distances
        ]

    # ---------------------------------------------------------
    # CHECK WHETHER METRIC SCALE IS SAFE
    # ---------------------------------------------------------

    def metric_status(self):

        result = self.estimate_from_references()

        print()
        print("=" * 60)
        print("METRIC SCALE STATUS")
        print("=" * 60)

        print(
            f"Metric scale available : "
            f"{result.scale_available}"
        )

        print(
            f"Scale (m / scene unit)  : "
            f"{result.scale_m_per_unit}"
        )

        print(
            f"Source                  : "
            f"{result.source}"
        )

        print(
            f"Confidence              : "
            f"{result.confidence:.2f}"
        )

        print(
            f"Uncertainty             : "
            f"{result.uncertainty_percent}%"
        )

        print()
        print(result.message)
        print("=" * 60)

        return result


# =============================================================
# AUTOMATIC SCALE-CUE ANALYSIS
# =============================================================

def analyze_scale_cues(
    image_width: int,
    image_height: int,
    horizon_y: Optional[float] = None,
):
    """
    Analyze whether the image geometry contains useful
    information for scale recovery.

    This function does NOT fabricate metric units.

    It provides diagnostic information that later modules
    can use when identifying repeated physical structures.
    """

    result = {
        "image_width": image_width,
        "image_height": image_height,
        "horizon_detected": horizon_y is not None,
        "horizon_y": horizon_y,
        "possible_scale_cues": [],
    }

    if horizon_y is not None:

        lower_region = image_height - horizon_y

        if lower_region > image_height * 0.25:
            result["possible_scale_cues"].append(
                "ground_plane"
            )

    result["possible_scale_cues"].extend([
        "parallel_floor_lines",
        "repeated_scene_geometry",
        "vertical_structures",
        "cross_camera_correspondence",
    ])

    return result


# =============================================================
# DIAGNOSTIC
# =============================================================

def print_scale_diagnostics(data):

    print()
    print("=" * 60)
    print("SCALE-CUE DIAGNOSTICS")
    print("=" * 60)

    print(
        f"Image size: "
        f"{data['image_width']} x {data['image_height']}"
    )

    print(
        f"Horizon detected: "
        f"{data['horizon_detected']}"
    )

    print("Potential visual cues:")

    for cue in data["possible_scale_cues"]:
        print(f"  - {cue}")

    print()
    print(
        "NOTE: Visual geometry alone may determine scene "
        "structure while absolute meters/feet can remain "
        "ambiguous."
    )

    print("=" * 60)


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("METRIC SCALE RECOVERY MODULE")
    print("=" * 60)

    recovery = MetricScaleRecovery()

    diagnostics = analyze_scale_cues(
        image_width=832,
        image_height=464,
        horizon_y=220,
    )

    print_scale_diagnostics(diagnostics)

    print()
    print("Testing without an external metric reference:")

    result = recovery.metric_status()

    print()
    print("Module ready.")