"""
STEP 10 — Metric scale + camera-height estimation

Purpose
-------
Adds an honest metric-scale layer to the multi-camera reconstruction.

Important:
- Two-view reconstruction has an arbitrary scale.
- This module NEVER invents metres.
- If a metric cue is unavailable, it reports relative/projective units.
- An optional human-height prior can provide a conditional metric cue when
  upright people are visible. The prior is explicitly reported as an assumption.
- Camera height is derived from the ground-plane geometry only when a metric
  scale is available; otherwise only relative height is reported.

This is an environment-independent utility. No airport dimensions, tile size,
camera coordinates, camera count, or camera metadata are hard-coded.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ScaleEstimate:
    available: bool
    meters_per_scene_unit: Optional[float]
    source: str
    confidence: float
    uncertainty_m: Optional[float]
    assumption: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class HeightEstimate:
    relative_height: Optional[float]
    metric_height_m: Optional[float]
    confidence: float
    scale_source: str
    uncertainty_m: Optional[float]
    status: str
    notes: Optional[str] = None


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def estimate_scale_from_reference_height(
    reference_height_scene_units: float,
    reference_height_m: float,
    reference_sigma_m: float = 0.10,
    confidence: float = 0.55,
) -> ScaleEstimate:
    """
    Convert one observed vertical reference into metres.

    The reference may come from a visual prior (for example an upright person).
    The physical reference value is NOT claimed to be exact.
    """
    if reference_height_scene_units <= 0 or reference_height_m <= 0:
        return ScaleEstimate(False, None, "none", 0.0, None)

    scale = reference_height_m / reference_height_scene_units
    return ScaleEstimate(
        available=True,
        meters_per_scene_unit=float(scale),
        source="imagery_reference_height_prior",
        confidence=_clip01(confidence),
        uncertainty_m=float(reference_sigma_m),
        assumption=(
            f"Reference height prior = {reference_height_m:.2f} m "
            f"(1-sigma ± {reference_sigma_m:.2f} m)."
        ),
        notes=(
            "Metric scale is conditional on the reference-height prior. "
            "It is not a direct measurement of the scene."
        ),
    )


def scale_relative_height(
    vertical_extent_scene_units: float,
    scale: ScaleEstimate,
) -> Optional[float]:
    if not scale.available or scale.meters_per_scene_unit is None:
        return None
    if vertical_extent_scene_units <= 0:
        return None
    return vertical_extent_scene_units * scale.meters_per_scene_unit


def estimate_camera_height_from_ground_reference(
    camera_ground_height_scene_units: float,
    scale: ScaleEstimate,
    geometry_confidence: float,
) -> HeightEstimate:
    """
    Convert a previously estimated relative camera height to metric height.

    camera_ground_height_scene_units must come from the camera/ground geometry
    stage. This function deliberately does not manufacture that geometry.
    """
    if camera_ground_height_scene_units <= 0:
        return HeightEstimate(
            relative_height=None,
            metric_height_m=None,
            confidence=0.0,
            scale_source=scale.source,
            uncertainty_m=None,
            status="no_relative_height",
            notes="No valid relative camera height was supplied.",
        )

    geometry_confidence = _clip01(geometry_confidence)

    if not scale.available or scale.meters_per_scene_unit is None:
        return HeightEstimate(
            relative_height=float(camera_ground_height_scene_units),
            metric_height_m=None,
            confidence=geometry_confidence,
            scale_source=scale.source,
            uncertainty_m=None,
            status="relative_only",
            notes=(
                "Camera height is recoverable only up to the reconstruction "
                "scale because no metric cue was established."
            ),
        )

    h_m = camera_ground_height_scene_units * scale.meters_per_scene_unit
    sigma = scale.uncertainty_m

    return HeightEstimate(
        relative_height=float(camera_ground_height_scene_units),
        metric_height_m=float(h_m),
        confidence=_clip01(geometry_confidence * scale.confidence),
        scale_source=scale.source,
        uncertainty_m=sigma,
        status="metric",
        notes=(
            "Metric camera height is conditional on the imagery-derived "
            "reference scale and the quality of the ground-plane estimate."
        ),
    )


def estimate_upright_person_scale(
    person_box: Sequence[float],
    observed_height_scene_units: float,
    assumed_person_height_m: float = 1.70,
    person_height_sigma_m: float = 0.10,
) -> ScaleEstimate:
    """
    Build a conditional metric cue from an upright-person prior.

    person_box = [x1, y1, x2, y2]. It is used only as evidence that the
    selected vertical extent belongs to a person. The actual scale conversion
    uses the supplied observed scene-unit height.
    """
    if len(person_box) != 4:
        return ScaleEstimate(False, None, "none", 0.0, None)

    x1, y1, x2, y2 = map(float, person_box)
    if x2 <= x1 or y2 <= y1 or observed_height_scene_units <= 0:
        return ScaleEstimate(False, None, "none", 0.0, None)

    # Conservative confidence: a population-height prior is useful but noisy.
    return estimate_scale_from_reference_height(
        observed_height_scene_units,
        assumed_person_height_m,
        person_height_sigma_m,
        confidence=0.45,
    )


def summarize_scale_and_height(
    scale: ScaleEstimate,
    heights: Sequence[HeightEstimate],
) -> Dict[str, Any]:
    valid_metric = [h for h in heights if h.status == "metric" and h.metric_height_m is not None]
    valid_relative = [h for h in heights if h.relative_height is not None]

    result: Dict[str, Any] = {
        "metric_scale": asdict(scale),
        "camera_height": {
            "status": "metric" if valid_metric else ("relative_only" if valid_relative else "unavailable"),
            "estimates": [asdict(h) for h in heights],
        },
        "scientific_limit": (
            "Absolute metric scale cannot be guaranteed from monocular/two-view "
            "geometry alone. A metric result is reported only when an explicit "
            "imagery-derived reference cue is available."
        ),
    }

    if valid_metric:
        vals = np.array([h.metric_height_m for h in valid_metric], dtype=float)
        result["camera_height"]["median_metric_height_m"] = float(np.median(vals))

    if valid_relative:
        vals = np.array([h.relative_height for h in valid_relative], dtype=float)
        result["camera_height"]["median_relative_height"] = float(np.median(vals))

    return result


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read-only inspection of the existing Step 9 report.

    It does not modify or assume camera metadata.
    """
    cameras = report.get("cameras", [])
    pairs = report.get("camera_pairs", report.get("pairs", []))

    return {
        "camera_count": len(cameras) if isinstance(cameras, list) else 0,
        "pair_count": len(pairs) if isinstance(pairs, list) else 0,
        "metric_scale_from_step9": report.get("metric_status", "not_available"),
        "note": (
            "Step 9 reconstruction is relative/projective unless a separate "
            "metric cue is established."
        ),
    }


def save_json(data: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 10: honest metric-scale and camera-height layer"
    )
    parser.add_argument(
        "--report",
        default="results/multi_camera_report.json",
        help="Step 9 JSON report",
    )
    parser.add_argument(
        "--output",
        default="results/step10_metric_height.json",
        help="Output JSON report",
    )
    parser.add_argument(
        "--person-height",
        type=float,
        default=None,
        help=(
            "Optional visual-prior height in metres for an upright person. "
            "If omitted, no metric prior is assumed."
        ),
    )
    parser.add_argument(
        "--person-height-sigma",
        type=float,
        default=0.10,
        help="Uncertainty of the optional person-height prior in metres",
    )
    parser.add_argument(
        "--observed-person-height",
        type=float,
        default=None,
        help="Observed person vertical extent in current scene/reconstruction units",
    )
    parser.add_argument(
        "--relative-camera-height",
        type=float,
        default=None,
        help="Relative camera height produced by a valid ground-geometry stage",
    )
    parser.add_argument(
        "--geometry-confidence",
        type=float,
        default=0.0,
        help="Confidence of the supplied ground/camera geometry estimate",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STEP 10 — METRIC SCALE + CAMERA HEIGHT")
    print("=" * 70)

    try:
        report = load_report(args.report)
        print(f"Step 9 report : {args.report}")
        print(json.dumps(inspect_report(report), indent=2))
    except Exception as e:
        print(f"Could not read report: {e}")
        report = {}

    if (
        args.person_height is not None
        and args.observed_person_height is not None
    ):
        scale = estimate_scale_from_reference_height(
            args.observed_person_height,
            args.person_height,
            args.person_height_sigma,
        )
        print("\nMetric cue : AVAILABLE")
        print(f"Source     : {scale.source}")
        print(f"Scale      : {scale.meters_per_scene_unit:.8f} m / scene unit")
        print(f"Confidence : {scale.confidence:.2f}")
        print(f"Assumption : {scale.assumption}")
    else:
        scale = ScaleEstimate(
            available=False,
            meters_per_scene_unit=None,
            source="none",
            confidence=0.0,
            uncertainty_m=None,
            notes=(
                "No imagery-derived metric reference was supplied. "
                "Relative geometry is retained; metres are not invented."
            ),
        )
        print("\nMetric cue : NOT AVAILABLE")
        print("Metric output will NOT be fabricated.")

    if args.relative_camera_height is not None:
        height = estimate_camera_height_from_ground_reference(
            args.relative_camera_height,
            scale,
            args.geometry_confidence,
        )
        print("\nCamera height:")
        print(f"Status     : {height.status}")
        print(f"Relative   : {height.relative_height}")
        print(f"Metric     : {height.metric_height_m}")
        print(f"Confidence : {height.confidence:.2f}")
    else:
        height = HeightEstimate(
            relative_height=None,
            metric_height_m=None,
            confidence=0.0,
            scale_source=scale.source,
            uncertainty_m=None,
            status="awaiting_ground_geometry",
            notes=(
                "A physical camera-height estimate requires a valid "
                "ground-plane/camera-height geometry result."
            ),
        )
        print("\nCamera height: AWAITING GROUND GEOMETRY")

    result = {
        "step": 10,
        "environment_independent": True,
        "metric_scale": asdict(scale),
        "camera_height": asdict(height),
        "limitations": [
            "Pure projective two-view reconstruction has arbitrary scale.",
            "Metric distance is conditional on a metric cue visible in the imagery.",
            "A person-height prior is an assumption, not a direct measurement.",
            "No airport dimensions, tile size, camera coordinates, or camera metadata are used.",
        ],
    }

    save_json(result, args.output)
    print(f"\nReport saved to : {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
