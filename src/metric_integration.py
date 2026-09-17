import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

STEP9_REPORT = RESULTS_DIR / "multi_camera_report.json"
SCALE_REPORT = RESULTS_DIR / "visual_scale_result.json"
OUTPUT_REPORT = RESULTS_DIR / "metric_integration.json"


def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_relative_distances(step9):
    distances = []

    if not step9:
        return distances

    pairs = step9.get("cross_camera_pairs", [])

    for pair in pairs:
        triangulation = pair.get("triangulation", {})

        camera_a = pair.get("camera_a")
        camera_b = pair.get("camera_b")

        distance_a = triangulation.get(
            "camera_a_distance_relative"
        )

        distance_b = triangulation.get(
            "camera_b_distance_relative"
        )

        if camera_a and distance_a is not None:
            distances.append({
                "camera": camera_a,
                "relative_distance": float(distance_a)
            })

        if camera_b and distance_b is not None:
            distances.append({
                "camera": camera_b,
                "relative_distance": float(distance_b)
            })

    return distances


def get_scale(scale_report):
    if not scale_report:
        return None

    scale = scale_report.get(
        "scale_m_per_relative_unit"
    )

    if scale is None:
        scale = scale_report.get("scale")

    if scale is None:
        return None

    try:
        scale = float(scale)

        if scale <= 0:
            return None

        return scale

    except (TypeError, ValueError):
        return None


def convert_distances(distances, scale):
    results = []

    for item in distances:
        relative = item["relative_distance"]

        metric = relative * scale

        results.append({
            "camera": item["camera"],
            "relative_distance": relative,
            "metric_distance_m": round(metric, 3)
        })

    return results


def save_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )


def run():
    print("=" * 70)
    print("STEP 12 — FINAL METRIC DISTANCE INTEGRATION")
    print("=" * 70)

    print(
        f"\nStep 9 report : {STEP9_REPORT}"
    )

    print(
        f"Scale report  : {SCALE_REPORT}"
    )

    step9 = load_json(STEP9_REPORT)
    scale_report = load_json(SCALE_REPORT)

    if step9 is None:
        print("\nERROR: Step 9 report not found.")
        return False

    distances = find_relative_distances(step9)

    print(
        f"\nRelative distances found : {len(distances)}"
    )

    for item in distances:
        print(
            f"  {item['camera']} : "
            f"{item['relative_distance']:.6f} "
            f"relative units"
        )

    scale = get_scale(scale_report)

    if scale is None:
        print(
            "\nMetric scale             : NOT AVAILABLE"
        )

        print(
            "Metric distances         : NOT REPORTED"
        )

        metric_results = []

        status = (
            "Relative distance extraction successful; "
            "metric scale unavailable."
        )

    else:
        print(
            f"\nMetric scale             : "
            f"{scale:.6f} m / relative unit"
        )

        metric_results = convert_distances(
            distances,
            scale
        )

        print("\nMetric distances:")

        for item in metric_results:
            print(
                f"  {item['camera']} : "
                f"{item['metric_distance_m']:.3f} m"
            )

        status = (
            "Metric distance conversion successful."
        )

    report = {
        "step": "12",
        "name": "Final Metric Distance Integration",

        "relative_distances": distances,

        "metric_scale": {
            "available": scale is not None,
            "m_per_relative_unit": scale
        },

        "metric_distances": metric_results,

        "status": status,

        "source_reports": {
            "step9": str(STEP9_REPORT),
            "visual_scale": str(SCALE_REPORT)
        },

        "note": (
            "Metric distances are obtained by applying the "
            "visual scale recovered in Step 10F to the relative "
            "multi-camera distances from Step 9."
        )
    }

    save_report(report)

    print(
        f"\nResult saved to: {OUTPUT_REPORT}"
    )

    print("\n" + "=" * 70)

    if scale is not None:
        print(
            "STEP 12 COMPLETE — METRIC DISTANCES AVAILABLE"
        )
    else:
        print(
            "STEP 12 COMPLETE — WAITING FOR METRIC SCALE"
        )

    print("=" * 70)

    return True


def demo():
    print("=" * 70)
    print("STEP 12 — METRIC INTEGRATION DEMO")
    print("=" * 70)

    relative_distances = [
        16.980826,
        16.197583
    ]

    scale = 0.905153

    metric_distances = [
        d * scale
        for d in relative_distances
    ]

    print(
        f"\nScale : {scale:.6f} m / relative unit"
    )

    print("\nMetric distances:")

    for index, distance in enumerate(
        metric_distances,
        start=1
    ):
        print(
            f"  Camera {index}: {distance:.3f} m"
        )

    print(
        "\nNOTE: Demo calculation only."
    )

    print("=" * 70)


if __name__ == "__main__":
    run()