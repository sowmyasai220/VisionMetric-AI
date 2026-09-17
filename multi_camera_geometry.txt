import cv2
import json
import sys
from pathlib import Path

from robust_geometry import analyze


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def discover_videos(folder):
    folder = Path(folder)

    videos = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)

    return sorted(videos)


def select_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None

    # Prefer a frame around the middle of the video.
    frame_number = total // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return frame


def analyze_video(video_path):
    frame = select_frame(video_path)

    if frame is None:
        return {
            "video": video_path.name,
            "status": "failed_to_read_frame"
        }

    result = analyze(frame)

    result["video"] = video_path.name
    result["video_path"] = str(video_path)
    result["status"] = "analyzed"

    return result


def analyze_folder(folder):
    videos = discover_videos(folder)

    results = []

    print("=" * 70)
    print("STEP 10D — MULTI-CAMERA ROBUST GEOMETRY")
    print("=" * 70)

    print(f"Input folder : {folder}")
    print(f"Videos found : {len(videos)}")
    print()

    for index, video in enumerate(videos, start=1):

        print("-" * 70)
        print(f"[{index}/{len(videos)}] Analyzing: {video.name}")

        result = analyze_video(video)
        results.append(result)

        if result.get("status") == "analyzed":

            print(
                f"  Geometry score : "
                f"{result.get('geometry_score')}"
            )

            print(
                f"  Horizon valid  : "
                f"{result.get('horizon_valid')}"
            )

            print(
                f"  Ground VP 1    : "
                f"{result.get('ground_vp_1')}"
            )

            print(
                f"  Ground VP 2    : "
                f"{result.get('ground_vp_2')}"
            )

            print(
                f"  Vertical VP    : "
                f"{result.get('vertical_vp')}"
            )

        else:
            print("  FAILED")

    successful = [
        r for r in results
        if r.get("status") == "analyzed"
    ]

    valid_geometry = [
        r for r in successful
        if r.get("horizon_valid") is True
    ]

    output = {
        "stage": "10D",
        "camera_count": len(videos),
        "successful_cameras": len(successful),
        "valid_geometry_cameras": len(valid_geometry),
        "metric_scale_status": "not_estimated",
        "camera_height_status": "relative_geometry_only",
        "cameras": results,
        "notes": [
            "Geometry is estimated independently for each camera.",
            "No airport-specific dimensions or camera metadata are used.",
            "Camera height remains relative/projective until a valid metric reconstruction is established.",
            "This stage does not fabricate metres."
        ]
    }

    output_path = Path("results") / "multi_camera_geometry.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 70)
    print("STEP 10D COMPLETE")
    print("=" * 70)

    print(f"Cameras analyzed       : {len(videos)}")
    print(f"Successful analyses    : {len(successful)}")
    print(f"Valid ground geometry  : {len(valid_geometry)}")
    print(f"Output                 : {output_path}")
    print("=" * 70)


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: py src\\multi_camera_geometry.py <video_folder>"
        )
        return

    folder = Path(sys.argv[1])

    if not folder.exists():
        print("Folder not found:", folder)
        return

    analyze_folder(folder)


if __name__ == "__main__":
    main()