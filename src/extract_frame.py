import cv2
import os
import sys


def extract_frame(video_path, output_path, frame_number=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video contains no readable frames.")

    if frame_number is None:
        # Choose a frame around the middle.
        frame_number = total_frames // 2

    frame_number = max(
        0,
        min(frame_number, total_frames - 1)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_number}."
        )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        frame
    )

    print("=" * 60)
    print("FRAME EXTRACTED")
    print("=" * 60)
    print(f"Video          : {video_path}")
    print(f"Resolution     : {frame.shape[1]} x {frame.shape[0]}")
    print(f"FPS            : {fps:.2f}")
    print(f"Total frames   : {total_frames}")
    print(f"Selected frame : {frame_number}")
    print(f"Saved to       : {output_path}")
    print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "python src\\extract_frame.py "
            "\"path\\to\\video.mp4\""
        )
        sys.exit(1)

    video_path = sys.argv[1]

    extract_frame(
        video_path,
        "frames/calibration_frame.jpg"
    )