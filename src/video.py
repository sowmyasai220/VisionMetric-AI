import cv2
import os


def get_video_info(video_path):
    """
    Get basic information about a video.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def extract_frame(
    video_path,
    output_path,
    frame_number=0
):
    """
    Extract one frame from a video.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise ValueError(
            f"Could not read frame {frame_number}"
        )

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    cv2.imwrite(
        output_path,
        frame
    )

    return output_path


def sample_frames(
    video_path,
    output_folder,
    interval_seconds=5
):
    """
    Extract one frame every few seconds
    from a CCTV video.
    """

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise ValueError(
            "Could not determine video FPS."
        )

    frame_interval = max(
        1,
        int(fps * interval_seconds)
    )

    frame_number = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            output_path = os.path.join(
                output_folder,
                f"frame_{saved_count:04d}.jpg"
            )

            cv2.imwrite(
                output_path,
                frame
            )

            saved_count += 1

        frame_number += 1

    cap.release()

    return saved_count


if __name__ == "__main__":

    print("=" * 60)
    print("VIDEO PROCESSING MODULE")
    print("=" * 60)
    print("Ready for CCTV footage.")