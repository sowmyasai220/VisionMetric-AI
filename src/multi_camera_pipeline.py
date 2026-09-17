"""
MULTI-CAMERA CCTV TARGET-POINT PIPELINE

Extension of the existing multi-camera geometry pipeline:
- accepts a user-selected point in Camera 01
- finds the corresponding visual point in the other camera views
- triangulates that target point using the recovered relative camera pose
- reports target-specific camera-to-point distances
- keeps metric scale separate from relative geometry

No fixed camera count, camera names, object class, airport dimensions,
tile size, camera metadata, or hard-coded distances are used.
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

try:
    from .geometry_pipeline import discover_videos, analyze_video, read_frame
    from .common_object_matching import match_images
    from .self_calibration import self_calibrate
    from .cross_camera_geometry import triangulate_correspondence
except ImportError:
    from geometry_pipeline import discover_videos, analyze_video, read_frame
    from common_object_matching import match_images
    from self_calibration import self_calibrate
    from cross_camera_geometry import triangulate_correspondence


def as_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_dict(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "__dict__"):
        return as_dict(value.__dict__)
    return value


def get_selected_frames(video_path, geometry_result, maximum=3):
    selected = getattr(geometry_result, "selected_frames", [])
    frames = []

    for item in selected[:maximum]:
        index = int(item.frame_index) if hasattr(item, "frame_index") else int(item["frame_index"])
        frame = read_frame(video_path, index)

        if frame is not None:
            frames.append({"frame_index": index, "frame": frame})

    return frames


def estimate_intrinsics(frame):
    height, width = frame.shape[:2]

    try:
        result = self_calibrate(frame)
        K = getattr(result, "intrinsic_matrix", None)

        if getattr(result, "success", False) and K is not None:
            return np.asarray(K, dtype=np.float64), True, float(getattr(result, "confidence", 0.0)), "self_calibration"
    except Exception:
        pass

    focal = float(max(width, height))
    K = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K, False, 0.20, "dimension_based_fallback"


class DegeneratePairError(RuntimeError):
    """Raised when a camera pair cannot produce a valid two-view geometry."""


def pose_is_physical(R, t, max_rotation_deg=75.0):
    """
    Reject poses that are physically implausible for fixed CCTV cameras.

    Fixed surveillance cameras look at roughly the same scene directions, so a
    recovered rotation near 90/120 degrees plus an in-plane flip signature is
    almost always the degenerate solution produced by duplicate/near-identical
    footage, not a real camera motion.
    """
    try:
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        cos_angle = (np.trace(R) - 1.0) / 2.0
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        if angle_deg > max_rotation_deg:
            return False
        return True
    except Exception:
        return False


def estimate_relative_pose(points_a, points_b, K_a, K_b):
    if len(points_a) < 8:
        return {"success": False, "message": "At least 8 correspondences are required."}

    pts_a = np.asarray(points_a, dtype=np.float64).reshape(-1, 1, 2)
    pts_b = np.asarray(points_b, dtype=np.float64).reshape(-1, 1, 2)

    try:
        norm_a = cv2.undistortPoints(pts_a, K_a, None).reshape(-1, 2)
        norm_b = cv2.undistortPoints(pts_b, K_b, None).reshape(-1, 2)

        # Identical (or mirrored) footage yields a degenerate essential matrix;
        # detect it early with a clear user-facing message.
        residual = norm_a - norm_b
        median_shift = float(np.median(np.linalg.norm(residual, axis=1)))

        E, mask = cv2.findEssentialMat(
            norm_a, norm_b, np.eye(3, dtype=np.float64),
            method=cv2.RANSAC, prob=0.999, threshold=0.002
        )

        if E is None or mask is None:
            return {"success": False, "message": "Essential matrix estimation failed."}

        E = E[:3, :3]
        inlier_mask = mask.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())

        if inlier_count < 8:
            return {"success": False, "message": "Too few essential-matrix inliers.", "inliers": inlier_count}

        _, R, t, pose_mask = cv2.recoverPose(
            E, norm_a[inlier_mask], norm_b[inlier_mask],
            np.eye(3, dtype=np.float64)
        )

        if R is None or t is None:
            return {"success": False, "message": "Relative pose recovery failed."}

        if not pose_is_physical(R, t):
            raise DegeneratePairError(
                "Recovered camera rotation is physically implausible. The two "
                "videos are likely duplicates of the same footage, or the views "
                "overlap too little for two-view geometry."
            )

        return {
            "success": True,
            "rotation": R.tolist(),
            "translation_direction": t.reshape(3).tolist(),
            "essential_inliers": inlier_count,
            "essential_matrix": E.tolist(),
            "median_pixel_shift": median_shift,
        }

    except DegeneratePairError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def make_projection_matrices(K_a, K_b, R, t):
    P_a = K_a @ np.hstack((
        np.eye(3, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
    ))

    P_b = K_b @ np.hstack((
        R,
        t.reshape(3, 1),
    ))

    return P_a, P_b


_SIFT_CACHE = {}
_SIFT_CACHE_MAX = 12


def _cached_sift(gray):
    """SIFT keypoints/descriptors for a grayscale frame, cached by content.

    Auto-selection verifies dozens of candidate patches against the SAME
    other-view frame; recomputing full-image SIFT for it on every candidate
    made that loop take seconds per candidate. With the cache the frame is
    processed once and every subsequent candidate reuses the keypoints.
    """
    key = (gray.shape, hash(gray.tobytes()))
    cached = _SIFT_CACHE.get(key)
    if cached is not None:
        return cached
    sift = cv2.SIFT_create(nfeatures=5000)
    kp, des = sift.detectAndCompute(gray, None)
    if len(_SIFT_CACHE) >= _SIFT_CACHE_MAX:
        _SIFT_CACHE.clear()
    _SIFT_CACHE[key] = (kp, des)
    return kp, des


def local_target_match(image_a, image_b, target_xy, radius=80):
    """
    Match a user-selected target point from image_a to image_b.

    The click itself does not need to land on a strong keypoint. A patch around
    the click is matched against image_b, a local/projective transform is
    estimated with RANSAC, and the exact click coordinate is transferred
    through it. This is point correspondence, not object-class detection.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if image_b.ndim == 3 else image_b

    x, y = float(target_xy[0]), float(target_xy[1])
    h, w = gray_a.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        return {
            "success": False,
            "message": "Selected point is outside Camera 01 frame.",
        }

    sift = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    kp_b, des_b = _cached_sift(gray_b)
    if des_b is None or len(kp_b) < 4:
        return {"success": False, "message": "Insufficient SIFT features in the other view."}

    # ---- Primary: local patch around the click -> homography transfer ----
    patch_radius = int(max(70, min(140, 0.12 * min(h, w))))
    x0, x1 = max(0, int(x - patch_radius)), min(w, int(x + patch_radius))
    y0, y1 = max(0, int(y - patch_radius)), min(h, int(y + patch_radius))
    patch = gray_a[y0:y1, x0:x1]

    if patch.size >= 400:
        kp_p, des_p = sift.detectAndCompute(patch, None)

        if des_p is not None and len(kp_p) >= 6:
            knn = matcher.knnMatch(des_p, des_b, k=2)
            good = [
                pair[0] for pair in knn
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
            ]

            if len(good) >= 6:
                src = np.float32([kp_p[m.queryIdx].pt for m in good])
                dst = np.float32([kp_b[m.trainIdx].pt for m in good])

                # Deterministic RANSAC: cv2.setRNGSeed BEFORE every call —
                # without it two runs can disagree on knife-edge matches
                # (same input, different box on the user's screen).
                cv2.setRNGSeed(20260916)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)

                if H is not None and mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(inlier_mask.sum())

                    if inliers >= 5:
                        click_local = np.float32([[[x - x0, y - y0]]])
                        pred = cv2.perspectiveTransform(click_local, H)[0, 0]
                        px, py = float(pred[0]), float(pred[1])

                        if 0 <= px < w and 0 <= py < h:
                            residuals = []
                            for a_pt, b_pt, keep in zip(src, dst, inlier_mask):
                                if keep:
                                    q = cv2.perspectiveTransform(np.float32([[a_pt]]), H)[0, 0]
                                    residuals.append(float(np.linalg.norm(q - b_pt)))

                            med_err = float(np.median(residuals)) if residuals else 999.0
                            inlier_ratio = inliers / max(len(good), 1)

                            confidence = max(0.0, min(1.0,
                                0.45 * min(inlier_ratio / 0.55, 1.0)
                                + 0.35 * min(inliers / 12.0, 1.0)
                                + 0.20 * max(0.0, 1.0 - med_err / 8.0)))

                            result = {
                                "success": confidence >= 0.45,
                                "point_a": [x, y],
                                "point_b": [px, py],
                                "click_point_a": [x, y],
                                "method": "patch_SIFT+RANSAC_homography_transfer",
                                "confidence": confidence,
                                "inliers": inliers,
                                "candidate_matches": len(good),
                                "median_transfer_error_px": med_err,
                                "patch_radius_px": patch_radius,
                            }

                            if result["success"]:
                                return result

    # ---- Secondary: normalized template matching on the click patch ----
    # Robust when texture is weak or the two views are similar; used only to
    # transfer the exact click location, with a conservative confidence.
    if patch.size >= 400:
        try:
            tm = cv2.matchTemplate(gray_b, patch, cv2.TM_CCOEFF_NORMED)
            _, peak, _, peak_loc = cv2.minMaxLoc(tm)

            if peak >= 0.72:
                px = float(peak_loc[0] + (x - x0))
                py = float(peak_loc[1] + (y - y0))

                if 0 <= px < w and 0 <= py < h:
                    confidence = float(max(0.0, min(1.0, (peak - 0.55) / 0.45)))

                    if confidence >= 0.40:
                        return {
                            "success": True,
                            "point_a": [x, y],
                            "point_b": [px, py],
                            "click_point_a": [x, y],
                            "method": "patch_template_match_NCC",
                            "confidence": confidence,
                            "ncc_peak": float(peak),
                            "patch_radius_px": patch_radius,
                        }
        except Exception:
            pass

    # ---- Fallback: nearest strong keypoint snap (original behaviour) ----
    kp_a, des_a = _cached_sift(gray_a)
    if des_a is None or des_b is None or len(kp_a) < 4:
        return {"success": False, "message": "Insufficient SIFT features."}

    selected = []
    for i, kp in enumerate(kp_a):
        d = float(np.hypot(kp.pt[0] - x, kp.pt[1] - y))
        if d <= radius:
            selected.append((d, i))

    if not selected:
        selected = sorted(
            (
                (float(np.hypot(kp.pt[0] - x, kp.pt[1] - y)), i)
                for i, kp in enumerate(kp_a)
            ),
            key=lambda z: z[0],
        )[:12]

    selected.sort(key=lambda z: z[0])
    selected = selected[:24]

    matches = []
    for _, idx in selected:
        pair = matcher.knnMatch(des_a[idx:idx + 1], des_b, k=2)
        if not pair or len(pair[0]) < 2:
            continue
        m, n = pair[0]
        if m.distance < 0.78 * n.distance:
            matches.append(m)

    if not matches:
        return {
            "success": False,
            "message": "No reliable local feature match found.",
        }

    matches.sort(key=lambda m: m.distance)

    best_score = float("inf")
    best_match = matches[0]

    for m in matches:
        src = np.asarray(kp_a[m.queryIdx].pt, dtype=np.float64)
        click_error = float(np.linalg.norm(src - np.asarray([x, y])))
        score = click_error + 0.15 * float(m.distance)
        if score < best_score:
            best_score = score
            best_match = m

    source_pt = np.asarray(kp_a[best_match.queryIdx].pt, dtype=np.float64)
    target_pt = np.asarray(kp_b[best_match.trainIdx].pt, dtype=np.float64)
    click_error = float(np.linalg.norm(source_pt - np.asarray([x, y])))

    # A keypoint far from the click does not represent the clicked object.
    if click_error > 2.0 * radius:
        return {
            "success": False,
            "message": (
                "The selected point has no reliable counterpart in the other "
                "view (nearest stable feature is too far from the click)."
            ),
            "click_to_feature_error_px": click_error,
        }

    confidence = max(
        0.0,
        min(
            1.0,
            0.90
            * np.exp(-click_error / max(radius, 1))
            * np.exp(-float(best_match.distance) / 250.0),
        ),
    )

    return {
        "success": True,
        "point_a": source_pt.tolist(),
        "point_b": target_pt.tolist(),
        "click_point_a": [x, y],
        "method": "nearest_keypoint_snap",
        "click_to_feature_error_px": click_error,
        "descriptor_distance": float(best_match.distance),
        "confidence": confidence,
    }


def triangulate_target(point_a, point_b, K_a, K_b, R, t):
    """
    Triangulate the target with cheirality handling.

    Two-view pose fixes the scale sign of t only up to a global ambiguity; a
    valid camera pair can still triangulate with negative depth for every
    point. If the first attempt fails cheirality, retry once with the flipped
    baseline before declaring failure.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    candidates = [(R, t)]
    candidates.append((R, -t))

    P_a = K_a @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))

    best = None

    for R_try, t_try in candidates:
        P_b_try = K_b @ np.hstack((R_try, t_try.reshape(3, 1)))
        try:
            result = triangulate_correspondence(
                point_a,
                point_b,
                P_a,
                P_b_try,
                rotation_a=np.eye(3, dtype=np.float64),
                translation_a=np.zeros(3, dtype=np.float64),
                rotation_b=R_try,
                translation_b=t_try,
            )
        except Exception as exc:
            best = {"valid": False, "message": str(exc)}
            continue

        result = as_dict(result)
        if result.get("valid", False):
            result["baseline_flipped"] = bool(np.allclose(t_try, -t) and not np.allclose(t_try, t))
            return result

        if best is None or (
            (result.get("reprojection_error_a") or 1e9)
            < (best.get("reprojection_error_a") or 1e9)
        ):
            best = result

    return best if best is not None else {"valid": False, "message": "Triangulation failed."}


def analyze_pair(
    camera_a,
    frame_a,
    camera_b,
    frame_b,
    target_xy=None,
):
    """
    Analyze one camera pair.

    If target_xy is supplied, the returned distance is for that target.
    Otherwise the function falls back to the existing generic feature
    reconstruction for compatibility.
    """

    match = match_images(
        frame_a,
        frame_b,
        path_a=f"{camera_a.name}:target_frame",
        path_b=f"{camera_b.name}:target_frame",
    )

    match_data = as_dict(match)
    points_a = match_data.get("matched_points_a", [])
    points_b = match_data.get("matched_points_b", [])

    result = {
        "camera_a": camera_a.name,
        "camera_b": camera_b.name,
        "generic_match_confidence": float(match_data.get("confidence", 0.0)),
        "matched_points": len(points_a),
    }

    if len(points_a) < 8:
        result["valid"] = False
        result["message"] = "Fewer than 8 generic correspondences."
        return result

    K_a, self_a, conf_a, source_a = estimate_intrinsics(frame_a)
    K_b, self_b, conf_b, source_b = estimate_intrinsics(frame_b)

    pose = estimate_relative_pose(points_a, points_b, K_a, K_b)

    result["camera_intrinsics"] = {
        "camera_a": {
            "source": source_a,
            "self_calibrated": self_a,
            "confidence": conf_a,
            "K": K_a.tolist(),
        },
        "camera_b": {
            "source": source_b,
            "self_calibrated": self_b,
            "confidence": conf_b,
            "K": K_b.tolist(),
        },
    }

    result["relative_pose"] = pose

    if not pose.get("success", False):
        result["valid"] = False
        result["message"] = pose.get("message", "Relative pose unavailable.")
        return result

    R = np.asarray(pose["rotation"], dtype=np.float64)
    t = np.asarray(pose["translation_direction"], dtype=np.float64)

    # Near-identical views produce a tiny/ill-defined baseline; two-view depths
    # become numerically unstable even when reprojection error stays low.
    if float(pose.get("median_pixel_shift", 999.0)) < 3.0:
        result["low_parallax_warning"] = (
            "The two views are nearly identical (very low parallax); the "
            "reconstructed relative depth is unreliable."
        )

    P_a, P_b = make_projection_matrices(K_a, K_b, R, t)

    result["projection_matrices"] = {
        "camera_a": P_a.tolist(),
        "camera_b": P_b.tolist(),
    }

    if target_xy is not None:
        target_match = local_target_match(
            frame_a,
            frame_b,
            target_xy,
        )

        result["target_correspondence"] = target_match

        if not target_match.get("success", False):
            result["valid"] = False
            result["message"] = target_match.get(
                "message",
                "Target correspondence failed.",
            )
            return result

        target = triangulate_target(
            target_match["point_a"],
            target_match["point_b"],
            K_a,
            K_b,
            R,
            t,
        )

        result["target_triangulation"] = target

        if not target.get("valid", False):
            result["valid"] = False
            result["message"] = target.get(
                "message",
                "Target triangulation failed.",
            )
            return result

        result["target_distance"] = {
            "camera_a_relative": abs(float(target["depth_camera_a"])),
            "camera_b_relative": abs(float(target["depth_camera_b"])),
            "point_3d": target.get("point_3d"),
            "reprojection_error_a": target.get("reprojection_error_a"),
            "reprojection_error_b": target.get("reprojection_error_b"),
            "unit": "relative_scene_units",
            "metric_available": False,
        }

        result["valid"] = True
        return result

    # Compatibility path: generic reconstruction.
    triangulated = []

    for pa, pb in zip(points_a, points_b):
        item = triangulate_target(
            pa, pb, K_a, K_b, R, t
        )
        if item.get("valid", False):
            triangulated.append(item)

    da = [
        abs(float(x["depth_camera_a"]))
        for x in triangulated
        if x.get("depth_camera_a") is not None
    ]

    db = [
        abs(float(x["depth_camera_b"]))
        for x in triangulated
        if x.get("depth_camera_b") is not None
    ]

    result["triangulation"] = {
        "success": len(triangulated) >= 4,
        "valid_points": len(triangulated),
        "camera_a_distance_relative": float(np.median(da)) if da else None,
        "camera_b_distance_relative": float(np.median(db)) if db else None,
        "points": triangulated,
    }

    result["valid"] = result["triangulation"]["success"]
    return result


def videos_are_duplicates(video_a, video_b, sample_frames=5, hash_size=8):
    """
    Detect whether two recordings are the same footage.

    Compares a handful of dHash fingerprints sampled at identical normalized
    positions. Used to warn about duplicate uploads, which make two-view
    triangulation impossible.
    """
    try:
        def fingerprint(frame):
            small = cv2.resize(frame, (hash_size + 1, hash_size))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return (gray[:, 1:] > gray[:, :-1]).flatten()

        info_a = get_video_info(video_a)
        info_b = get_video_info(video_b)

        cap_a = cv2.VideoCapture(str(video_a))
        cap_b = cv2.VideoCapture(str(video_b))

        same = 0
        compared = 0

        for frac in np.linspace(0.1, 0.9, sample_frames):
            ia = int(frac * max(info_a["frame_count"] - 1, 0))
            ib = int(frac * max(info_b["frame_count"] - 1, 0))

            cap_a.set(cv2.CAP_PROP_POS_FRAMES, ia)
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, ib)

            ok_a, fa = cap_a.read()
            ok_b, fb = cap_b.read()

            if not ok_a or not ok_b:
                continue

            compared += 1
            ha = fingerprint(fa)
            hb = fingerprint(fb)
            distance = int(np.count_nonzero(ha != hb))

            if distance <= 8:
                same += 1

        cap_a.release()
        cap_b.release()

        return compared >= 3 and same >= max(3, int(0.8 * compared))

    except Exception:
        return False


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 1.0}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    cap.release()
    return {"frame_count": frame_count, "fps": fps if fps > 0 else 1.0}


def analyze_folder(
    input_folder,
    target_x=None,
    target_y=None,
    target_frame=None,
    candidate_count=12,
    keep_count=4,
    frames_per_camera=3,
):
    videos = discover_videos(input_folder)

    if len(videos) < 2:
        raise RuntimeError("At least two CCTV videos are required.")

    cameras = []

    for video in videos:
        geometry = analyze_video(
            video,
            candidate_count=candidate_count,
            keep_count=keep_count,
        )

        frames = get_selected_frames(
            video,
            geometry,
            maximum=frames_per_camera,
        )

        cameras.append({
            "video": video,
            "geometry": geometry,
            "frames": frames,
        })

    # If UI provides a frame index, use it for Camera 01 when available.
    if target_frame is not None and cameras[0]["frames"]:
        selected = read_frame(cameras[0]["video"], int(target_frame))
        if selected is not None:
            camera_a_frame = selected
            camera_a_frame_index = int(target_frame)
        else:
            camera_a_frame = cameras[0]["frames"][0]["frame"]
            camera_a_frame_index = int(cameras[0]["frames"][0]["frame_index"])
    else:
        camera_a_frame = cameras[0]["frames"][0]["frame"]
        camera_a_frame_index = int(cameras[0]["frames"][0]["frame_index"])

    target_results = []
    pair_results = []

    # Duplicate uploads cannot contribute to multi-view geometry (two copies of
    # the same footage have zero parallax, so 3D triangulation is impossible).
    # They are detected and skipped early; the per-camera metric layer still
    # provides camera->ground and camera->object estimates for them.
    unique_videos = [cameras[0]["video"]]

    for index in range(1, len(cameras)):
        camera_b = cameras[index]

        if not camera_b["frames"]:
            continue

        # Pick the strongest available frame for Camera B.
        frame_b_item = camera_b["frames"][0]

        duplicate_of = None
        for unique_video in unique_videos:
            if videos_are_duplicates(unique_video, camera_b["video"]):
                duplicate_of = unique_video.name
                break

        if duplicate_of is not None:
            # Skip 3D for duplicates, but still transfer the clicked point so
            # the per-camera single-view object distance remains available.
            transferred = {"success": False}
            if target_x is not None and target_y is not None:
                try:
                    transferred = local_target_match(
                        camera_a_frame,
                        frame_b_item["frame"],
                        (float(target_x), float(target_y)),
                    )
                except Exception:
                    transferred = {"success": False}

            pair_results.append({
                "camera_a": cameras[0]["video"].name,
                "camera_b": camera_b["video"].name,
                "valid": False,
                "skipped_duplicate": True,
                "duplicate_of": duplicate_of,
                "best_frame_a": camera_a_frame_index,
                "best_frame_b": int(frame_b_item["frame_index"]),
                "target_correspondence": transferred,
                "message": (
                    f"Skipped: this video is a duplicate copy of '{duplicate_of}'. "
                    "Duplicates are ignored for 3D reconstruction; the per-camera "
                    "camera-to-ground and camera-to-object estimates still apply."
                ),
            })
            continue

        unique_videos.append(camera_b["video"])

        pair = analyze_pair(
            cameras[0]["video"],
            camera_a_frame,
            camera_b["video"],
            frame_b_item["frame"],
            target_xy=(
                [float(target_x), float(target_y)]
                if target_x is not None and target_y is not None
                else None
            ),
        )

        pair["best_frame_a"] = camera_a_frame_index
        pair["best_frame_b"] = int(frame_b_item["frame_index"])

        pair_results.append(pair)

        if pair.get("target_distance"):
            target_results.append({
                "camera": cameras[0]["video"].name,
                "distance_relative": pair["target_distance"]["camera_a_relative"],
                "reprojection_error": pair["target_distance"]["reprojection_error_a"],
                "source_pair": [
                    cameras[0]["video"].name,
                    camera_b["video"].name,
                ],
            })

            target_results.append({
                "camera": camera_b["video"].name,
                "distance_relative": pair["target_distance"]["camera_b_relative"],
                "reprojection_error": pair["target_distance"]["reprojection_error_b"],
                "source_pair": [
                    cameras[0]["video"].name,
                    camera_b["video"].name,
                ],
            })

    # Remove duplicate Camera 01 estimates by robust median.
    camera_distance_map = {}

    for item in target_results:
        camera_distance_map.setdefault(item["camera"], []).append(
            float(item["distance_relative"])
        )

    final_target_distances = {
        camera: float(np.median(values))
        for camera, values in camera_distance_map.items()
    }

    return {
        "pipeline": "multi_camera_target_point_geometry",
        "input_folder": str(Path(input_folder).resolve()),
        "camera_count": len(videos),
        "target": {
            "camera": videos[0].name,
            "frame_index": camera_a_frame_index,
            "image_point": (
                [float(target_x), float(target_y)]
                if target_x is not None and target_y is not None
                else None
            ),
            "selection_mode": (
                "user_click"
                if target_x is not None and target_y is not None
                else "generic"
            ),
        },
        "camera_distances_relative": final_target_distances,
        "cross_camera_pairs": pair_results,
        "unique_camera_count": len(unique_videos),
        "skipped_duplicate_cameras": len(videos) - len(unique_videos),
        "metric_status": "not_available_without_valid_metric_scale",
        "distance_unit": "relative_scene_units",
        "environment_independent": True,
        "limitations": [
            "The target correspondence uses local visual features around the selected point.",
            "The selected point must contain stable visual texture or a nearby feature.",
            "Two-view geometry alone does not determine absolute metric scale.",
            "Camera 01 is used as the target-reference view for this stage.",
            "Duplicate copies of the same footage are detected and ignored for 3D reconstruction.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera target-point distance pipeline."
    )

    parser.add_argument("input_folder")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-frame", type=int, default=None)
    parser.add_argument(
        "--output",
        default="results/target_point_report.json",
    )
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--keep-count", type=int, default=4)
    parser.add_argument("--frames-per-camera", type=int, default=3)

    args = parser.parse_args()

    report = analyze_folder(
        args.input_folder,
        target_x=args.target_x,
        target_y=args.target_y,
        target_frame=args.target_frame,
        candidate_count=args.candidate_count,
        keep_count=args.keep_count,
        frames_per_camera=args.frames_per_camera,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(as_dict(report), indent=4),
        encoding="utf-8",
    )

    print("=" * 70)
    print("STEP 13 — TARGET-POINT DISTANCE PIPELINE")
    print("=" * 70)
    print(f"Cameras discovered : {report['camera_count']}")
    print(
        "Target selection   : "
        + report["target"]["selection_mode"]
    )

    if report["target"]["image_point"]:
        print(
            "Target point       : "
            f"{report['target']['image_point']}"
        )

    print()
    print("TARGET DISTANCES")

    for camera, distance in report["camera_distances_relative"].items():
        print(
            f"  {camera}: "
            f"{distance:.4f} relative units"
        )

    print()
    print(
        "Metric status      : "
        f"{report['metric_status']}"
    )
    print(f"Report saved       : {output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
