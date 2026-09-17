from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
# The webapp is distributed inside a wrapper folder, while the user project
# keeps input_videos/results/src at the project root.
INPUT_DIR = PROJECT / "input_videos"
RESULTS_DIR = PROJECT / "results"
SRC_DIR = PROJECT / "src"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
INPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CCTV Distance Estimation", version="4.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

STATE: dict[str, Any] = {"selected_target": None, "last_result": None}


class TargetSelection(BaseModel):
    camera_index: int = Field(0, ge=0)
    frame_index: int = Field(0, ge=0)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)


class AnalyzeRequest(TargetSelection):
    scale_mode: str = Field(default="visual_prior")


def videos() -> list[Path]:
    return sorted(
        [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT],
        key=lambda p: str(p.relative_to(INPUT_DIR)).lower(),
    )


def safe_video_path(name: str) -> Path:
    root = INPUT_DIR.resolve()
    p = (INPUT_DIR / name).resolve()
    if root not in p.parents or p.suffix.lower() not in VIDEO_EXT or not p.exists():
        raise HTTPException(404, "Video not found")
    return p


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"name": path.name, "frames": 0, "fps": 0, "width": 0, "height": 0, "duration": 0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "name": path.name,
        "relative_path": str(path.relative_to(INPUT_DIR)),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": frames / fps if fps else 0,
    }


def read_frame(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def jpeg_bytes(frame: np.ndarray, quality: int = 86) -> bytes:
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(500, "Unable to encode frame")
    return enc.tobytes()


def temp_jpeg(frame: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="cctv_frame_")
    os.close(fd)
    Path(path).write_bytes(jpeg_bytes(frame))
    return path


def camera_matrix(width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.012, edgeThreshold=10)
    return cv2.ORB_create(nfeatures=7000)


def extract_features(image: np.ndarray):
    det = feature_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = det.detectAndCompute(gray, None)
    return kp or [], des


def robust_matches(a: np.ndarray, b: np.ndarray):
    kpa, da = extract_features(a)
    kpb, db = extract_features(b)
    if da is None or db is None or len(kpa) < 12 or len(kpb) < 12:
        return kpa, kpb, [], None
    norm = cv2.NORM_L2 if da.dtype == np.float32 else cv2.NORM_HAMMING
    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in knn if m.distance < 0.72 * n.distance]
    if len(good) < 8:
        return kpa, kpb, good, None
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    H, mask_h = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995)
    if mask_h is not None:
        good = [m for m, keep in zip(good, mask_h.ravel()) if keep]
    return kpa, kpb, good, H


def selected_patch(image: np.ndarray, x: float, y: float, radius: int = 70):
    h, w = image.shape[:2]
    x0, x1 = max(0, int(x - radius)), min(w, int(x + radius))
    y0, y1 = max(0, int(y - radius)), min(h, int(y + radius))
    return image[y0:y1, x0:x1], (x0, y0)


def target_correspondence(ref: np.ndarray, other: np.ndarray, click: tuple[float, float]) -> dict[str, Any]:
    """Locate an arbitrary user-clicked fixed point in another camera.

    The click itself does not have to land on a feature.  We match a local patch
    around the click, estimate a local/projective transform with RANSAC, and then
    transfer the exact click coordinate through that transform.  A global image
    homography is used only as a fallback for approximately planar scenes.
    """
    x, y = map(float, click)
    h, w = ref.shape[:2]
    patch_radius = int(max(70, min(140, 0.12 * min(h, w))))
    patch, (x0, y0) = selected_patch(ref, x, y, patch_radius)

    def local_feature_match() -> dict[str, Any] | None:
        if patch.size == 0:
            return None
        det = feature_detector()
        pg = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        og = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)
        kpa, da = det.detectAndCompute(pg, None)
        kpb, db = det.detectAndCompute(og, None)
        if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
            return None
        norm = cv2.NORM_L2 if da.dtype == np.float32 else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm)
        knn = matcher.knnMatch(da, db, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < 6:
            return None
        src = np.float32([kpa[m.queryIdx].pt for m in good])
        dst = np.float32([kpb[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=5000, confidence=0.995)
        if H is None or mask is None:
            return None
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        if inliers < 5:
            return None
        click_local = np.float32([[[x - x0, y - y0]]])
        pred = cv2.perspectiveTransform(click_local, H)[0, 0]
        px, py = float(pred[0]), float(pred[1])
        if not (0 <= px < other.shape[1] and 0 <= py < other.shape[0]):
            return None
        inlier_ratio = inliers / max(len(good), 1)
        residuals = []
        for a, b, keep in zip(src, dst, inlier_mask):
            if keep:
                q = cv2.perspectiveTransform(np.float32([[a]]), H)[0, 0]
                residuals.append(float(np.linalg.norm(q - b)))
        med_err = float(np.median(residuals)) if residuals else 999.0
        conf = max(0.0, min(1.0, 0.45 * min(inlier_ratio / 0.55, 1.0) + 0.35 * min(inliers / 12.0, 1.0) + 0.20 * max(0.0, 1.0 - med_err / 8.0)))
        return {
            "success": conf >= 0.45,
            "point": [px, py],
            "method": "local-patch-SIFT/ORB+RANSAC-homography",
            "confidence": float(conf),
            "inliers": inliers,
            "candidate_matches": len(good),
            "median_transfer_error_px": med_err,
            "patch_radius_px": patch_radius,
        }

    local = local_feature_match()
    if local is not None and local.get("success"):
        return local

    # Fallback 1: global feature correspondence, but choose the match nearest the
    # clicked location only when it is genuinely close to the click.
    kpa, kpb, matches, H_global = robust_matches(ref, other)
    if len(matches) >= 8:
        pts_a = np.float32([kpa[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in matches])
        F, mask_f = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.5, 0.995)
        fm = [m for m, keep in zip(matches, mask_f.ravel()) if keep] if F is not None and F.shape == (3, 3) and mask_f is not None else matches
        nearby = []
        for m in fm:
            pa = np.array(kpa[m.queryIdx].pt, dtype=float)
            pb = np.array(kpb[m.trainIdx].pt, dtype=float)
            click_err = float(np.linalg.norm(pa - np.array([x, y])))
            if click_err <= max(35.0, 0.045 * min(ref.shape[:2])):
                epi = 0.0
                if F is not None and F.shape == (3, 3):
                    line = F @ np.array([x, y, 1.0])
                    epi = abs(float(line @ np.array([pb[0], pb[1], 1.0]))) / max(math.hypot(line[0], line[1]), 1e-9)
                score = 0.45 * click_err + 1.2 * epi + float(m.distance)
                nearby.append((score, pb, click_err, epi, float(m.distance)))
        if nearby:
            nearby.sort(key=lambda z: z[0])
            best = nearby[0]
            conf = max(0.0, min(1.0, 1.0 - best[0] / 150.0))
            if conf >= 0.40:
                return {
                    "success": True,
                    "point": [float(best[1][0]), float(best[1][1])],
                    "method": "near-click-feature+fundamental-RANSAC",
                    "confidence": float(conf),
                    "click_error_px": float(best[2]),
                    "epipolar_error_px": float(best[3]),
                    "descriptor_distance": float(best[4]),
                    "inliers": len(fm),
                }

    # Fallback 2: a global homography is appropriate for a fixed point on a common
    # approximately planar surface (floor, wall, sign, desk, etc.).
    if H_global is not None:
        pred = cv2.perspectiveTransform(np.float32([[[x, y]]]), H_global)[0, 0]
        px, py = float(pred[0]), float(pred[1])
        if 0 <= px < other.shape[1] and 0 <= py < other.shape[0]:
            return {
                "success": True,
                "point": [px, py],
                "method": "global-RANSAC-homography",
                "confidence": 0.55,
                "inliers": len(matches),
            }

    return {
        "success": False,
        "reason": "No reliable correspondence for selected target; choose a textured fixed point/feature visible in both views",
        "inliers": len(matches),
    }


def recover_relative_pose(ref: np.ndarray, other: np.ndarray):
    kpa, kpb, matches, _ = robust_matches(ref, other)
    if len(matches) < 12:
        return None
    pts_a = np.float32([kpa[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in matches])
    K1 = camera_matrix(ref.shape[1], ref.shape[0])
    K2 = camera_matrix(other.shape[1], other.shape[0])
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K1, method=cv2.RANSAC, prob=0.999, threshold=1.2)
    if E is None:
        return None
    if E.shape != (3, 3):
        E = E[:3, :3]
    count, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, K1, mask=mask)
    return {"K1": K1, "K2": K2, "R": R, "t": t.reshape(3), "inliers": int(count), "matches": len(matches)}


def triangulate(P1, P2, p1, p2):
    Xh = cv2.triangulatePoints(
        P1, P2,
        np.array([[p1[0]], [p1[1]]], dtype=np.float64),
        np.array([[p2[0]], [p2[1]]], dtype=np.float64),
    )
    if abs(float(Xh[3, 0])) < 1e-12:
        raise ValueError("Degenerate triangulation")
    X = (Xh[:3, 0] / Xh[3, 0]).astype(float)
    e1p = P1 @ np.r_[X, 1.0]
    e2p = P2 @ np.r_[X, 1.0]
    e1 = float(np.linalg.norm(e1p[:2] / e1p[2] - np.array(p1)))
    e2 = float(np.linalg.norm(e2p[:2] / e2p[2] - np.array(p2)))
    return X, e1, e2


def estimate_person_scale(frame: np.ndarray):
    # Optional, explicitly estimated scale cue. Never treated as ground truth.
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        r = model(frame, verbose=False)[0]
        heights = []
        for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(cls) == 0:
                h = float(box[3] - box[1])
                if h >= 55:
                    heights.append(h)
        if not heights:
            return None, 0.0, "No sufficiently clear person detected"
        px = float(np.median(heights))
        # We only use the statistical prior to establish an estimated visual scale.
        # It is intentionally labelled as an estimate and not company ground truth.
        return 1.70 / px, 0.55, "Statistical person-height prior (1.70 m)"
    except Exception as exc:
        return None, 0.0, f"Visual scale unavailable: {type(exc).__name__}"


def analyze(req: AnalyzeRequest):
    vs = videos()
    if len(vs) < 2:
        raise HTTPException(400, "At least two CCTV videos are required for multi-camera analysis.")
    if req.camera_index >= len(vs):
        raise HTTPException(400, "Reference camera index is out of range.")

    ref_path = vs[req.camera_index]
    ref_info = video_info(ref_path)
    frame_idx = min(int(req.frame_index), max(ref_info["frames"] - 1, 0))
    ref = read_frame(ref_path, frame_idx)
    if ref is None:
        raise HTTPException(400, "Unable to read the selected reference frame.")
    if req.x >= ref.shape[1] or req.y >= ref.shape[0]:
        raise HTTPException(400, "Selected target is outside the reference frame.")

    rows: list[dict[str, Any]] = []
    all_valid = []
    for i, path in enumerate(vs):
        if i == req.camera_index:
            continue
        info = video_info(path)
        # Match the normalized video position rather than always using the middle frame.
        pos = frame_idx / max(ref_info["frames"] - 1, 1)
        other_idx = int(round(pos * max(info["frames"] - 1, 0)))
        other = read_frame(path, other_idx)
        row: dict[str, Any] = {"camera": path.name, "frame_index": other_idx}
        if other is None:
            row.update(valid=False, reason="Unable to read corresponding frame")
            rows.append(row)
            continue

        corr = target_correspondence(ref, other, (req.x, req.y))
        row["correspondence"] = corr
        if not corr.get("success"):
            row.update(valid=False, reason=corr.get("reason", "Target correspondence rejected"))
            rows.append(row)
            continue

        pose = recover_relative_pose(ref, other)
        if pose is None:
            row.update(valid=False, reason="Relative camera pose could not be estimated")
            rows.append(row)
            continue

        P1 = pose["K1"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = pose["K2"] @ np.hstack([pose["R"], pose["t"][:, None]])
        try:
            X, e1, e2 = triangulate(P1, P2, (req.x, req.y), corr["point"])
        except Exception as exc:
            row.update(valid=False, reason=str(exc))
            rows.append(row)
            continue

        X_other = pose["R"] @ X + pose["t"]
        d_ref = float(np.linalg.norm(X))
        d_other = float(np.linalg.norm(X_other))
        vertical_ref = float(abs(X[1]))
        horizontal_ref = float(math.sqrt(max(d_ref * d_ref - vertical_ref * vertical_ref, 0.0)))
        vertical_other = float(abs(X_other[1]))
        horizontal_other = float(math.sqrt(max(d_other * d_other - vertical_other * vertical_other, 0.0)))
        valid = bool(X[2] > 0 and X_other[2] > 0 and max(e1, e2) <= 8.0 and corr.get("confidence", 0) >= 0.35)
        row.update(
            valid=valid,
            point_3d_relative=X.tolist(),
            reference_camera_distance_relative=d_ref,
            camera_distance_relative=d_other,
            reference_horizontal_relative=horizontal_ref,
            reference_vertical_relative=vertical_ref,
            camera_horizontal_relative=horizontal_other,
            camera_vertical_relative=vertical_other,
            reprojection_error_px=max(e1, e2),
            pose_inliers=pose["inliers"],
            generic_matches=pose["matches"],
            reason=None if valid else "Rejected by correspondence/depth/reprojection checks",
            camera_center_relative=(-pose["R"].T @ pose["t"]).tolist(),
        )
        rows.append(row)
        if valid:
            all_valid.append(row)

    scale = None
    scale_conf = 0.0
    scale_source = "Not established"
    if req.scale_mode == "visual_prior" and all_valid:
        scale, scale_conf, scale_source = estimate_person_scale(ref)

    # Reference camera's target distance is derived from the reconstructed target in the
    # reference camera coordinate system. Use the median only if multiple pairwise rows exist.
    ref_distances = [float(r["reference_camera_distance_relative"]) for r in all_valid]
    ref_distance = float(np.median(ref_distances)) if ref_distances else None
    if ref_distance is not None:
        rows.insert(0, {
            "camera": ref_path.name,
            "frame_index": frame_idx,
            "valid": True,
            "reference": True,
            "distance_relative": ref_distance,
            "horizontal_relative": float(np.median([r["reference_horizontal_relative"] for r in all_valid])),
            "vertical_relative": float(np.median([r["reference_vertical_relative"] for r in all_valid])),
            "distance_m_est": float(ref_distance * scale) if scale is not None else None,
            "metric_status": "estimated" if scale is not None else "relative_only",
        })
    else:
        rows.insert(0, {
            "camera": ref_path.name,
            "frame_index": frame_idx,
            "valid": False,
            "reference": True,
            "reason": "No valid cross-camera reconstruction was available for the selected target",
            "distance_relative": None,
            "distance_m_est": None,
            "metric_status": "unavailable",
        })

    for r in rows:
        if r.get("reference"):
            continue
        if r.get("valid"):
            r["distance_relative"] = r["camera_distance_relative"]
            r["horizontal_relative"] = r["camera_horizontal_relative"]
            r["vertical_relative"] = r["camera_vertical_relative"]
            r["distance_m_est"] = float(r["camera_distance_relative"] * scale) if scale is not None else None
            r["metric_status"] = "estimated" if scale is not None else "relative_only"
        else:
            r["distance_relative"] = None
            r["horizontal_relative"] = None
            r["vertical_relative"] = None
            r["distance_m_est"] = None
            r["metric_status"] = "unavailable"

    result = {
        "version": "2.0",
        "reference_camera": ref_path.name,
        "reference_frame": frame_idx,
        "selected_point_px": [float(req.x), float(req.y)],
        "cameras": [v.name for v in vs],
        "rows": rows,
        "valid_camera_count": len(all_valid) + (1 if ref_distance is not None else 0),
        "valid_pair_count": len(all_valid),
        "metric_scale": {
            "available": scale is not None,
            "m_per_relative_unit": scale,
            "confidence": scale_conf,
            "source": scale_source,
            "note": "Metric values are estimates when visual person-height prior is available; otherwise results remain relative/projective.",
        },
        "status": "completed_metric_estimate" if scale is not None and all_valid else ("completed_relative" if all_valid else "target_reconstruction_failed"),
    }
    (RESULTS_DIR / "web_target_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    STATE["last_result"] = result
    STATE["selected_target"] = req.model_dump()
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/videos")
def api_videos():
    return {"videos": [video_info(p) for p in videos()]}




@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No video files selected")
    saved = []
    for up in files:
        name = Path(up.filename or "").name
        if not name or Path(name).suffix.lower() not in VIDEO_EXT:
            continue
        # Keep uploads local and prevent path traversal.
        dest = INPUT_DIR / name
        stem, suffix = dest.stem, dest.suffix
        n = 1
        while dest.exists():
            dest = INPUT_DIR / f"{stem}_{n}{suffix}"
            n += 1
        with dest.open("wb") as out:
            while True:
                chunk = await up.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        await up.close()
        saved.append(video_info(dest))
    if not saved:
        raise HTTPException(400, "No supported video files were uploaded")
    return {"ok": True, "videos": saved}

@app.get("/api/frame/{name:path}")
def api_frame(name: str, frame: int = 0):
    f = read_frame(safe_video_path(name), frame)
    if f is None:
        raise HTTPException(400, "Unable to read frame")
    return FileResponse(temp_jpeg(f), media_type="image/jpeg")


@app.post("/api/target")
def api_target(req: TargetSelection):
    vs = videos()
    if not vs or req.camera_index >= len(vs):
        raise HTTPException(400, "Reference camera not available")
    f = read_frame(vs[req.camera_index], req.frame_index)
    if f is None:
        raise HTTPException(400, "Selected frame cannot be read")
    if req.x >= f.shape[1] or req.y >= f.shape[0]:
        raise HTTPException(400, "Selected point lies outside the frame")
    STATE["selected_target"] = req.model_dump()
    return {"ok": True, "target": STATE["selected_target"], "frame_size": [f.shape[1], f.shape[0]]}


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    return analyze(req)


@app.get("/api/result")
def api_result():
    return STATE.get("last_result") or {"status": "idle"}


@app.get("/api/self-test")
def api_self_test():
    K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]], float)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.eye(3), np.array([[1.0], [0.0], [0.0]])])
    X_true = np.array([2.0, 1.0, 20.0])
    p1h = P1 @ np.r_[X_true, 1]
    p2h = P2 @ np.r_[X_true, 1]
    p1 = (p1h[:2] / p1h[2]).tolist()
    p2 = (p2h[:2] / p2h[2]).tolist()
    X, e1, e2 = triangulate(P1, P2, p1, p2)
    err = abs(float(np.linalg.norm(X_true)) - float(np.linalg.norm(X)))
    return {"pass": bool(err < 1e-6 and e1 < 1e-6 and e2 < 1e-6), "ground_truth_distance": float(np.linalg.norm(X_true)), "estimated_distance": float(np.linalg.norm(X)), "absolute_error": err, "reprojection_error_a": e1, "reprojection_error_b": e2}
