"""
AUTOMATIC FRAME + CONSTANT-OBJECT SELECTION

Given any set of recorded CCTV videos, automatically:
1. select the best representative frame of every video
   (sharpness / brightness / edge / line score, via the existing scorer),
2. propose a "constant object" point in the first view: a ground-level,
   texture-rich patch that does not move over time (temporal NCC check)
   and is verified to appear in the other views (patch match transfer).

No object class, no scene prior, no hard-coded coordinates: works with any
footage. The user can always override the object by clicking manually.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .geometry_pipeline import discover_videos, sample_video_frames, select_best_frames
    from .multi_camera_pipeline import local_target_match
except ImportError:
    from geometry_pipeline import discover_videos, sample_video_frames, select_best_frames
    from multi_camera_pipeline import local_target_match


# ---------------------------------------------------------------------
# 1. BEST FRAME PER VIDEO
# ---------------------------------------------------------------------

def auto_select_frame(video_path, candidate_count=14):
    """Return (frame_index, score) of the automatically selected best frame."""
    try:
        candidates = sample_video_frames(Path(video_path), candidate_count=candidate_count)
        best = select_best_frames(candidates, keep_count=1)
        if best:
            return int(best[0].frame_index), float(best[0].score)
    except Exception:
        pass

    # Fallback: a frame at ~40% of the video.
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return int(count * 0.4), 0.0


# ---------------------------------------------------------------------
# 2. CONSTANT-OBJECT CANDIDATE SCORING
# ---------------------------------------------------------------------

PATCH_HALF = 44          # patch half-size in px for texture/static checks

# The static check spans *seconds*, not a few frames: over a 2-7 s window
# walking people keep changing while a genuinely fixed object does not.
# Time-based offsets are used when the FPS is known, frame-based otherwise.
STATIC_WINDOW_SECONDS = (2.0, 4.5, 7.0)
STATIC_WINDOW_FRAMES = (120, 270, 420)

# A patch whose residual motion (after compensating the camera's own
# pan/shake) exceeds the adaptive cap is dynamic and rejected. The cap is
# max(MOTION_ENERGY_MAX, half the scene median): on locked-off footage the
# median is small so the absolute floor dominates, while on translating
# footage parallax inflates everything and the relative rule separates
# fixed patches from people crossing them.
MOTION_ENERGY_MAX = 12.0

_YOLO_MODEL = None


def detect_objects(frame_bgr, confidence=0.35):
    """YOLO detections on a frame: [{"name", "conf", "box"}], or None when
    YOLO is unavailable. Persons are included (name == "person") — callers
    filter what they need."""
    global _YOLO_MODEL
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if _YOLO_MODEL is None:
        try:
            _YOLO_MODEL = YOLO("yolo11n.pt")
        except Exception:
            return None
    try:
        results = _YOLO_MODEL(frame_bgr, verbose=False)
    except Exception:
        return None
    out = []
    for result in results or []:
        if result.boxes is None:
            continue
        names = getattr(result, "names", None) or {}
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue
            cls_id = int(box.cls[0])
            name = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(cls_id)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({"name": name, "conf": conf,
                        "box": [float(x1), float(y1), float(x2), float(y2)]})
    return out


def detect_person_boxes(frame_bgr, confidence=0.45):
    """YOLO person boxes [x0, y0, x1, y1] on a frame, or None when YOLO is
    unavailable (the person filter is then skipped)."""
    dets = detect_objects(frame_bgr, confidence=confidence)
    if dets is None:
        return None
    return [d["box"] for d in dets if d["name"] == "person"]


def _patch(gray, x, y, half=PATCH_HALF):
    h, w = gray.shape[:2]
    x0, x1 = max(0, int(x - half)), min(w, int(x + half))
    y0, y1 = max(0, int(y - half)), min(h, int(y + half))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return gray[y0:y1, x0:x1]


def _ncc(a, b):
    """Normalized cross-correlation of two same-size grayscale patches."""
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return -1.0
    return float((a * b).sum() / denom)


def candidate_score(gray, x, y):
    """
    Composite score for a candidate point, higher = better constant object:
      - texture (Laplacian variance) so matching is reliable,
      - margin from image edges,
      - below-horizon, near-ground placement (y in the lower ~55% of frame).
    """
    h, w = gray.shape[:2]
    patch = _patch(gray, x, y)
    if patch is None:
        return -1.0, {}

    texture = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    tex_n = min(texture / 1500.0, 1.0)

    margin = min(x, y, w - x, h - y)
    margin_n = min(margin / (0.10 * min(w, h)), 1.0)

    ground_n = 1.0 if (0.45 * h) <= y <= 0.92 * h else (
        0.35 if 0.30 * h <= y < 0.45 * h else 0.0
    )

    score = 0.50 * tex_n + 0.25 * margin_n + 0.25 * ground_n
    return score, {"texture": texture, "texture_n": tex_n,
                   "margin_n": margin_n, "ground_n": ground_n}


def _static_offsets(video_path, base_idx):
    """Frame indices for the static check: base first, then +/- 2/4.5/7 s."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps > 0:
        steps = [int(round(s * fps)) for s in STATIC_WINDOW_SECONDS]
    else:
        steps = list(STATIC_WINDOW_FRAMES)
    offs = [0]
    for st_ in steps:
        offs += [st_, -st_]

    def _clip(idx):
        if count > 0:
            return min(max(0, idx), count - 1)
        return max(0, idx)

    return [_clip(base_idx + o) for o in offs]


def align_sequence(gray_seq, target_width=640):
    """Compensate global camera motion (pan/shake) between the window frames.

    Many recordings are not from a perfectly locked-off camera: the view
    drifts, so even a physically fixed object changes pixel position and a
    naive per-pixel static check rejects everything. Each frame is aligned
    back onto the base frame with a dominant-plane homography estimated from
    sparse LK optical flow + RANSAC; after this warp a fixed object stays
    pixel-aligned and only genuinely moving content changes.

    If a frame pair cannot be aligned (too few inliers) it is kept raw, so
    the static check stays conservative instead of trusting a bad warp.
    """
    frames = [g for g in gray_seq if g is not None]
    if len(frames) <= 1:
        return frames
    base = frames[0]
    bh, bw = base.shape[:2]
    scale = target_width / float(bw)
    small_size = (target_width, max(1, int(round(bh * scale))))
    small_base = cv2.resize(base, small_size, interpolation=cv2.INTER_AREA)

    p0 = cv2.goodFeaturesToTrack(
        small_base, maxCorners=900, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if p0 is None or len(p0) < 60:
        return frames

    lk = dict(
        winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def _compose_small(H_small, fw, fh):
        """Lift a small-frame homography to full resolution."""
        sx, sy = bw / float(fw), bh / float(fh)
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1.0]])
        return (S @ H_small @ np.linalg.inv(S))

    aligned = [base]
    for frame in frames[1:]:
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (target_width, max(1, int(round(fh * scale)))),
                           interpolation=cv2.INTER_AREA)
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(small_base, small, p0, None, **lk)
        p2, st2, _ = cv2.calcOpticalFlowPyrLK(small, small_base, p1, None, **lk)
        good = (
            (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1)
            & (np.linalg.norm(p0.reshape(-1, 2) - p2.reshape(-1, 2), axis=1) < 1.5)
        )
        src = p0.reshape(-1, 2)[good]
        dst = p1.reshape(-1, 2)[good]
        if len(src) < 25:
            aligned.append(frame)  # cannot align reliably — stay conservative
            continue
        H_small, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H_small is None or inliers is None or int(inliers.sum()) < 25:
            aligned.append(frame)
            continue
        try:
            H_full = _compose_small(H_small, small_size[0], small_size[1])
            if not np.isfinite(H_full).all():
                aligned.append(frame)
                continue
            # dst(base coords) samples `frame` at H_full @ p_base.
            warped = cv2.warpPerspective(
                frame, H_full, (bw, bh),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            aligned.append(warped)
        except cv2.error:
            aligned.append(frame)
    return aligned


def motion_energy(gray_seq, x, y):
    """Mean temporal std-dev (0-255 scale) of the patch across the window.

    High values mean something moves through the patch (walking people,
    water, foliage) — exactly the patches that must NOT be picked as the
    constant object.
    """
    patches = [_patch(g, x, y) for g in gray_seq[1:]]
    patches = [p.astype(np.float32) for p in patches if p is not None]
    if len(patches) < 2:
        return 0.0
    return float(np.stack(patches).std(axis=0).mean())


def static_ncc(gray_seq, x, y):
    """Median NCC of the patch at (x, y) across neighbouring frames. High = static."""
    base = _patch(gray_seq[0], x, y)
    if base is None:
        return -1.0
    vals = []
    for g in gray_seq[1:]:
        v = _ncc(base, _patch(g, x, y))
        if v > -0.5:
            vals.append(v)
    return float(np.median(vals)) if vals else -1.0


def temporal_median(gray_seq, max_frames=5):
    """Median image across the aligned window: walking people cancel out and
    only the FIXED scene content survives — ideal for locating the actual
    constant object around a candidate point."""
    frames = [g for g in gray_seq[:max_frames] if g is not None]
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def snap_to_object(median_gray, x, y, half=PATCH_HALF):
    """Snap a candidate point onto the physical object it sits on.

    In the temporal-median image the object shows up as the gradient-energy
    blob around the point while the flat pavement stays silent. Otsu on the
    local gradient magnitude + connected components gives the tight bounding
    box of that blob; the point moves to its centroid.

    Returns (nx, ny, [x0, y0, x1, y1]) or None when no object blob is found.
    """
    if median_gray is None:
        return None
    h, w = median_gray.shape[:2]
    x0, x1 = int(max(0, x - 2.0 * half)), int(min(w, x + 2.0 * half))
    y0, y1 = int(max(0, y - 2.0 * half)), int(min(h, y + 2.0 * half))
    win = median_gray[y0:y1, x0:x1]
    if win.size == 0 or win.shape[0] < 24 or win.shape[1] < 24:
        return None
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag * 4.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(float(thr), 60.0)  # weak-texture ground must not swallow the box
    edges = (mag >= thr).astype(np.uint8) * 255
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_lab, lab, stats, cent = cv2.connectedComponentsWithStats(edges, 8)
    cxr, cyr = float(x) - x0, float(y) - y0
    substantial, minor = [], []
    for i in range(1, n_lab):
        bx, by, bw_, bh_, area = stats[i]
        if area < 40 or bw_ > 4.5 * half or bh_ > 4.5 * half:
            continue
        ccx, ccy = cent[i]
        d = float(np.hypot(ccx - cxr, ccy - cyr))
        # Substantial blobs (real objects: dustbin, tripod, bollard, post)
        # win over small pavement markings, then nearest wins within a tier.
        (substantial if area >= 250 else minor).append((d, bx, by, bw_, bh_))
    for cands, reach in ((substantial, 2.0 * half), (minor, 1.5 * half)):
        cands = [c for c in cands if c[0] <= reach]
        if cands:
            cands.sort(key=lambda c: c[0])
            d, bx, by, bw_, bh_ = cands[0]
            break
    else:
        return None
    nx = float(np.clip(bx + bw_ / 2.0 + x0, 0, w - 1))
    # Ground-contact point: the distance model d = f*h/(y - horizon) projects
    # the object's position ON THE GROUND, i.e. the box bottom — the same
    # convention as a person's feet. Centroid would shorten every distance.
    ny = float(np.clip(by + bh_ + y0 - 2.0, 0, h - 1))
    box = [float(bx + x0), float(by + y0), float(bx + bw_ + x0), float(by + bh_ + y0)]
    return nx, ny, box


def read_grays(video_path, indices):
    """Read frames at the given indices, returned as grayscale images."""
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, frame = cap.read()
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok and frame is not None else None)
    cap.release()
    return out


# ---------------------------------------------------------------------
# 3. FULL AUTO OBJECT SELECTION
# ---------------------------------------------------------------------

def auto_select_object(base_video, base_frame_index, other_videos=(),
                       other_frames=None, max_candidates=40):
    """
    Pick a constant-object point in the base view and verify it transfers.

    Returns a dict:
      success, point [x, y], score, details, verified_views [names],
      status ("verified" | "static-only" | "none")
    """
    gray_seq = align_sequence([
        g for g in read_grays(base_video, _static_offsets(base_video, base_frame_index))
        if g is not None
    ])
    base_gray = gray_seq[0] if gray_seq else None
    if base_gray is None:
        return {"success": False, "status": "none", "message": "Could not read the selected frame."}

    # One YOLO pass: persons can never be the constant object (they walk);
    # every OTHER detected object (tripod, dustbin, bench, sign board, …) is
    # a first-class constant-object candidate tried before texture patches.
    detections = detect_objects(cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR))
    persons = None if detections is None else [
        d["box"] for d in detections if d["name"] == "person"
    ]

    def _hits_person(x, y):
        """True when the candidate's CENTRE falls inside a person box.

        A full patch-overlap test rejected every static object that merely
        stood next to (or partially behind) a walking person — bollards,
        tripods and dustbins in crowded plazas — which is why the selector
        kept landing on random background. Walking people still fail the
        filter: their box centre is the person itself.
        """
        if not persons:
            return False
        for bx0, by0, bx1, by1 in persons:
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                return True
        return False

    h, w = base_gray.shape[:2]
    step = int(min(max(min(w, h) // 14, 40), 160))
    xs = range(int(0.08 * w), int(0.92 * w), step)
    ys = range(int(0.30 * h), int(0.94 * h), step)

    scored = []
    for y in ys:
        for x in xs:
            s, parts = candidate_score(base_gray, x, y)
            if s > 0:
                scored.append((s, int(x), int(y), parts))

    # Residual motion (after camera-motion compensation) for EVERY candidate,
    # and an adaptive cap: fixed patches move at most half as much as the
    # scene median, with MOTION_ENERGY_MAX as the absolute floor. Filtering on
    # physics first matters: texture-rich regions are usually crowds, so a
    # top-by-texture cut would keep only patches full of walking people.
    motion_by_pt = {}
    ncc_by_pt = {}
    for _s, x, y, _parts in scored:
        motion_by_pt[(x, y)] = motion_energy(gray_seq, x, y)
        ncc_by_pt[(x, y)] = static_ncc(gray_seq, x, y)
    scene_median_motion = float(np.median(list(motion_by_pt.values()))) if motion_by_pt else 0.0
    strict_cap = max(MOTION_ENERGY_MAX, 0.5 * scene_median_motion)
    relaxed_cap = max(20.0, 0.65 * scene_median_motion)

    def _collect(motion_cap, ncc_min):
        """Physically-static, person-free candidates, ground-level first."""
        ok = []
        for score, x, y, parts in scored:
            motion = motion_by_pt.get((x, y), 255.0)
            if motion > motion_cap:
                continue
            static = ncc_by_pt.get((x, y), -1.0)
            if static < ncc_min:
                # Content inverts across the window: something moved through.
                continue
            if _hits_person(x, y):
                continue
            ok.append((score, x, y, parts, static, motion))
        ok.sort(key=lambda t: (-(t[3].get("ground_n", 0.0)), -t[0], t[1], t[2]))
        return ok[:max_candidates]

    # Two passes: the strict cap keeps only genuinely static patches; the
    # relaxed pass exists because crowded plazas rarely leave a perfectly
    # stable ground-level patch — there the goal (a ground-level object for
    # the distance model) justifies a slightly higher motion tolerance.
    strict_cands = _collect(strict_cap, -0.1)
    relaxed_cands = _collect(relaxed_cap, 0.15)

    # Pre-read the other views' TIME WINDOWS (aligned like the base view) so
    # candidate verification can also check the transferred location is
    # STATIC in the other view: a true constant object is static in every
    # view, while a coincidental match onto a walking person fails there.
    view_variants = []
    view_seqs = {}
    for name, (vid, fid) in (other_frames or {}).items():
        cap = cv2.VideoCapture(str(vid))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps > 0:
            offs = [int(round(s * fps)) for s in STATIC_WINDOW_SECONDS]
        else:
            offs = list(STATIC_WINDOW_FRAMES)
        # The RAW auto-selected frame MUST be first: align_sequence warps the
        # window frames onto frames[0], so every matched/transferred point
        # stays in the coordinates of the frame the UI actually displays.
        # (Previously fid was missing from the window entirely, so transfers
        # belonged to a warped frame from +2 s — boxes landed off-object.)
        fid_c = min(max(0, fid), count - 1) if count > 0 else max(0, fid)
        idxs = [fid_c]
        for o in offs:
            for f in (fid + o, fid - o):
                f = min(max(0, f), count - 1) if count > 0 else max(0, f)
                if f not in idxs:
                    idxs.append(f)
        seq = [g for g in read_grays(vid, idxs) if g is not None]
        seq = align_sequence(seq)
        if not seq:
            continue
        view_seqs[name] = seq
        # Match against the view's base frame and (as fallbacks for timing
        # mismatches) up to two window frames.
        variants = [cv2.cvtColor(seq[0], cv2.COLOR_GRAY2BGR)]
        for extra in seq[1:3]:
            variants.append(cv2.cvtColor(extra, cv2.COLOR_GRAY2BGR))
        view_variants.append((name, variants))

    def _static_in_other(name, px, py):
        """True when the other view's content at (px, py) is static across its
        own time window — required for a genuine common constant object."""
        oseq = view_seqs.get(name)
        if oseq is None:
            return True  # no window available: keep the geometric verdict
        return static_ncc(oseq, float(px), float(py)) >= 0.15

    # ---- Pass 0: real detected objects (tripod, dustbin, bench, …) --------
    # A YOLO-detected non-person object that is static over the window and
    # matches into the other views is exactly the "constant object present
    # in all frames" the analysis needs — preferred over any texture patch.
    def _region_motion(x0, y0, x1, y1):
        """Temporal std-dev over the object's box region across the window."""
        half = max(24, int(0.5 * max(x1 - x0, y1 - y0)))
        cxr, cyr = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        regs = []
        for g in gray_seq[1:]:
            r = _patch(g, cxr, cyr, half=half)
            if r is not None:
                regs.append(r.astype(np.float32))
        if len(regs) < 2:
            return 0.0
        hh = min(r.shape[0] for r in regs)
        ww = min(r.shape[1] for r in regs)
        regs = [cv2.resize(r, (ww, hh)) for r in regs]
        return float(np.stack(regs).std(axis=0).mean())

    other_dets = {}
    for name, variants in view_variants:
        other_dets[name] = detect_objects(variants[0])

    def _same_object_box(name, point, cls_name):
        """Same-class YOLO box in the other view nearest to the matched point."""
        dets = other_dets.get(name)
        if not dets:
            return None
        best, best_d = None, 1e18
        for d in dets:
            bx0, by0, bx1, by1 = d["box"]
            bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            dist = float(np.hypot(bcx - point[0], bcy - point[1]))
            if dist < best_d:
                best_d, best = dist, d
        if best is not None and best_d <= max(0.12 * max(w, h), 140.0):
            return best
        return None

    p0_verified = None
    p0_static = None
    base_cam = Path(base_video).name
    if detections:
        cands0 = []
        for det in detections:
            if det["name"] == "person":
                continue
            bx0, by0, bx1, by1 = det["box"]
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            if _hits_person(cx, cy):
                continue
            motion = _region_motion(bx0, by0, bx1, by1)
            if motion > relaxed_cap:
                continue
            static = static_ncc(gray_seq, cx, cy)
            if static < 0.15:
                continue
            ground = 1.0 if (0.45 * h) <= cy <= 0.92 * h else (
                0.35 if 0.30 * h <= cy < 0.45 * h else 0.0
            )
            cands0.append((det, motion, static, ground))
        # ground-level first, then most-static, then highest YOLO confidence
        cands0.sort(key=lambda t: (-(1 if t[3] >= 1.0 else 0), -t[2], -t[1]["conf"]))

        def _p0_result(det, motion, static, ground, verified, transferred, boxes):
            bx0, by0, bx1, by1 = det["box"]
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            return {
                "success": True,
                "status": "verified" if verified else "static-only",
                "source": "yolo-object",
                "point": [round(float(cx), 1), round(float(cy), 1)],
                "box": [round(float(v), 1) for v in (bx0, by0, bx1, by1)],
                "label": det["name"],
                "frame_index": int(base_frame_index),
                "camera": base_cam,
                "score": round(float(det["conf"]), 4),
                "details": {
                    "class": det["name"], "conf": round(float(det["conf"]), 3),
                    "static_ncc": round(static, 4),
                    "motion_energy": round(motion, 2),
                    "motion_cap": round(relaxed_cap, 2), "ground_n": ground,
                    "person_filter": "yolo" if persons is not None else "unavailable",
                },
                "verified_views": verified,
                "points_by_camera": transferred,
                "boxes_by_camera": boxes,
                "message": (
                    f"Detected a static '{det['name']}' and verified it in all views."
                    if verified else
                    f"Detected a static '{det['name']}' (could not verify it in the "
                    "other views automatically — check the red box, click a different "
                    "fixed object if needed)."
                ),
            }

        for det, motion, static, ground in cands0:
            bx0, by0, bx1, by1 = det["box"]
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            verified, transferred = [], {}
            boxes = {base_cam: {"box": [round(float(v), 1) for v in (bx0, by0, bx1, by1)],
                                "label": det["name"]}}
            for name, variants in view_variants:
                m = local_target_match(base_gray, variants[0], (cx, cy))
                if not m.get("success"):
                    continue
                pb = m.get("point_b")
                if pb is None or not _static_in_other(name, float(pb[0]), float(pb[1])):
                    continue
                verified.append(name)
                if pb:
                    transferred[name] = [round(float(pb[0]), 1), round(float(pb[1]), 1)]
                    ob = _same_object_box(name, (float(pb[0]), float(pb[1])), det["name"])
                    boxes[name] = (
                        {"box": [round(float(v), 1) for v in ob["box"]], "label": det["name"]}
                        if ob else
                        {"point": [round(float(pb[0]), 1), round(float(pb[1]), 1)], "label": det["name"]}
                    )
            if verified:
                res = _p0_result(det, motion, static, ground, verified, transferred, boxes)
                if p0_verified is None or (
                    ground >= 1.0 and (p0_verified["details"].get("ground_n", 0.0) < 1.0)
                ):
                    p0_verified = res
                if ground >= 1.0:
                    break  # ideal: verified ground-level object
            elif p0_static is None:
                p0_static = _p0_result(det, motion, static, ground, [], transferred, boxes)

    if p0_verified is not None:
        return p0_verified

    best_static_only = None  # fallback: static + person-free but unverified
    best_ground = None       # verified AND ground-level (best for distances)
    best_any = None          # verified, any height

    def _make_result(score, x, y, parts, static, motion, cap, verified,
                     transferred, tier):
        is_ground = parts.get("ground_n", 0.0) >= 1.0
        if verified:
            status = "verified"
            message = (
                "Selected a static, ground-level object verified in the other views."
                if is_ground else
                "Automatically selected a static point verified in the other views."
            )
        else:
            status = "static-only"
            message = (
                "Selected a static, person-free point (could not verify it in the "
                "other views automatically — check the red box and click a different "
                "fixed object if needed)."
            )
        return {
            "success": True,
            "status": status,
            "point": [x, y],
            "frame_index": int(base_frame_index),
            "camera": Path(base_video).name,
            "score": round(float(score), 4),
            "details": {**parts, "static_ncc": round(static, 4),
                        "motion_energy": round(motion, 2),
                        "motion_cap": round(cap, 2),
                        "scene_median_motion": round(scene_median_motion, 2),
                        "person_filter": "yolo" if persons is not None else "unavailable",
                        "pass": tier},
            "verified_views": verified,
            "points_by_camera": transferred,
            "message": message,
        }

    def _verify_point(px, py):
        """Try to transfer (px, py) into every other view; the transferred
        spot must (a) be STATIC in that view's own time window and (b) map
        BACK onto the original point (round-trip/cycle consistency).
        (b) is what separates the same physical object from a look-alike:
        rows of identical bins match each other by appearance, but only the
        true correspondence survives the return trip."""
        verified, transferred = [], {}
        base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        for name, variants in view_variants:
            for other_bgr in variants:
                m = local_target_match(base_gray, other_bgr, (px, py))
                if m.get("success"):
                    pb = m.get("point_b")
                    if pb is None or not _static_in_other(name, float(pb[0]), float(pb[1])):
                        # Geometric match exists but lands on DYNAMIC
                        # content in the other view: not a constant object.
                        continue
                    m_back = local_target_match(other_bgr, base_bgr, (float(pb[0]), float(pb[1])))
                    back = m_back.get("point_b") if m_back.get("success") else None
                    if back is None or float(np.hypot(back[0] - px, back[1] - py)) > 1.5 * PATCH_HALF:
                        # Look-alike: forward match exists but the round
                        # trip lands elsewhere (twin object) — reject.
                        continue
                    verified.append(name)
                    if pb:
                        transferred[name] = [round(float(pb[0]), 1), round(float(pb[1]), 1)]
                    break
        return verified, transferred

    for tier, cap, cands in (("strict", strict_cap, strict_cands),
                             ("relaxed", relaxed_cap, relaxed_cands)):
        for score, x, y, parts, static, motion in cands:
            verified, transferred = _verify_point(x, y)
            if not verified and parts.get("ground_n", 0.0) >= 1.0:
                # Ground-level candidates are exactly what the distance model
                # needs; a near-miss on the SIFT inlier knife-edge (the grid
                # centre may sit a few px off the object) gets half-step
                # retries before giving up on the patch.
                half_step = max(24, step // 2)
                for dy in (-half_step, 0, half_step):
                    for dx in (-half_step, 0, half_step):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if not (0 <= nx < w and 0 <= ny < h) or _hits_person(nx, ny):
                            continue
                        verified, transferred = _verify_point(nx, ny)
                        if verified:
                            x, y = nx, ny
                            break
                    if verified:
                        break

            if other_videos and not verified:
                # Not verifiable in the other views (e.g. a textureless patch
                # the matcher cannot transfer). Keep as a last-resort fallback
                # but prefer anything that verifies.
                if best_static_only is None:
                    best_static_only = _make_result(
                        score, x, y, parts, static, motion, cap, [], {}, tier)
                continue

            result = _make_result(score, x, y, parts, static, motion, cap,
                                  verified, transferred, tier)
            is_ground = parts.get("ground_n", 0.0) >= 1.0
            if is_ground and best_ground is None:
                best_ground = result
                break  # verified ground-level object: the ideal outcome
            if best_any is None:
                best_any = result
        # A ground-level verified object is the ideal outcome; stop widening
        # the motion tolerance once we have one.
        if best_ground is not None:
            break

    # Snap the chosen point onto the physical object it sits on: in the
    # temporal-median image walking people have vanished, so the tight
    # gradient blob around the point IS the constant object (dustbin, tripod
    # base, bollard, sign post…). The red UI box then outlines the real
    # object instead of a generic square around a bare pixel.
    median_img = temporal_median(gray_seq)

    def _with_box(result):
        if not result or not result.get("success") or result.get("box"):
            return result
        px, py = result["point"]
        snap = snap_to_object(median_img, float(px), float(py))
        if not snap:
            return result
        nx, ny, box = snap
        # The snapped (ground-contact) point must STILL transfer into the
        # other views — the cross-view common-object guarantee beats the
        # nicer box. Without this re-check the stored per-view transfer
        # would silently belong to the old pre-snap point.
        verified, transferred = _verify_point(nx, ny)
        if not verified:
            half_step = max(24, step // 2)
            for dy in (-half_step, 0, half_step):
                for dx in (-half_step, 0, half_step):
                    if dx == 0 and dy == 0:
                        continue
                    tx, ty = nx + dx, ny + dy
                    if not (0 <= tx < w and 0 <= ty < h) or _hits_person(tx, ty):
                        continue
                    verified, transferred = _verify_point(tx, ty)
                    if verified:
                        nx, ny = float(tx), float(ty)
                        break
                if verified:
                    break
        if verified:
            result["point"] = [round(nx, 1), round(ny, 1)]
            result["box"] = [round(float(v), 1) for v in box]
            result["verified_views"] = verified
            result["points_by_camera"] = transferred
            result["status"] = "verified"
        # else: keep the original verified point + transfers untouched.
        return result

    if best_ground is not None:
        return _with_box(best_ground)
    if best_any is not None:
        return _with_box(best_any)
    if p0_static is not None:
        return _with_box(p0_static)
    if best_static_only is not None:
        return _with_box(best_static_only)

    return {
        "success": False,
        "status": "none",
        "message": "No reliable constant object found automatically; click one manually in Camera 01.",
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automatic frame + constant-object selection")
    parser.add_argument("videos", nargs="?", default="input_videos")
    parser.add_argument("--output", default="results/auto_selection.json")
    args = parser.parse_args()

    videos = discover_videos(Path(args.videos))
    if not videos:
        print("[AUTO-SELECT] No videos found.")
        return 1

    frames = {}
    for v in videos:
        idx, score = auto_select_frame(v)
        frames[v.name] = {"path": str(v), "frame_index": idx, "score": round(score, 4)}
        print(f"[AUTO-SELECT] {v.name}: best frame {idx} (score {score:.3f})")

    base = videos[0]
    others = {v.name: (v, frames[v.name]["frame_index"]) for v in videos[1:]}
    obj = auto_select_object(base, frames[base.name]["frame_index"], videos[1:], others)

    result = {"videos": frames, "object": obj}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if obj.get("success"):
        print(f"[AUTO-SELECT] Object at {obj['point']} in {obj['camera']} "
              f"({obj['status']}, score {obj['score']})")
    else:
        print(f"[AUTO-SELECT] {obj.get('message')}")
    print(f"[AUTO-SELECT] Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
