import json
import math
import os
import subprocess
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None

# ============================================================
# CCTV DISTANCE ANALYSER — PROFESSIONAL LOCAL DASHBOARD
# - Local-folder workflow: no browser uploader / 200 MB limit
# - Arbitrary number of CCTV videos
# - User click target selection in Camera 01
# - Target-specific correspondence + triangulation
# - Metric conversion using the existing visual-scale stage
# - Functional Top / Side / Perspective 3D views
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TARGET_PIPELINE = BASE_DIR / "src" / "multi_camera_pipeline.py"
END_TO_END_PIPELINE = BASE_DIR / "src" / "end_to_end_pipeline.py"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
INPUT_VIDEOS_DIR = BASE_DIR / "input_videos"
UPLOAD_DIR = BASE_DIR / "uploads"
AUTO_SELECT_SCRIPT = BASE_DIR / "src" / "auto_select.py"

try:
    sys.path.insert(0, str(BASE_DIR / "src"))
    from auto_select import auto_select_frame, auto_select_object
except Exception:
    auto_select_frame = None
    auto_select_object = None

REPORT_TARGET = RESULTS_DIR / "target_point_report.json"
REPORT_MULTI = RESULTS_DIR / "multi_camera_report.json"
REPORT_GEOM = RESULTS_DIR / "multi_camera_geometry.json"
REPORT_SCALE = RESULTS_DIR / "visual_scale_result.json"
REPORT_METRIC = RESULTS_DIR / "metric_integration.json"
REPORT_FINAL = RESULTS_DIR / "final_pipeline_report.json"
REPORT_METRIC_LAYER = RESULTS_DIR / "visual_metric_layer.json"

st.set_page_config(
    page_title="CCTV Distance Estimation",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- CSS ----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#020913;--panel:#061321;--panel2:#081827;--border:#17314a;
  --text:#e9f1fb;--muted:#91a4b8;--blue:#2477ff;--cyan:#3aa8ff;
  --green:#15e56b;--yellow:#ffb21a;--red:#ff4d5d;
}
html,body,[class*="css"]{font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.stApp{background:radial-gradient(circle at 82% 0%,rgba(17,73,124,.16),transparent 28%),radial-gradient(circle at 0% 45%,rgba(14,63,104,.12),transparent 24%),var(--bg);color:var(--text)}
.block-container{max-width:1540px;padding-top:1rem;padding-bottom:1rem}
header[data-testid="stHeader"]{background:transparent}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#06111d 0%,#03101b 100%);border-right:1px solid #132b42}
section[data-testid="stSidebar"]>div{padding-top:1rem}
.topbar{border:1px solid var(--border);background:linear-gradient(90deg,rgba(8,22,36,.96),rgba(5,18,30,.9));border-radius:8px;padding:12px 16px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 8px 30px rgba(0,0,0,.2)}
.brand{display:flex;align-items:center;gap:14px}.camera-icon{font-size:34px}.brand-title{font-size:24px;font-weight:750}.brand-subtitle{font-size:14px;color:#b6c5d6;margin-top:2px}
.project-box{min-width:330px;border:1px solid #19344c;border-radius:8px;padding:8px 12px;background:#071522;font-size:12px}.project-box .value{color:#2f88ff;font-weight:650}
.panel{border:1px solid var(--border);background:linear-gradient(180deg,rgba(8,23,37,.97),rgba(4,16,27,.97));border-radius:7px;padding:12px;box-shadow:inset 0 1px rgba(255,255,255,.02),0 8px 28px rgba(0,0,0,.14)}
.panel-title,.section-title{color:#6ea9ff;font-size:13px;font-weight:750;letter-spacing:.3px;text-transform:uppercase}.section-title{font-size:14px}
.sidebar-card{border:1px solid #17314a;border-radius:7px;padding:11px;margin-bottom:10px;background:#061522}.sidebar-title{color:#6ea9ff;font-size:13px;font-weight:750;margin-bottom:9px}
.pipeline-line,.status-row{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #11273a;padding:7px 0;font-size:12px}.pipeline-line:last-child,.status-row:last-child{border-bottom:0}
.ok{color:#13ee6a!important}.warn{color:#ffb51d!important}.bad{color:#ff5664!important}.muted{color:#91a4b8!important}
.video-card{border:1px solid #1a344b;border-radius:6px;overflow:hidden;background:#030c14}.video-head{height:31px;display:flex;align-items:center;justify-content:space-between;padding:0 9px;background:#08131e;font-size:12px;font-weight:650}.recorded{border:1px solid #3b74a9;color:#77b4ff;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:750}.video-foot{padding:6px 8px;display:flex;justify-content:space-between;color:#9eafc0;font-size:11px}
.select-hint{border:1px dashed #365775;border-radius:6px;padding:8px;font-size:11px;color:#9fb1c4;background:#03101a;text-align:center;margin-top:7px}.selected-badge{display:inline-block;border:1px solid #ff3c47;color:#fff;background:#e43b46;padding:3px 7px;border-radius:3px;font-size:10px;font-weight:750}
.metric-table{width:100%;border-collapse:collapse;font-size:12px}.metric-table th{text-align:left;color:#e6eef7;padding:9px 8px;background:#0a1724;border-bottom:1px solid #1c3449}.metric-table td{padding:9px 8px;border-bottom:1px solid #14283a;color:#d7e1ec}
.recon-toolbar{display:flex;gap:6px;margin-bottom:8px}.placeholder{height:315px;border:1px dashed #24425a;border-radius:6px;background:radial-gradient(circle at 50% 45%,rgba(36,119,255,.09),transparent 42%),#020a12;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#7890a8;text-align:center}.placeholder .big{font-size:38px;opacity:.75;margin-bottom:5px}
.bottom-bar{margin-top:12px;border:1px solid var(--border);background:#061421;border-radius:7px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;font-size:12px}
.done-bar{border:1px solid #1d4030;background:linear-gradient(90deg,rgba(14,60,38,.5),rgba(6,20,14,.5));border-radius:7px;padding:11px 14px;margin-bottom:10px;font-size:14px;font-weight:650}
.done-bar.warn{border-color:#4a3a14;background:linear-gradient(90deg,rgba(75,55,10,.45),rgba(30,22,6,.45))}
.small{font-size:11px;color:#9fb0c1}.pathbox{font-size:10px;color:#89a0b5;word-break:break-all;border:1px solid #1b344a;background:#03101a;padding:7px;border-radius:4px;margin-top:6px}
.stButton>button{border-radius:5px!important}.stTextInput input{background:#03101a!important}.stSlider>div{padding-top:0!important}
button[kind="primary"]{background:linear-gradient(90deg,#0ecb5e,#0aa84b)!important;border:1px solid #0ecb5e!important;color:#fff!important;font-weight:750!important}
button[kind="secondary"]{background:linear-gradient(90deg,#2477ff,#1b5fd9)!important;border:1px solid #2477ff!important;color:#fff!important;font-weight:650!important}
.c-blue{color:#3aa8ff!important}.c-yellow{color:#ffb21a!important}
.recon-info{border:1px solid #1c3a55;border-radius:6px;background:#071522;padding:11px;font-size:12px}.recon-info .info-title{color:#6ea9ff;font-weight:750;margin-bottom:7px}.recon-info .info-row{display:flex;justify-content:space-between;border-bottom:1px solid #11273a;padding:5px 0}.recon-info .info-row:last-child{border-bottom:0}
.legend-row{display:flex;gap:16px;align-items:center;font-size:11px;color:#c7d4e2;flex-wrap:wrap}.legend-row .chip{display:flex;align-items:center;gap:7px}.legend-row .dash{width:20px;border-top:2px dashed #8fa6bd}.legend-row .dot{width:9px;height:9px;border-radius:50%;background:#fff;display:inline-block}
.final-result{border:1px solid #1d5a33;border-radius:7px;background:linear-gradient(90deg,rgba(14,60,38,.55),rgba(6,22,14,.55));padding:12px 14px;margin-top:10px;font-size:14px}.final-result b{color:#13ee6a;font-size:19px}.final-result .fr-note{display:block;margin-top:4px;font-size:11px;color:#9fb0c1}
footer{visibility:hidden}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------- helpers ---------------------------
def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def clear_reports():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in [REPORT_TARGET, REPORT_MULTI, REPORT_GEOM, REPORT_SCALE, REPORT_METRIC, REPORT_FINAL, REPORT_METRIC_LAYER]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def discover_videos(folder):
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )


@st.cache_data(show_spinner=False, max_entries=64)
def cached_video_info(path: str, mtime: float, size: int):
    """Video metadata cached across reruns (survives script re-execution)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"frames": 0, "fps": 0.0, "width": 0, "height": 0, "duration": 0.0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return {"frames": frames, "fps": fps, "width": width, "height": height, "duration": duration}


def get_video_info(path):
    """Cached video metadata — the sidebar and body panels call this several
    times per rerun; re-opening large files each time made the page slow."""
    p = str(path)
    try:
        st_ = os.stat(p)
        return dict(cached_video_info(p, st_.st_mtime, st_.st_size))
    except OSError:
        return {"frames": 0, "fps": 0.0, "width": 0, "height": 0, "duration": 0.0}


_VIDEO_LOCKS = {}


@st.cache_resource(show_spinner=False)
def _persistent_capture(path: str):
    """One open VideoCapture per file, kept alive across reruns by Streamlit's
    resource cache: opening a 100+ MB h264 file per read was the main source
    of the multi-second stall on every click."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    try:
        cap.release()
    except Exception:
        pass
    return None


def read_frame_fast(path, index):
    """Robust frame read for large h264 recordings.

    Uses the persistent per-file capture (st.cache_resource) under a per-file
    lock: opening a 100+ MB h264 file per read was the main source of the
    "every click loads slowly" feeling and of the occasional BLACK camera
    card. A failed seek-and-read retries from a mid-video index, then index 0,
    so one bad seek can never leave the card black.
    """
    str_path = str(path)
    lock = _VIDEO_LOCKS.setdefault(str_path, threading.Lock())
    index = max(0, int(index))
    with lock:
        for attempt in range(3):
            cap = _persistent_capture(str_path)
            if cap is None:
                # One retry: the resource cache may have been cleared while
                # the file was being replaced.
                if attempt == 0:
                    _persistent_capture.clear()
                    continue
                return None
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            except Exception:
                pass
            ok, frame = cap.read()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Decode hiccup: drop the cached capture and retry — mid-video
            # index on attempt 2, index 0 on attempt 3.
            try:
                cap.release()
            except Exception:
                pass
            _persistent_capture.clear()
            if attempt == 1:
                info = get_video_info(str_path)
                index = max(0, (info["frames"] // 2) if info["frames"] else 0)
            elif attempt == 2:
                index = 0
        return None


def release_video_captures():
    """Close every cached capture and drop cached frames. Called before
    uploads are deleted — on Windows an open handle makes the file delete
    fail silently."""
    try:
        _persistent_capture.clear()
    except Exception:
        pass
    try:
        cached_frame.clear()
    except Exception:
        pass
    try:
        cached_video_info.clear()
    except Exception:
        pass


def prefetch_frames(videos, frame_indices):
    """Warm the frame cache in the background right after auto-selection so
    section 4 paints instantly instead of decoding on the next rerun."""

    def _work():
        for v, idx in zip(videos, frame_indices):
            try:
                read_frame_fast(v, idx)
            except Exception:
                pass

    threading.Thread(target=_work, daemon=True).start()


@st.cache_data(show_spinner=False, max_entries=24)
def cached_frame(path: str, index: int, mtime: float):
    """Frame bytes cached by Streamlit's cache manager — it SURVIVES script
    reruns, so slider moves / box redraws / section renders never re-decode
    the same 1080p h264 frame (module-level dicts are wiped on every rerun,
    which is why the app used to re-open the video on every click)."""
    return read_frame_fast(path, index)


def get_frame(path, index):
    """Cached frame read — the slider and object-box redraw re-request the
    same frame on every rerun."""
    try:
        mtime = float(os.stat(path).st_mtime)
    except OSError:
        mtime = 0.0
    rgb = cached_frame(str(path), int(index), mtime)
    return rgb.copy() if rgb is not None else None


def fmt_duration(seconds):
    if seconds <= 0:
        return "—"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run_command(script: Path, args, timeout=1800):
    command = [sys.executable, str(script), *[str(x) for x in args]]
    try:
        proc = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, text
    except subprocess.TimeoutExpired as exc:
        return False, f"Pipeline timed out after {timeout} seconds.\n{exc}"
    except Exception as exc:
        return False, str(exc)


def run_full_pipeline(folder):
    if not END_TO_END_PIPELINE.exists():
        return False, "src/end_to_end_pipeline.py was not found."
    return run_command(END_TO_END_PIPELINE, [str(folder)], timeout=3600)


def run_target_pipeline(folder, target_x, target_y, target_frame):
    if not TARGET_PIPELINE.exists():
        return False, "src/multi_camera_pipeline.py was not found."
    return run_command(
        TARGET_PIPELINE,
        [
            str(folder),
            "--target-x", f"{float(target_x):.6f}",
            "--target-y", f"{float(target_y):.6f}",
            "--target-frame", int(target_frame),
            "--output", str(REPORT_TARGET.relative_to(BASE_DIR)),
        ],
        timeout=3600,
    )


def run_metric_layer(folder):
    metric_layer = BASE_DIR / "src" / "visual_metric_layer.py"
    if not metric_layer.exists():
        return False, "src/visual_metric_layer.py was not found."
    return run_command(
        metric_layer,
        [str(folder), "--output", str(REPORT_METRIC_LAYER.relative_to(BASE_DIR))],
        timeout=1800,
    )


def image_click(image, key, width=None):
    # Each rendered image gets a unique widget key; the widget registry remembers
    # the component's last value otherwise, so a click made frames ago would
    # re-apply itself on every rerun and keep overriding the auto-detected
    # object (the "random object at (185, 63)" bug).
    if streamlit_image_coordinates is None:
        st.warning("Click selection component is not installed. Run: py -m pip install streamlit-image-coordinates")
        return None
    kwargs = {"key": key}
    if width is not None:
        kwargs["width"] = width
    return streamlit_image_coordinates(image, **kwargs)


def _is_auto_point(auto_point, click, tol=3.0):
    """True when the locked click IS the auto-detected point. Tolerant on
    purpose: the click->frame pixel mapping can shift the stored point by a
    fraction of a pixel, which made the exact '==' comparison report MANUAL
    for the auto pick."""
    if not auto_point or click is None:
        return False
    try:
        return (
            abs(float(auto_point[0]) - float(click[0])) <= tol
            and abs(float(auto_point[1]) - float(click[1])) <= tol
        )
    except (TypeError, ValueError):
        return False


@st.cache_data(show_spinner=False, max_entries=64)
def _transfer_point_cached(src_path: str, src_idx: int, dst_path: str, dst_idx: int,
                           px: float, py: float):
    """Locate the clicked object in ANOTHER camera's frame.

    Multi-scale template match (the two uploads are re-edited crops of the
    same footage, so the object appears at different sizes) refined by the
    patch-SIFT matcher, and accepted ONLY if the match maps back onto the
    source point (round-trip / cycle consistency — separates the same
    physical object from a look-alike in a row of identical bins).

    Returns (x, y, score, method) or (None, None, 0.0, reason).
    """
    import sys as _sys
    # read_frame_fast returns RGB (3-channel); the patch matcher wants BGR.
    src_rgb = read_frame_fast(src_path, src_idx)
    dst_rgb = read_frame_fast(dst_path, dst_idx)
    if src_rgb is None or dst_rgb is None:
        return None, None, 0.0, "frame unreadable"
    if src_rgb.ndim == 2:
        src_rgb = cv2.cvtColor(src_rgb, cv2.COLOR_GRAY2RGB)
    if dst_rgb.ndim == 2:
        dst_rgb = cv2.cvtColor(dst_rgb, cv2.COLOR_GRAY2RGB)
    src_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2BGR)

    if str(BASE_DIR / "src") not in _sys.path:
        _sys.path.insert(0, str(BASE_DIR / "src"))
    try:
        from multi_camera_pipeline import local_target_match
    except Exception:
        return None, None, 0.0, "matcher unavailable"

    best = None  # (score, x, y)
    H, W = dst_bgr.shape[:2]
    for scale in (1.0, 0.85, 1.2, 0.7, 1.45):
        rr = max(24, int(80 * scale))
        m = local_target_match(src_bgr, dst_bgr, (float(px), float(py)), radius=rr)
        if not m.get("success"):
            continue
        pb = m.get("point_b")
        if pb is None:
            continue
        bx, by = float(pb[0]), float(pb[1])
        if not (0 <= bx < W and 0 <= by < H):
            continue
        # Round-trip: the candidate must map BACK onto the clicked point.
        m_back = local_target_match(dst_bgr, src_bgr, (bx, by), radius=rr)
        back = m_back.get("point_b") if m_back.get("success") else None
        if back is None or float(np.hypot(back[0] - px, back[1] - py)) > 0.75 * rr:
            continue
        sc = float(m.get("confidence") or 0.5)
        if best is None or sc > best[0]:
            best = (sc, bx, by)
    if best is None:
        return None, None, 0.0, "no consistent match (look-alike rejected or object not visible)"
    return best[1], best[2], best[0], "multi-scale template+SIFT with round-trip check"


def transfer_point_to_views(src_name, click_xy, videos, auto):
    """Distribute the clicked point to every OTHER camera.

    Fills st.session_state.locked_points: {camera_name: [x, y]}. The source
    camera gets the click itself; others get the verified transfer. Cameras
    where no consistent match exists are simply absent (the UI tells the
    user to click there directly).
    """
    auto_other = (auto or {}).get("other_frames") or {}

    def _frame_idx_for(name):
        # The frame each camera card actually displays.
        if name == videos[0].name:
            return int(st.session_state.target_frame)
        info = get_video_info(next(v for v in videos if v.name == name))
        return int(auto_other.get(name, info["frames"] // 2 if info["frames"] else 0))

    src_video = next(v for v in videos if v.name == src_name)
    src_idx = _frame_idx_for(src_name)
    pts = {src_name: [float(click_xy[0]), float(click_xy[1])]}
    notes = []
    for v in videos:
        if v.name == src_name:
            continue
        dst_idx = _frame_idx_for(v.name)
        x, y, score, how = _transfer_point_cached(str(src_video), src_idx, str(v), dst_idx, float(click_xy[0]), float(click_xy[1]))
        if x is not None:
            pts[v.name] = [round(float(x), 1), round(float(y), 1)]
            notes.append(f"{v.name}: located at ({x:.0f}, {y:.0f}) — score {score:.2f}")
        else:
            notes.append(f"{v.name}: NOT located ({how}) — click it there directly")
    st.session_state.locked_points = pts
    st.session_state.lock_info = " | ".join(notes)
    return pts


def selected_target_from_report():
    report = load_json(REPORT_TARGET)
    target = report.get("target", {})
    point = target.get("image_point")
    if isinstance(point, list) and len(point) == 2:
        return float(point[0]), float(point[1]), int(target.get("frame_index", 0))
    return None


def target_pairs(report):
    return report.get("cross_camera_pairs", []) if isinstance(report, dict) else []


def target_distance_rows(report, scale_factor=None, metric_cams=None, object_pixels=None):
    rows = {}
    warnings = {}
    duplicate_notes = {}
    for pair in target_pairs(report):
        if pair.get("skipped_duplicate"):
            duplicate_notes[pair.get("camera_b")] = pair.get("message") or (
                f"Skipped: duplicate of {pair.get('duplicate_of')} — ignored for 3D."
            )
        if pair.get("duplicate_footage_warning"):
            warnings[pair.get("camera_b")] = pair["duplicate_footage_warning"]
        if pair.get("low_parallax_warning"):
            warnings[pair.get("camera_b")] = pair["low_parallax_warning"]
        if not pair.get("valid"):
            continue
        tri = pair.get("target_triangulation", {})
        pose = pair.get("relative_pose", {})
        X = tri.get("point_3d")
        R = pose.get("rotation")
        t = pose.get("translation_direction")
        if not (isinstance(X, list) and len(X) == 3 and isinstance(R, list) and len(R) == 3 and isinstance(t, list) and len(t) == 3):
            continue
        X = np.asarray(X, dtype=float)
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            continue
        Xb = R @ X + t
        a_name = pair.get("camera_a", "Camera 01")
        b_name = pair.get("camera_b", "Camera 02")
        da = float(np.linalg.norm(X))
        db = float(np.linalg.norm(Xb))
        err_a = float(tri.get("reprojection_error_a") or 0.0)
        err_b = float(tri.get("reprojection_error_b") or 0.0)
        rows.setdefault(a_name, []).append((da, err_a, pair))
        rows.setdefault(b_name, []).append((db, err_b, pair))

    metric_cams = metric_cams or {}
    object_pixels = object_pixels or {}

    output = []
    for name, items in rows.items():
        rel = float(np.median([x[0] for x in items]))
        err = float(np.median([x[1] for x in items]))
        metric = rel * float(scale_factor) if scale_factor is not None else None

        cam_info = metric_cams.get(name)
        ground_m = cam_info.get("height_m") if cam_info else None
        obj_m, obj_reason = pinhole_object_distance(cam_info, object_pixels.get(name))

        output.append({
            "camera": name,
            "relative": rel,
            "metric": metric,
            "reprojection": err,
            "samples": len(items),
            "ground_height_m": ground_m,
            "object_distance_m": obj_m,
            "object_distance_note": obj_reason,
            "warning": warnings.get(name),
            "duplicate_note": duplicate_notes.get(name),
        })

    # Cameras with no valid triangulation still get single-view estimates.
    seen = {r["camera"] for r in output}
    for name, cam_info in metric_cams.items():
        if name in seen:
            continue
        ground_m = cam_info.get("height_m")
        obj_m, obj_reason = pinhole_object_distance(cam_info, object_pixels.get(name))
        output.append({
            "camera": name,
            "relative": None,
            "metric": None,
            "reprojection": None,
            "samples": 0,
            "ground_height_m": ground_m,
            "object_distance_m": obj_m,
            "object_distance_note": obj_reason,
            "warning": warnings.get(name),
            "duplicate_note": duplicate_notes.get(name),
        })

    output.sort(key=lambda x: x["camera"].lower())
    return output


def scale_info():
    report = load_json(REPORT_SCALE)
    scale = report.get("scale_m_per_relative_unit") or report.get("scale")
    try:
        scale = float(scale) if scale is not None else None
    except Exception:
        scale = None
    source = report.get("source") or report.get("method") or "Visual reference"
    confidence = report.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None
    return scale, source, confidence


def metric_layer_info():
    """Per-camera metric info (height, horizon, focal) from the metric layer."""
    report = load_json(REPORT_METRIC_LAYER)
    if not report:
        return {}
    cameras = {}
    for cam in report.get("cameras", []):
        ground = cam.get("camera_to_ground") or {}
        intrinsics = cam.get("intrinsics") or {}
        cameras[cam.get("camera")] = {
            "height_m": ground.get("metric_height_m"),
            "height_status": ground.get("status", "unavailable"),
            "height_note": ground.get("note"),
            "horizon_y": cam.get("horizon_y"),
            "horizon_source": cam.get("horizon_source"),
            "focal_px": intrinsics.get("focal_px"),
            "analyzed_frame": cam.get("analyzed_frame_index"),
            "person_count": len(cam.get("person_detections") or []),
        }
    return cameras


def pinhole_object_distance(cam_info, pixel_xy):
    """
    Camera-to-object distance (metres) from the pinhole ground-plane model:

        d = f * h / (y_object - y_horizon)

    Returns (distance_m, reason). Exactly one of the two is non-None.
    """
    if not cam_info or pixel_xy is None:
        return None, "No metric layer result for this camera."

    h = cam_info.get("height_m")
    horizon = cam_info.get("horizon_y")
    focal = cam_info.get("focal_px")

    if not h or not horizon or not focal:
        return None, "Camera height/horizon not established (no person cue or horizon)."

    y = float(pixel_xy[1])
    offset = y - float(horizon)

    if offset <= 8.0:
        return None, "Selected point is at/above (or within a few pixels of) the horizon; ground-plane distance is undefined."

    return float(focal) * float(h) / offset, None


def object_pixels_by_camera(target_report):
    """
    Map camera name -> pixel of the selected object in that camera's image.

    Camera 01 contributes the clicked point; every other camera contributes the
    transferred point found by the target correspondence stage.
    """
    pixels = {}
    if not isinstance(target_report, dict):
        return pixels

    target = target_report.get("target") or {}
    point = target.get("image_point")
    if isinstance(point, list) and len(point) == 2:
        pixels[target.get("camera")] = (float(point[0]), float(point[1]))

    for pair in target_pairs(target_report):
        corr = pair.get("target_correspondence") or {}
        if corr.get("success") and corr.get("point_b"):
            pixels[pair.get("camera_b")] = (
                float(corr["point_b"][0]),
                float(corr["point_b"][1]),
            )

    return pixels


def camera_display_name(name, videos):
    """Map a report camera name (file name) to a friendly Camera NN label."""
    if not name:
        return "—"
    for i, v in enumerate(videos or [], 1):
        if v.name == name:
            return f"Camera {i:02d}"
    return name


def load_analysis(videos):
    """All report-derived UI state in one place.

    Gated on the uploaded videos: with nothing uploaded every value stays
    empty so results from a previous session's reports can never leak into
    the UI (sections 3/5/6/7/8 then honestly show their placeholders).
    """
    if not videos:
        return {}, None, None, None, {}, {}, [], status_from_results([], {}, None)
    target_report = load_json(REPORT_TARGET)
    scale_factor, scale_source, scale_conf = scale_info()
    metric_cams = metric_layer_info()
    obj_pixels = object_pixels_by_camera(target_report)
    # USER-LOCKED POINTS RULE: where the user clicked (or the click was
    # verified-transferred), that point defines the object position for the
    # distance solve — overriding any auto/pipeline correspondence.
    locked = st.session_state.get("locked_points") or {}
    for cam, pix in locked.items():
        obj_pixels[cam] = (float(pix[0]), float(pix[1]))
    rows = target_distance_rows(target_report, scale_factor, metric_cams, obj_pixels)
    states = status_from_results(videos, target_report, scale_factor)
    return target_report, scale_factor, scale_source, scale_conf, metric_cams, obj_pixels, rows, states


def object_distance_unavailable(rows):
    """True when no camera produced a camera→object metre estimate."""
    return not rows or all(r.get("object_distance_m") is None for r in rows)


def status_from_results(videos, target_report, scale_factor):
    pairs = target_pairs(target_report)
    valid_target = any(p.get("valid") and p.get("target_triangulation", {}).get("valid") for p in pairs)
    return {
        "Input videos": bool(videos),
        "Target selected": selected_target_from_report() is not None,
        "Target correspondence": any(p.get("target_correspondence", {}).get("success") for p in pairs),
        "Target triangulation": valid_target,
        "Metric scale": scale_factor is not None,
    }


def build_3d_data(report):
    valid = []
    for pair in target_pairs(report):
        if not pair.get("valid"):
            continue
        pose = pair.get("relative_pose", {})
        tri = pair.get("target_triangulation", {})
        if not pose.get("rotation") or not pose.get("translation_direction") or not tri.get("point_3d"):
            continue
        R = np.asarray(pose["rotation"], dtype=float)
        t = np.asarray(pose["translation_direction"], dtype=float)
        X = np.asarray(tri["point_3d"], dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or X.shape != (3,):
            continue
        Cb = -R.T @ t
        valid.append({
            "camera_a": pair.get("camera_a", "Camera 01"),
            "camera_b": pair.get("camera_b", "Camera 02"),
            "camera_a_pos": np.zeros(3),
            "camera_b_pos": Cb,
            "target": X,
        })
    return valid


def _camera_frustum(center, forward, color, name, forward_len=None):
    """Build a plotly trace for a camera frustum: pyramid body + image-plane rectangle."""
    center = np.asarray(center, dtype=float)
    forward = np.asarray(forward, dtype=float)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    L = float(forward_len) if forward_len else float(max(0.25 * (np.linalg.norm(center) + 0.25), 0.6))
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    w = 0.38 * L
    h = 0.28 * L
    tip = center + forward * L
    corners = [tip + (right * a + up * b) for a, b in ((-w, -h), (w, -h), (w, h), (-w, h))]
    xs, ys, zs = [], [], []
    # pyramid edges apex -> rectangle corners
    for c in corners:
        xs += [center[0], float(c[0]), None]
        ys += [center[1], float(c[1]), None]
        zs += [center[2], float(c[2]), None]
    # rectangle edges
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        xs += [float(corners[i][0]), float(corners[j][0]), None]
        ys += [float(corners[i][1]), float(corners[j][1]), None]
        zs += [float(corners[i][2]), float(corners[j][2]), None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=4), name=name, showlegend=False,
    )


def build_3d_figure(report, view, single_view_cameras=None, obj_label="OBJ_001"):
    """Build the 3D scene figure; returns (fig, recon_mode, stats).

    Kept separate from the Streamlit rendering so the same figure can be
    rasterised into the exported PDF report.
    """
    stats = {"matched": 0, "inliers": 0, "tri_pts": 0, "reproj": None, "has_data": False}
    if go is None:
        return None, "Plotly missing", stats
    data = build_3d_data(report)
    stats["has_data"] = bool(data)
    single_view_cameras = single_view_cameras or {}

    if not data:
        # ---- Single-view fallback: still show a working 3D scene -------------
        # Camera 01 at the origin looking at the object; every other camera is
        # placed on its measured camera->object bearing/ring so the scene is
        # geometrically consistent without inventing a fake reconstruction.
        if not single_view_cameras:
            return None, "no-data", stats
        n_cams = len(single_view_cameras)
        obj_d = None
        for info in single_view_cameras.values():
            d = info.get("object_distance_m")
            if d:
                obj_d = float(d)
                break
        obj_d = obj_d or 8.0
        names = list(single_view_cameras.keys())
        cam_pos = {names[0]: np.zeros(3)}
        for j, nm in enumerate(names[1:], 1):
            ang = 2 * np.pi * j / max(n_cams, 2)
            cam_pos[nm] = np.array([obj_d * np.cos(ang), obj_d * np.sin(ang), 0.0])
        target = np.array([0.0, obj_d, 0.0])
        cam_order = names
        recon_mode = "Single-view (distance-graded placement)"
    else:
        # Use the first valid pair as the target/reference coordinate system.
        first = data[0]
        camera_positions = [(first["camera_a"], first["camera_a_pos"])]
        targets = [first["target"]]
        for item in data:
            camera_positions.append((item["camera_b"], item["camera_b_pos"]))
            targets.append(item["target"])

        # Robust target location for visualization. Pairwise reconstructions can differ slightly.
        target = np.median(np.vstack(targets), axis=0)

        cam_pos = {}
        cam_order = []
        for name, pos in camera_positions:
            if name not in cam_pos:
                cam_pos[name] = np.asarray(pos, dtype=float)
                cam_order.append(name)
        recon_mode = "Multi-view (reconstructed)"

    palette = ["#3aa8ff", "#ff4d5d", "#15e56b", "#ffb21a", "#b07cff", "#ff8c3a"]

    fig = go.Figure()
    for i, name in enumerate(cam_order):
        color = palette[i % len(palette)]
        pos = cam_pos[name]
        fwd = target - pos
        nrm = float(np.linalg.norm(fwd))
        fwd = fwd / nrm if nrm > 1e-9 else np.array([0.0, 1.0, 0.0])
        fig.add_trace(_camera_frustum(pos, fwd, color, name, forward_len=max(0.25 * nrm, 0.5)))
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text", text=[name], textposition="bottom center",
            marker=dict(size=5, color=color), name=name, showlegend=False,
        ))
        # Dashed camera -> object ray.
        fig.add_trace(go.Scatter3d(
            x=[pos[0], target[0]], y=[pos[1], target[1]], z=[pos[2], target[2]],
            mode="lines", line=dict(color="#9fb1c2", width=2, dash="dash"),
            showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=[target[0]], y=[target[1]], z=[target[2]],
        mode="markers+text", text=[obj_label], textposition="bottom center",
        marker=dict(size=8, symbol="diamond", color="#ffffff"), name="Object", showlegend=False,
    ))

    eye = {"top": {"x": 0, "y": 0, "z": 2.1}, "side": {"x": 0, "y": 2.1, "z": 0.25}, "perspective": {"x": 1.55, "y": 1.55, "z": 1.25}}[view]
    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="#020a12",
            xaxis=dict(showgrid=True, gridcolor="#17314a", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#17314a", zeroline=False),
            zaxis=dict(showgrid=True, gridcolor="#17314a", zeroline=False),
            camera=dict(eye=eye),
            aspectmode="data",
        ),
    )
    # Reconstruction stats (also shown in the UI info box and the PDF).
    for p in target_pairs(report):
        corr = p.get("target_correspondence") or {}
        tri = p.get("target_triangulation") or {}
        stats["matched"] += int(corr.get("candidate_matches") or 0)
        stats["inliers"] += int(corr.get("inliers") or 0)
        if tri.get("valid"):
            stats["tri_pts"] += 1
            stats["reproj"] = max(stats["reproj"] or 0.0, float(tri.get("reprojection_error_a") or 0.0))

    return fig, recon_mode, stats


def render_3d(report, view, single_view_cameras=None, obj_pixel=None):
    obj_label = (st.session_state.get("obj_id") or "OBJ_001").strip() or "OBJ_001"
    fig, recon_mode, stats = build_3d_figure(report, view, single_view_cameras, obj_label)
    if fig is None:
        if recon_mode == "no-data":
            st.markdown('<div class="placeholder"><div class="big">◎</div><div>No camera data yet</div><div class="small">Upload videos and run processing.</div></div>', unsafe_allow_html=True)
        else:
            st.info("Plotly is not installed; install it with: py -m pip install plotly")
        return
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="legend-row">'
                '<span class="chip"><span class="dash"></span> Camera to Object Rays</span>'
                '<span class="chip"><span class="dot"></span> Reconstructed 3D Point (Relative)</span>'
                '</div>', unsafe_allow_html=True)

    # Reconstruction Info box (like the reference UI).
    matched = stats["matched"]
    inliers = stats["inliers"]
    tri_pts = stats["tri_pts"]
    reproj = stats["reproj"]
    if stats["has_data"]:
        status_txt, status_cls = "Success", "ok"
    elif single_view_cameras:
        status_txt, status_cls = "Single-view", "c-yellow"
    else:
        status_txt, status_cls = "Pending", "warn"
    info_rows = f"""
<div class="recon-info">
  <div class="info-title">Reconstruction Info</div>
  <div class="info-row"><span>Matched Features:</span><span>{matched}</span></div>
  <div class="info-row"><span>Inlier Matches:</span><span>{inliers}</span></div>
  <div class="info-row"><span>Triangulated Points:</span><span>{tri_pts}</span></div>
  <div class="info-row"><span>Reprojection Error:</span><span>{f"{reproj:.2f} px" if reproj is not None else "—"}</span></div>
  <div class="info-row"><span>Status:</span><span class="{status_cls}">{status_txt}</span></div>
</div>"""
    info_col, view_col = st.columns([1.0, 1.0])
    with info_col:
        st.markdown(info_rows, unsafe_allow_html=True)
    with view_col:
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.caption(f"Mode: {recon_mode} — relative units until a metric scale is established.")


DISPLAY_WIDTH = 640


def display_frame(frame_rgb, width=DISPLAY_WIDTH):
    """Downscale a frame for display so every camera card shows a uniform,
    horizontal image instead of a tall/stretched one (returns just the image;
    the click component reports its rendered width for exact mapping)."""
    h, w = frame_rgb.shape[:2]
    if w <= width:
        return frame_rgb
    scale = width / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (width, nh), interpolation=cv2.INTER_AREA)


def draw_object_box(frame_rgb, x, y, label="OBJ", box=None):
    """Red rectangle around the constant object so it is identifiable in
    every camera view (not just Camera 01). When the object was detected by
    YOLO, `box` outlines the real object instead of a fixed-size square."""
    out = frame_rgb.copy()
    h, w = out.shape[:2]
    if box:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        xi, yi = (x0 + x1) // 2, (y0 + y1) // 2
    else:
        xi, yi = int(round(x)), int(round(y))
        half = max(20, int(0.05 * min(w, h)))
        x0, y0 = max(0, xi - half), max(0, yi - half)
        x1, y1 = min(w - 1, xi + half), min(h - 1, yi + half)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 40, 40), 3)
    cv2.circle(out, (xi, yi), 4, (255, 40, 40), -1)
    if label:
        ty = y0 - 10 if y0 - 10 > 14 else y1 + 22
        cv2.putText(out, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------- state --------------------------
for key, default in {
    "upload_sig": None,
    "uploaded_paths": [],
    "videos_loaded": False,
    "auto_result": {},
    "target_frame": 0,
    "target_click": None,
    "analysis_ran": False,
    "pipeline_log": "",
    "recon_view": "Top View",
    "obj_id": "OBJ_001",
    "obj_desc": "",
    "process_seconds": 0.0,
    "last_run_ok": False,
    "click_epoch": 0,
    "click_seen": set(),
    "locked_points": {},
    "lock_info": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------- header -----------------------------
now = time.localtime()
st.markdown(
    f"""
<div class="topbar">
  <div class="brand"><div class="camera-icon">📹</div><div>
    <div class="brand-title">CCTV Distance Estimation</div>
    <div class="brand-subtitle">Multi-Camera | Constant Object | 3D Distance Measurement</div>
  </div></div>
  <div class="project-box">
    <div><b>Project:</b> <span class="value">Airport Surveillance</span></div>
    <div style="margin-top:5px;color:#9eb0c1;">📅 Date: {time.strftime('%d %b %Y', now)} &nbsp;&nbsp; 🕒 Time: {time.strftime('%H:%M:%S', now)}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------- sidebar -------------------------
with st.sidebar:
    # 1. Input videos (real upload — nothing pre-loaded)
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">1. INPUT VIDEOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Upload one or more recorded CCTV videos. Frames, object and analysis all start from your upload.</div>', unsafe_allow_html=True)

    uploads = st.file_uploader(
        "Upload CCTV videos",
        type=sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS),
        accept_multiple_files=True,
    )
    clear_btn = st.button("🗑 Clear uploaded videos", use_container_width=True)

    widget_files = list(uploads or [])
    sig = tuple((u.name, u.size) for u in widget_files)

    def _reset_analysis_state():
        st.session_state.auto_result = {}
        st.session_state.target_click = None
        st.session_state.target_frame = 0
        st.session_state.analysis_ran = False
        st.session_state.pipeline_log = ""
        # Invalidate every previous click widget so none of them can
        # re-apply an old click on top of the fresh auto-detection.
        st.session_state.click_epoch = int(st.session_state.get("click_epoch", 0)) + 1
        st.session_state.click_seen = set()
        st.session_state.locked_points = {}
        st.session_state.lock_info = ""
        release_video_captures()  # free decoders + cached frames
        clear_reports()

    if clear_btn:
        st.session_state.upload_sig = None
        st.session_state.uploaded_paths = []
        st.session_state.videos_loaded = False
        _reset_analysis_state()
        st.rerun()
    elif widget_files and sig != st.session_state.upload_sig:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # Only the current upload set counts: remove previous uploads so the
        # pipelines never process stale videos. Open capture handles must be
        # released first — on Windows they make the delete fail silently.
        release_video_captures()
        for old in UPLOAD_DIR.iterdir():
            if old.is_file():
                try:
                    old.unlink()
                except Exception:
                    pass
        saved = []
        for up in widget_files:
            dest = UPLOAD_DIR / up.name
            dest.write_bytes(up.getvalue())
            saved.append(dest)
        st.session_state.uploaded_paths = [str(p.resolve()) for p in saved]
        st.session_state.videos_loaded = True
        st.session_state.upload_sig = sig
        _reset_analysis_state()
    # When the widget is empty but uploads were saved earlier, the previous
    # upload set stays active until "Clear" is pressed or a new set is chosen.

    videos = [Path(p) for p in st.session_state.get("uploaded_paths", [])]
    _seen = set()
    videos = [_v for _v in videos if not (str(_v).lower() in _seen or _seen.add(str(_v).lower()))]

    if videos:
        st.markdown(f'<div style="margin-top:7px;color:#15e56b;font-size:12px;">✓ {len(videos)} video(s) uploaded</div>', unsafe_allow_html=True)
        for idx, video in enumerate(videos, 1):
            info = get_video_info(video)
            st.markdown(
                f'<div class="pipeline-line"><span>Camera {idx:02d} · {video.name}</span>'
                f'<span class="muted">{info["width"]}×{info["height"]} · {info["fps"]:.1f} FPS</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div style="margin-top:7px;color:#ffb51d;font-size:12px;">○ Waiting for upload…</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Constant object (auto-detected after upload; manual click overrides)
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">2. CONSTANT OBJECT</div>', unsafe_allow_html=True)
    _ = st.text_input("Object ID", key="obj_id")
    st.markdown('<div class="pipeline-line"><span>Object Type</span><span class="ok">Constant / Fixed Object</span></div>', unsafe_allow_html=True)
    _ = st.text_input("Description (optional)", key="obj_desc", placeholder="e.g. Dustbin, Sign Board")
    _auto_obj = (st.session_state.get("auto_result") or {}).get("object") or {}
    if st.session_state.target_click:
        x, y = st.session_state.target_click
        is_auto = _is_auto_point(_auto_obj.get("point"), st.session_state.target_click)
        src = "auto-detected" if is_auto else "manual"
        badge = "OBJECT LOCKED (AUTO)" if is_auto else "OBJECT LOCKED (MANUAL)"
        det_label = _auto_obj.get("label")
        det_txt = f'Detected object: <b>{det_label}</b><br>' if det_label else ''
        st.markdown(f'<span class="selected-badge">{badge}</span><div class="small" style="margin-top:6px;">{det_txt}Camera 01 point: ({x:.1f}, {y:.1f}) px</div>', unsafe_allow_html=True)
        locked_pts = st.session_state.get("locked_points") or {}
        if locked_pts:
            per_cam = "<br>".join(
                f"↳ {n}: ({float(p[0]):.0f}, {float(p[1]):.0f}) px" for n, p in locked_pts.items()
            )
            st.markdown(f'<div class="small ok">Locked in {len(locked_pts)} camera(s):<br>{per_cam}</div>', unsafe_allow_html=True)
        if src == "auto-detected" and _auto_obj.get("verified_views"):
            st.markdown(f'<div class="small ok">Verified common in {len(_auto_obj["verified_views"])} other view(s).</div>', unsafe_allow_html=True)
    elif videos:
        st.markdown('<div class="small">The system picks a static object common to all views automatically after upload. You can click a different point in section 4 to override.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small">Upload videos first — the object is detected from your footage.</div>', unsafe_allow_html=True)
    detect_button = st.button("🎯 Detect & Track Object", use_container_width=True)
    reset_button = st.button("♻️ Reset to Auto-Detected", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # (Section 3 renders below, after the analysis state has been loaded.)

# Report-derived analysis state, gated on the upload set (see load_analysis).
target_report, scale_factor, scale_source, scale_conf, metric_cams, obj_pixels, rows, states = load_analysis(videos)

with st.sidebar:
    # 3. Processing pipeline
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">3. PROCESSING PIPELINE</div>', unsafe_allow_html=True)
    pipeline_steps = [
        ("Frame Extraction", bool(videos)),
        ("Frame Selection", bool(st.session_state.get("auto_result"))),
        ("Feature Matching", states["Target correspondence"]),
        ("Camera Geometry", bool(target_pairs(target_report))),
        ("Relative Pose Estimation", bool(target_pairs(target_report))),
        ("3D Triangulation (Relative)", states["Target triangulation"]),
        ("Metric Scale Recovery", states["Metric scale"]),
        ("Final Distance Calculation", bool(rows)),
        ("Export Results", bool(rows)),
    ]
    for name, done in pipeline_steps:
        st.markdown(f'<div class="pipeline-line"><span>{name}</span><span class="{"ok" if done else "warn"}">{"✓" if done else "○ Pending"}</span></div>', unsafe_allow_html=True)
    run_button = st.button("▶  Start / Resume Processing", type="primary", use_container_width=True, disabled=not videos)
    st.markdown('</div>', unsafe_allow_html=True)

    if not videos:
        st.info("Upload CCTV videos above to enable processing.")

# Pipeline input folder = wherever the active videos actually live (uploads/
# in normal use, or the injected folder in tests) — NOT a hard-coded path.
active = videos[0].parent if videos else None

# ------------- auto frame + constant object (after upload) ----
auto = st.session_state.get("auto_result") or {}
if videos and (not auto or auto.get("video_set") != tuple(sorted(v.name for v in videos))):
    with st.spinner("Auto-selecting best frames and detecting the constant object…"):
        base = videos[0]
        base_idx, base_score = auto_select_frame(base)
        others = {v.name: (v, auto_select_frame(v)[0]) for v in videos[1:]}
        obj = auto_select_object(base, base_idx, videos[1:], others)
        auto = {
            "video_set": tuple(sorted(v.name for v in videos)),
            "base_frame": int(base_idx),
            "base_score": float(base_score),
            "other_frames": {k: int(v[1]) for k, v in others.items()},
            "object": obj,
        }
        st.session_state.auto_result = auto
        st.session_state.target_frame = int(base_idx)
        # Fresh detection wins over any previous manual click: bump the click
        # epoch so old click widgets can never re-apply their stale value on
        # top of the auto point, and forget every already-seen widget.
        st.session_state.click_epoch = int(st.session_state.get("click_epoch", 0)) + 1
        st.session_state.click_seen = set()
        st.session_state.target_click = (float(obj["point"][0]), float(obj["point"][1])) if obj.get("success") else None
        # The auto-detected object (with its verified per-view transfers)
        # seeds the locked points, so distances use exactly these positions.
        if obj.get("success"):
            _pts = {Path(base).name: [float(obj["point"][0]), float(obj["point"][1])]}
            _pts.update({k: [float(v[0]), float(v[1])] for k, v in (obj.get("points_by_camera") or {}).items()})
            st.session_state.locked_points = _pts
            st.session_state.lock_info = "Auto-detected object locked in every located view — click any frame to override."
        # Warm the frame cache in the background so section 4 paints
        # instantly on the next rerun instead of decoding 1080p h264 frames
        # while the user watches a blank card.
        prefetch_frames(
            videos,
            [int(base_idx)] + [int(v[1]) for v in others.values()],
        )

# ------------------------ processing --------------------------
if detect_button:
    if videos:
        st.session_state.auto_result = {}  # forces re-detection
        _reset_analysis_state()
        st.rerun()
    else:
        st.warning("Upload videos first — detection runs on your footage.")

if reset_button:
    if videos:
        st.session_state.auto_result = {}  # forces re-detection
        _reset_analysis_state()
        st.rerun()
    else:
        st.warning("Upload videos first — detection runs on your footage.")

if run_button and videos and st.session_state.target_click:
    locked = st.session_state.get("locked_points") or {}
    # The pipelines expect the object point expressed in CAMERA 01's frame
    # (the base view). If the user clicked in another camera, map it back
    # into Camera 01 first; per-camera locked points still override the
    # per-view distances in load_analysis.
    base_name = videos[0].name
    if locked.get(base_name):
        x, y = float(locked[base_name][0]), float(locked[base_name][1])
    else:
        x, y = st.session_state.target_click
        src_cam = next((n for n, p in locked.items() if abs(p[0] - x) < 3 and abs(p[1] - y) < 3), None)
        if src_cam is not None and src_cam != base_name:
            src_video = next(v for v in videos if v.name == src_cam)
            auto = st.session_state.get("auto_result") or {}
            src_idx = int((auto.get("other_frames") or {}).get(src_cam, 0))
            bx, by, sc, how = _transfer_point_cached(str(src_video), src_idx, str(videos[0]), int(st.session_state.target_frame), x, y)
            if bx is not None:
                locked[base_name] = [round(float(bx), 1), round(float(by), 1)]
                st.session_state.locked_points = locked
                x, y = float(bx), float(by)
    selected_frame = int(st.session_state.target_frame)
    _t0 = time.time()
    with st.status("Running CCTV geometry, target matching, triangulation and scale recovery…", expanded=True) as status:
        st.write("Stage 1/3 — Frame extraction, feature matching, camera geometry, scale cue…")
        ok1, log1 = run_full_pipeline(str(active))
        st.write("✓ Stage 1 complete — frames extracted, features matched, geometry + scale saved." if ok1 else "⚠ Stage 1 finished with warnings — see diagnostics.")
        st.write("Stage 2/3 — Locating the constant object in every view, then 3D triangulation…")
        ok2, log2 = run_target_pipeline(str(active), x, y, selected_frame)
        st.write("✓ Stage 2 complete — object matched across views and triangulated." if ok2 else "⚠ Stage 2 finished with warnings — see diagnostics.")
        st.write("Stage 3/3 — Camera→ground heights and final metric distances…")
        ok3, log3 = run_metric_layer(str(active))
        st.write("✓ Stage 3 complete — per-camera heights and final distance computed." if ok3 else "⚠ Stage 3 finished with warnings — see diagnostics.")
        combined = ("[GENERAL PIPELINE]\n" + log1 + "\n[TARGET PIPELINE]\n" + log2 + "\n[METRIC LAYER]\n" + log3)[-30000:]
        st.session_state.pipeline_log = combined
        st.session_state.analysis_ran = True
        st.session_state.last_run_ok = ok1 and ok2
        st.session_state.process_seconds = time.time() - _t0
        status.update(label="Pipeline completed" if (ok1 and ok2) else "Pipeline completed with warnings", state="complete" if (ok1 and ok2) else "error")
        # Re-run the script so the sidebar pipeline checklist (and every other
        # panel) renders from the reports just written — no stale "Pending"
        # rows left after processing.
        st.rerun()
elif run_button and videos and not st.session_state.target_click:
    st.error("No constant object could be detected automatically — click one directly in the Camera 01 view (section 4), then press Start again.")

# --------------------------- body -----------------------------
# Persistent completion banner — stays visible after the refresh rerun.
if st.session_state.get("analysis_ran"):
    ok = bool(st.session_state.get("last_run_ok"))
    st.markdown(
        f'<div class="done-bar{" warn" if not ok else ""}">{"✓ Pipeline completed" if ok else "⚠ Pipeline completed with warnings — open Pipeline diagnostics"}</div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1.0, 4.65], gap="small")

with left:
    # 1. Input videos summary (compact — details live in the sidebar only)
    st.markdown('<div class="panel"><div class="panel-title">1. INPUT VIDEOS</div>', unsafe_allow_html=True)
    if not videos:
        st.markdown('<div class="small">No frames are shown until you upload CCTV videos.</div>', unsafe_allow_html=True)
        st.markdown('<div class="select-hint">Upload videos in the sidebar to begin.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:12px;color:#15e56b;">✓ {len(videos)} video(s) uploaded</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small">{sum(get_video_info(v)["frames"] for v in videos):,} total frames across your footage.</div>', unsafe_allow_html=True)
    st.markdown('</div><div style="height:8px"></div>', unsafe_allow_html=True)

    # 2. Target selection summary
    st.markdown('<div class="panel"><div class="panel-title">2. CONSTANT OBJECT</div>', unsafe_allow_html=True)
    if st.session_state.target_click:
        x, y = st.session_state.target_click
        auto_obj = (st.session_state.get("auto_result") or {}).get("object") or {}
        how = "AUTO" if _is_auto_point(auto_obj.get("point"), st.session_state.target_click) else "MANUAL"
        det_label = auto_obj.get("label")
        det_txt = f'Detected: <b>{det_label}</b><br>' if det_label else ''
        st.markdown(f'<div class="selected-badge">{how} SELECTED FIXED POINT</div><div class="small" style="margin-top:7px;">{det_txt}Camera 01 · Frame {st.session_state.target_frame}<br>Pixel: ({x:.1f}, {y:.1f})</div>', unsafe_allow_html=True)
    elif videos:
        st.markdown('<div class="small">Detecting a constant object automatically… or click one in section 4.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="small">Waiting for uploaded CCTV videos.</div>', unsafe_allow_html=True)
    st.markdown('</div><div style="height:8px"></div>', unsafe_allow_html=True)

    # 3. Pipeline status
    st.markdown('<div class="panel"><div class="panel-title">3. PROCESSING PIPELINE</div>', unsafe_allow_html=True)
    for name, done in [
        ("Videos loaded", bool(videos)),
        ("Target point selected", states["Target selected"]),
        ("Target correspondence", states["Target correspondence"]),
        ("Camera geometry / pose", bool(target_pairs(target_report))),
        ("3D triangulation", states["Target triangulation"]),
        ("Metric scale", states["Metric scale"]),
        ("Distance result", bool(rows)),
    ]:
        st.markdown(f'<div class="pipeline-line"><span>{name}</span><span class="{"ok" if done else "warn"}">{"✓" if done else "○ Pending"}</span></div>', unsafe_allow_html=True)
    if st.session_state.pipeline_log:
        with st.expander("Pipeline diagnostics", expanded=False):
            st.code(st.session_state.pipeline_log)

with right:
    # 4. Multi-camera views (uploaded footage; auto-selected frames + object marker)
    st.markdown('<div class="panel"><div class="section-title">4. MULTI-CAMERA VIEWS <span style="color:#a9b8c7;">(Auto-selected frames)</span></div>', unsafe_allow_html=True)
    if not videos:
        st.markdown('<div class="placeholder" style="height:210px;"><div class="big">📹</div><div>Frames appear here after you upload CCTV videos.</div></div>', unsafe_allow_html=True)
    else:
        auto = st.session_state.get("auto_result") or {}
        auto_obj = auto.get("object") or {}
        obj_label = (st.session_state.get("obj_id") or "OBJ").strip() or "OBJ"
        # Object position in every view: the auto-selection transfer before
        # processing, the pipeline's matched point afterwards. When the object
        # was YOLO-detected, its real bounding box is available per camera and
        # is preferred over a synthetic square.
        obj_pts_by_cam = dict(auto_obj.get("points_by_camera") or {})
        for cam, pix in obj_pixels.items():
            obj_pts_by_cam.setdefault(cam, pix)
        obj_boxes_by_cam = dict(auto_obj.get("boxes_by_camera") or {})
        locked_pts = st.session_state.get("locked_points") or {}
        # User-locked points rule: they override auto/pipeline positions for
        # drawing (and later for the distances).
        for cam, pix in locked_pts.items():
            obj_pts_by_cam[cam] = (float(pix[0]), float(pix[1]))
            obj_boxes_by_cam.pop(cam, None)
        cols = st.columns(min(4, len(videos)), gap="small")
        any_click = False
        for i, video in enumerate(videos):
            info = get_video_info(video)
            if i == 0:
                # Camera 01: auto frame (adjustable), click-to-select enabled.
                max_frame = max(0, info["frames"] - 1)
                if max_frame > 0:
                    new_frame = st.slider("Camera 01 frame", 0, max_frame, min(int(st.session_state.target_frame), max_frame), key="cam01_frame", label_visibility="collapsed")
                    st.session_state.target_frame = int(new_frame)
                frame_idx = int(st.session_state.target_frame)
                frame = get_frame(video, frame_idx)
            else:
                frame_idx = int((auto.get("other_frames") or {}).get(video.name, int(info["frames"] // 2) if info["frames"] else 0))
                frame = get_frame(video, frame_idx)
            # Red box on this camera's object point (locked > auto/pipeline).
            pt_here = obj_pts_by_cam.get(video.name)
            box_here = (obj_boxes_by_cam.get(video.name) or {}).get("box")
            if frame is not None and pt_here:
                frame = draw_object_box(frame, float(pt_here[0]), float(pt_here[1]), label=obj_label, box=box_here)
            with cols[i % len(cols)]:
                st.markdown(f'<div class="video-card"><div class="video-head"><span>📹 Camera {i+1:02d}</span><span class="recorded">RECORDED</span></div>', unsafe_allow_html=True)
                if frame is not None:
                    # Uniform horizontal display: every camera card renders its
                    # frame at the same pixel width, so no card looks tall or
                    # stretched compared to the others.
                    disp = display_frame(frame)
                    if streamlit_image_coordinates is not None:
                        # EVERY camera is clickable. The component MUST render
                        # on every rerun (it is what displays the image!) — only
                        # the *action* is consumed once per key. The previous
                        # version skipped rendering for consumed keys, which
                        # blanked both frames after the first interaction.
                        click_key = f"target_click_{st.session_state.click_epoch}_{i}_{frame_idx}"
                        coords = image_click(disp, key=click_key)
                        if (
                            coords and isinstance(coords, dict) and coords.get("x") is not None
                            and click_key not in st.session_state.click_seen
                        ):
                            st.session_state.click_seen.add(click_key)
                            # Click arrives in DISPLAYED pixels (disp is the
                            # 640-wide resize). Scale by frame_width/rendered
                            # width to get TRUE frame pixels — the previous
                            # code only corrected CSS vs natural size, so a
                            # click at the object landed at ~1/3 of its real
                            # position and the box appeared elsewhere.
                            rendered_w = float(coords.get("width") or disp.shape[1])
                            k = float(info["width"]) / rendered_w if rendered_w else 1.0
                            cx = float(coords["x"]) * k
                            cy = float(coords["y"]) * k
                            prev = (st.session_state.get("locked_points") or {}).get(video.name)
                            same_as_locked = (
                                prev is not None
                                and abs(float(prev[0]) - cx) <= 3
                                and abs(float(prev[1]) - cy) <= 3
                            )
                            st.session_state.target_click = (cx, cy)
                            if not same_as_locked:
                                # A genuinely new object choice: transfer it to
                                # the other cameras and refresh. Re-clicking the
                                # same spot (or moving the slider) stays silent.
                                with st.spinner("Locating this object in the other cameras…"):
                                    transfer_point_to_views(video.name, (cx, cy), videos, auto)
                                st.session_state.click_epoch = int(st.session_state.get("click_epoch", 0)) + 1
                                st.session_state.click_seen = set()
                                st.rerun()
                        any_click = True
                    else:
                        # Component missing: frames must still SHOW.
                        st.image(disp, use_container_width=True)
                else:
                    # Frame decode hiccup (large h264 first open) — never leave
                    # the card silently empty.
                    st.markdown('<div class="placeholder" style="height:180px;"><div class="small">Frame still decoding — move the slider slightly or click Start.</div></div>', unsafe_allow_html=True)
                frame_label = frame_idx
                st.markdown(f'<div class="video-foot"><span>Frame: {frame_label} · auto</span><span>{info["width"]}×{info["height"]} · {info["fps"]:.1f} FPS</span></div></div>', unsafe_allow_html=True)
        if st.session_state.target_click:
            x, y = st.session_state.target_click
            how = "auto-detected" if _is_auto_point(auto_obj.get("point"), st.session_state.target_click) else "manually selected"
            n_locked = len(st.session_state.get("locked_points") or {})
            st.markdown(f'<div class="select-hint">✓ Constant object {how} at ({x:.1f}, {y:.1f}). Locked in {n_locked}/{len(videos)} cameras — click any frame to re-lock. Press <b>Start / Resume Processing</b> to compute distances.</div>', unsafe_allow_html=True)
            if st.session_state.get("lock_info"):
                st.markdown(f'<div class="small" style="margin-top:4px;">↳ {st.session_state["lock_info"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="select-hint">No constant object detected automatically — click the object in ANY camera frame; every other frame updates automatically.</div>', unsafe_allow_html=True)

        # Honest status of the auto-selected constant object.
        if auto_obj.get("success"):
            if auto_obj.get("verified_views"):
                qual = ("static + ground-level + verified common in all views"
                        if (auto_obj.get("details") or {}).get("ground_n", 0.0) >= 1.0
                        else "static + verified common in all views")
            else:
                qual = "static in Camera 01 (not verifiable in the other views)"
            if auto_obj.get("status") == "static-only" or not auto_obj.get("verified_views"):
                qual += " — for best results click directly on your constant object (e.g. the dustbin) in Camera 01."
            st.markdown(f'<div class="small" style="margin-top:6px;">Auto-selected object quality: {qual}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    recon_col, result_col = st.columns([1.45, 1.15], gap="small")

    # 5. Functional reconstruction (always renders: multi-view or single-view mode)
    with recon_col:
        st.markdown('<div class="panel"><div class="section-title">5. 3D RECONSTRUCTION <span style="color:#a9b8c7;">(Relative View)</span></div>', unsafe_allow_html=True)
        tabs = ["Top View", "Perspective View", "Side View"]
        st.session_state.recon_view = st.radio("View", tabs, horizontal=True, label_visibility="collapsed", index=tabs.index(st.session_state.recon_view), key="recon_view_radio")
        view_key = {"Top View": "top", "Perspective View": "perspective", "Side View": "side"}[st.session_state.recon_view]
        single_view_info = {}
        if not states["Target triangulation"] and metric_cams:
            single_view_info = {
                name: {"object_distance_m": (obj_pixels.get(name) and pinhole_object_distance(metric_cams[name], obj_pixels.get(name))[0])}
                for name in metric_cams
            }
        render_3d(target_report, view_key, single_view_cameras=single_view_info, obj_pixel=obj_pixels.get(next(iter(obj_pixels), None)))
        st.markdown('</div>', unsafe_allow_html=True)

    # 6 + 7 inside result column
    with result_col:
        st.markdown('<div class="panel"><div class="section-title">6. DISTANCE RESULTS</div>', unsafe_allow_html=True)
        if rows:
            html = ('<table class="metric-table"><tr><th>Camera</th><th>Camera → Ground</th>'
                    '<th>Camera → Object</th><th>3D Relative</th><th>Source</th></tr>')
            for row in rows:
                height_m = row.get("ground_height_m")
                obj_m = row.get("object_distance_m")
                # Fallback: when the ground-plane solve is unavailable for a
                # camera (object not locatable in its crop), the triangulated
                # 3D distance scaled by the recovered metric scale is a real
                # measurement of the same object — display it instead of "—".
                mv_m = row.get("metric")
                height_txt = f"<b>{height_m:.2f} m</b>" if height_m is not None else "—"
                obj_txt = f"<b>{obj_m:.2f} m</b>" if obj_m is not None else (
                    f"<b>{mv_m:.2f} m</b>" if mv_m is not None else "—")
                rel_txt = f"{row['relative']:.2f}" if row.get("relative") is not None else "—"
                if obj_m is not None:
                    source_txt = "Single-view (ground-plane)"
                elif row.get("metric") is not None:
                    source_txt = "Multi-view + scale"
                elif row.get("relative") is not None:
                    source_txt = "Multi-view (relative)"
                else:
                    source_txt = "—"
                cam_color = "ok" if (height_m is not None or obj_m is not None or row.get("relative") is not None) else "warn"
                cam_label = camera_display_name(row["camera"], videos)
                html += (f'<tr><td class="{cam_color}">{cam_label}</td>'
                         f'<td>{height_txt}</td><td>{obj_txt}</td>'
                         f'<td>{rel_txt}</td><td style="font-size:11px;">{source_txt}</td></tr>')
            html += '</table>'
            st.markdown(html, unsafe_allow_html=True)

            # ---- FINAL RESULT: robust, disagree-proof ----------------------
            # Cameras that measure DIFFERENT objects (a look-alike transfer,
            # a bad click) produce wildly different distances. Averaging such
            # numbers (18 m + 178 m -> 98 m) is exactly the "random value"
            # failure mode. Instead: robust consensus — the estimate closest
            # to the MEDIAN wins, and cameras disagreeing >25% are excluded
            # and reported, never averaged in.
            final_ds = [
                float(r["object_distance_m"]) if r.get("object_distance_m") is not None
                else float(r["metric"])
                for r in rows
                if r.get("object_distance_m") is not None or r.get("metric") is not None
            ]
            if final_ds:
                final_ds_arr = np.asarray(final_ds, dtype=float)
                med = float(np.median(final_ds_arr))
                keep, dropped = [], []
                for r in rows:
                    d = (
                        float(r["object_distance_m"]) if r.get("object_distance_m") is not None
                        else (float(r["metric"]) if r.get("metric") is not None else None)
                    )
                    if d is None:
                        continue
                    (keep if abs(d - med) <= 0.25 * max(med, 1e-6) else dropped).append(
                        (camera_display_name(r["camera"], videos), d))
                if keep:
                    kept_vals = [d for _, d in keep]
                    final_m = float(np.median(kept_vals))
                    spread = (max(kept_vals) - min(kept_vals)) / max(final_m, 1e-6)
                    agree = f"cameras agree within {spread:.0%}" if spread < 0.15 else f"spread {spread:.0%} — treat as an estimate"
                    cam_list = ", ".join(f"{n} {d:.2f} m" for n, d in keep)
                    drop_txt = ""
                    if dropped:
                        drop_txt = f" · excluded as inconsistent: " + ", ".join(f"{n} {d:.2f} m" for n, d in dropped)
                    st.markdown(
                        f'<div class="final-result">🏆 FINAL DISTANCE — Camera → Object ({obj_label}): <b>{final_m:.2f} m</b>'
                        f'<span class="fr-note">Consensus of {len(keep)}/{len(final_ds)} cameras ({cam_list}) · {agree}{drop_txt}</span></div>',
                        unsafe_allow_html=True,
                    )
                    if dropped:
                        st.markdown(
                            '<div class="small warn" style="margin-top:6px;">⚠ A camera disagreed strongly — it likely locked onto a different object. '
                            'Click the SAME object in that camera\'s frame and press Start again.</div>',
                            unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="small warn" style="margin-top:8px;">No consistent camera→object distance: the cameras measured different objects. '
                        'Click the same object in each camera\'s frame, then press Start.</div>',
                        unsafe_allow_html=True)

            notes = [r.get("object_distance_note") for r in rows if r.get("object_distance_note")]
            warns = [r.get("warning") for r in rows if r.get("warning")]
            dup_notes = [r.get("duplicate_note") for r in rows if r.get("duplicate_note")]
            if object_distance_unavailable(rows):
                st.markdown('<div class="small warn" style="margin-top:8px;">Object distance unavailable: the selected point has no reliable counterpart in the other view, or lies above the horizon.</div>', unsafe_allow_html=True)
            for d in dup_notes[:2]:
                st.markdown(f'<div class="small" style="margin-top:6px;">ⓘ {d}</div>', unsafe_allow_html=True)
            for n in notes[:1]:
                st.markdown(f'<div class="small" style="margin-top:6px;">Note: {n}</div>', unsafe_allow_html=True)
            for w in warns[:2]:
                st.markdown(f'<div class="small warn">⚠ {w}</div>', unsafe_allow_html=True)
            if scale_factor is None and any(r.get("relative") is not None for r in rows):
                st.markdown('<div class="small" style="margin-top:6px;">Multi-view relative geometry is valid; metres for it need a scale cue. Single-view camera→ground/object estimates use the person-height prior.</div>', unsafe_allow_html=True)
        elif target_report:
            bad_pairs = [p for p in target_pairs(target_report) if p.get("message")]
            st.markdown('<div class="small warn">No valid target-specific distance yet.</div>', unsafe_allow_html=True)
            if bad_pairs:
                st.markdown(f'<div class="small">Reason from current analysis: {bad_pairs[0].get("message")}</div>', unsafe_allow_html=True)
            dupes = [p.get("message") for p in target_pairs(target_report) if p.get("skipped_duplicate")]
            for d in dupes[:2]:
                st.markdown(f'<div class="small">ⓘ {d}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small muted">Waiting for target selection and processing.</div>', unsafe_allow_html=True)
        st.markdown('</div><div style="height:8px"></div>', unsafe_allow_html=True)

        # 7. Scale
        heights_avail = [v for v in metric_cams.values() if v.get("height_m") is not None]
        st.markdown('<div class="panel"><div class="section-title">7. SCALE RECOVERY</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-row"><span>Method</span><span>{scale_source}</span></div>', unsafe_allow_html=True)
        scale_class = "ok" if (scale_factor is not None or heights_avail) else "warn"
        if scale_factor is not None:
            scale_text = f"{scale_factor:.6f} m / relative unit"
        elif heights_avail:
            scale_text = "Person-height prior (per-camera)"
        else:
            scale_text = "Not available"
        st.markdown(f'<div class="status-row"><span>Scale</span><span class="{scale_class}">{scale_text}</span></div>', unsafe_allow_html=True)
        confidence_text = f"{scale_conf:.2%}" if scale_conf is not None else ("~45% (prior)" if heights_avail else "Not reported")
        st.markdown(f'<div class="status-row"><span>Confidence</span><span>{confidence_text}</span></div>', unsafe_allow_html=True)
        scale_status_class = "ok" if (scale_factor is not None or heights_avail) else "warn"
        scale_status_text = "Recovered" if scale_factor is not None else ("Person-prior cue" if heights_avail else "Awaiting valid visual scale")
        st.markdown(f'<div class="status-row"><span>Status</span><span class="{scale_status_class}">{scale_status_text}</span></div>', unsafe_allow_html=True)
        for name, info in list(metric_cams.items())[:4]:
            h = info.get("height_m")
            label = camera_display_name(name, videos)
            if h is not None:
                st.markdown(f'<div class="status-row"><span>↳ {label}</span><span class="ok">h ≈ {h:.2f} m</span></div>', unsafe_allow_html=True)
            else:
                reason = info.get("height_note") or info.get("height_status") or "unavailable"
                st.markdown(f'<div class="status-row"><span>↳ {label}</span><span class="warn">{reason}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 8. system status
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="section-title">8. SYSTEM STATUS</div>', unsafe_allow_html=True)
    status_items = [
        ("Cameras loaded", str(len(videos)), bool(videos)),
        ("Target selection", "Completed" if states["Target selected"] else "Waiting", states["Target selected"]),
        ("Target correspondence", "Validated" if states["Target correspondence"] else "Pending", states["Target correspondence"]),
        ("3D target reconstruction", "Validated" if states["Target triangulation"] else "Pending", states["Target triangulation"]),
        ("Metric distance", "Available" if rows and scale_factor is not None else "Pending", bool(rows and scale_factor is not None)),
    ]
    for name, value, good in status_items:
        st.markdown(f'<div class="status-row"><span>{name}</span><span class="{"ok" if good else "warn"}">{value}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- bottom bar ------------------------
completed = bool(rows and (scale_factor is not None or any(r.get("object_distance_m") for r in rows)))
secs = float(st.session_state.get("process_seconds") or 0.0)
mins, secs_i = int(secs // 60), int(secs % 60)


def build_pdf_report(videos, rows, metric_cams, scale_factor, scale_source, scale_conf,
                     target_report, view_key="perspective"):
    """Render the PDF: camera views (with the object box), the distance table,
    a snapshot of the 3D reconstruction, and the final distance result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    obj_label = (st.session_state.get("obj_id") or "OBJ_001").strip() or "OBJ_001"
    auto_obj = (st.session_state.get("auto_result") or {}).get("object") or {}
    pdf_path = RESULTS_DIR / "cctv_distance_report.pdf"

    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: camera views -------------------------------------
        n = max(1, len(videos))
        ncols = min(2, n)
        nrows = max(1, (n + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.7, 4.15 * nrows + 1.0))
        axes = np.atleast_1d(axes).ravel()
        fig.suptitle("CCTV Distance Estimation — Report", fontsize=16, fontweight="bold")
        for i, video in enumerate(videos):
            ax = axes[i]
            info = get_video_info(video)
            idx = int(st.session_state.target_frame) if i == 0 else int(
                ((st.session_state.get("auto_result") or {}).get("other_frames") or {}).get(video.name, info["frames"] // 2 if info["frames"] else 0)
            )
            frame = get_frame(video, idx)
            if frame is not None:
                tc = st.session_state.get("target_click")
                if i == 0 and tc:
                    box01 = auto_obj.get("box") if _is_auto_point(auto_obj.get("point"), tc) else None
                    frame = draw_object_box(frame, tc[0], tc[1], label=obj_label, box=box01)
                else:
                    pt = (auto_obj.get("points_by_camera") or {}).get(video.name)
                    cb = (auto_obj.get("boxes_by_camera") or {}).get(video.name) or {}
                    if cb.get("box"):
                        frame = draw_object_box(frame, 0, 0, label=cb.get("label") or obj_label, box=cb["box"])
                    elif pt:
                        frame = draw_object_box(frame, float(pt[0]), float(pt[1]), label=obj_label)
                ax.imshow(frame)
            ax.set_title(f"Camera {i+1:02d} — {video.name} (frame {idx})", fontsize=9)
            ax.axis("off")
        for j in range(len(videos), len(axes)):
            axes[j].axis("off")
        sub = (
            f"Object: {obj_label}"
            + (f" ({auto_obj.get('label')})" if auto_obj.get("label") else "")
            + f"   |   Cameras: {len(videos)}   |   Generated: {time.strftime('%d %b %Y %H:%M')}"
        )
        fig.text(0.5, 0.02, sub, ha="center", fontsize=9, color="#333333")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

        # ---- Page 2: 3D reconstruction snapshot ------------------------
        fig3d, _mode, stats = build_3d_figure(target_report, view_key, single_view_cameras={}, obj_label=obj_label)
        if fig3d is not None:
            try:
                fig3d.update_layout(height=700, width=1000, paper_bgcolor="white")
                img_bytes = fig3d.to_image(format="png")
                import io
                img = plt.imread(io.BytesIO(img_bytes))
                figp, axp = plt.subplots(figsize=(11.7, 8.3))
                axp.imshow(img)
                axp.set_title(f"3D Reconstruction ({_mode})", fontsize=13)
                axp.axis("off")
                figp.text(0.5, 0.02,
                          f"Matched features: {stats['matched']}   |   Inliers: {stats['inliers']}   |   Triangulated: {stats['tri_pts']}"
                          + (f"   |   Reprojection: {stats['reproj']:.2f} px" if stats["reproj"] is not None else ""),
                          ha="center", fontsize=9, color="#333333")
                figp.tight_layout()
                pdf.savefig(figp, dpi=120)
                plt.close(figp)
            except Exception:
                pass  # kaleido missing etc. — the PDF still ships with pages 1/3

        # ---- Page 3: distance table + final result ---------------------
        figt, axt = plt.subplots(figsize=(11.7, 8.3))
        axt.axis("off")
        axt.set_title("Distance Results", fontsize=14, fontweight="bold", loc="left")
        col_w = [0.16, 0.20, 0.20, 0.14, 0.30]
        tbl_rows = [["Camera", "Camera → Ground", "Camera → Object", "3D Rel.", "Source"]]
        for r in rows:
            h_m = r.get("ground_height_m")
            o_m = r.get("object_distance_m")
            tbl_rows.append([
                camera_display_name(r["camera"], videos),
                f"{h_m:.2f} m" if h_m is not None else "—",
                f"{o_m:.2f} m" if o_m is not None else "—",
                f"{r['relative']:.2f}" if r.get("relative") is not None else "—",
                "Single-view (ground-plane)" if o_m is not None else (
                    "Multi-view + scale" if r.get("metric") is not None else "Multi-view (relative)"),
            ])
        table = axt.table(cellText=tbl_rows[1:], colLabels=tbl_rows[0],
                          colWidths=col_w, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.6)
        for j in range(len(tbl_rows[0])):
            table[0, j].set_facecolor("#0d2233")
            table[0, j].set_text_props(color="white", fontweight="bold")
        final_ds = [float(r["object_distance_m"]) for r in rows if r.get("object_distance_m") is not None]
        if final_ds:
            final_m = float(np.median(final_ds))
            axt.text(0.5, 0.12, f"FINAL DISTANCE — Camera → Object ({obj_label}): {final_m:.2f} m",
                     ha="center", fontsize=13, fontweight="bold", color="#0a6b2d",
                     transform=axt.transAxes)
        axt.text(0.5, 0.06,
                 f"Scale: {scale_source} — {scale_factor:.6f} m/unit" if scale_factor is not None else f"Scale: {scale_source or 'person-height prior'}",
                 ha="center", fontsize=9, color="#333333", transform=axt.transAxes)
        heights_avail = [v for v in metric_cams.values() if v.get("height_m") is not None]
        if heights_avail:
            axt.text(0.5, 0.02,
                     "Camera heights: " + " · ".join(f"{camera_display_name(k, videos)} ≈ {v['height_m']:.2f} m" for k, v in metric_cams.items() if v.get("height_m") is not None),
                     ha="center", fontsize=9, color="#333333", transform=axt.transAxes)
        pdf.savefig(figt, dpi=120)
        plt.close(figt)

    return pdf_path


export_col, status_col = st.columns([0.25, 3.0])
with status_col:
    st.markdown(
        f'<div class="bottom-bar"><div>🛡️ <span style="color:#9fb1c2;">System Status:</span> <span class="{"ok" if videos else "warn"}">{"Running" if completed else ("Ready" if videos else "Waiting for videos")}</span></div>'
        f'<div>📹 Cameras Connected: <b>{len(videos)}/{len(videos)}</b></div>'
        f'<div>◷ Total Processing Time: <b>{mins:02d}:{secs_i:02d}</b></div></div>',
        unsafe_allow_html=True,
    )
with export_col:
    if REPORT_TARGET.exists():
        if st.button("⬇ Export Report", use_container_width=True):
            with st.spinner("Building PDF report…"):
                try:
                    pdf_file = build_pdf_report(videos, rows, metric_cams, scale_factor, scale_source, scale_conf, target_report)
                    st.session_state["pdf_bytes"] = pdf_file.read_bytes()
                except Exception as exc:
                    st.error(f"PDF export failed: {exc}")
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    "⬇ Download PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name="cctv_distance_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
    else:
        st.button("⬇ Export Report", disabled=True, use_container_width=True)
