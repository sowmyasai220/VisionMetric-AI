
<style>
.stApp { background:#07111d; }
.block-container { max-width:1500px; padding-top:1rem; }
.panel, .section {
    background:linear-gradient(180deg,#0e1b2b,#0a1523);
    border:1px solid #243b53; border-radius:10px;
    box-shadow:0 8px 28px rgba(0,0,0,.18);
}
.section-title { letter-spacing:.08em; }
.small { color:#8fa4b8; }
div[data-testid="stButton"] > button {
    border:1px solid #2b4661; border-radius:7px;
    background:#102239; color:#e8f2fb; font-weight:600;
}
div[data-testid="stButton"] > button:hover {
    border-color:#39bdf8; color:#fff;
}
div[data-testid="stTextInput"] input {
    background:#091625; border-color:#29445f; color:#edf6ff;
}
</style>

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
PIPELINE = BASE_DIR / "src" / "end_to_end_pipeline.py"
MULTI_REPORT = RESULTS_DIR / "multi_camera_report.json"
GEOMETRY_REPORT = RESULTS_DIR / "multi_camera_geometry.json"
SCALE_REPORT = RESULTS_DIR / "visual_scale_result.json"
METRIC_REPORT = RESULTS_DIR / "metric_integration.json"
FINAL_REPORT = RESULTS_DIR / "final_pipeline_report.json"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

DEFAULT_LOCAL_FOLDER = str(VIDEO_DIR)

def discover_local_videos(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: str(p).lower(),
    )

def choose_local_folder():
    """Open a native Windows folder picker when the app is running locally."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select CCTV Video Folder")
        root.destroy()
        return selected or ""
    except Exception:
        return ""

st.set_page_config(
    page_title="CCTV Distance Estimation",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root{--bg:#020b15;--panel:#071827;--panel2:#06131f;--line:#17334a;--blue:#4b9aff;--green:#12ee6a;--yellow:#ffb31b;--red:#ff4e59;--text:#e8f0f8;--muted:#91a5b8}
html,body,[class*="css"]{font-family:Inter,Segoe UI,Arial,sans-serif}
.stApp{background:radial-gradient(circle at 50% -10%,#0a1b2c 0%,var(--bg) 42%);color:var(--text)}
.block-container{max-width:1536px;padding:14px 18px 12px}
header[data-testid="stHeader"]{background:transparent}
footer{display:none}
section[data-testid="stSidebar"]{display:none!important}
[data-testid="column"]{min-width:0!important}
.hero{height:72px;border:1px solid var(--line);border-radius:8px;background:linear-gradient(90deg,#071827,#061421);display:flex;align-items:center;justify-content:space-between;padding:10px 16px;margin-bottom:10px}
.hero-left{display:flex;align-items:center;gap:13px}.hero-icon{font-size:36px}.hero-title{font-size:25px;font-weight:800;line-height:1.05}.hero-sub{color:#b5c3d1;font-size:14px;margin-top:5px}
.project-box{min-width:305px;border:1px solid #1b3850;border-radius:8px;padding:8px 12px;background:#061521;font-size:12px;line-height:1.7}.project-box b{color:#e7eef7}.project-name{color:#2e8cff;font-weight:700}
.panel{border:1px solid var(--line);border-radius:7px;background:linear-gradient(180deg,#071827,#04111c);padding:12px;box-shadow:0 8px 22px rgba(0,0,0,.14);margin-bottom:8px}
.section-title{color:#66a7ff;font-weight:800;font-size:14px;letter-spacing:.2px;margin-bottom:9px}.section-title .white{color:#dce7f2}.muted{color:var(--muted)}.green{color:var(--green)}.yellow{color:var(--yellow)}
/* uploader: only the real uploader, no fake second input */
div[data-testid="stFileUploader"]{margin:0!important}div[data-testid="stFileUploaderDropzone"]{min-height:112px!important;padding:0!important;border:1px dashed #42647f!important;border-radius:6px!important;background:#07121e!important}div[data-testid="stFileUploaderDropzone"]>div{padding:10px!important}div[data-testid="stFileUploaderDropzoneInstructions"]{color:#9fb1c3!important}div[data-testid="stFileUploaderDropzoneInstructions"] svg{display:none!important}div[data-testid="stFileUploaderDropzoneInstructions"] span{font-size:0!important}div[data-testid="stFileUploaderDropzoneInstructions"] span::after{content:"Drag & Drop videos here\A or";white-space:pre;font-size:12px;line-height:1.9}div[data-testid="stFileUploaderDropzone"] button{background:#071827!important;color:#8ebeff!important;border:1px solid #2e7ee1!important;border-radius:5px!important;font-weight:700!important}
.stTextInput input{background:#0a111a!important;color:#eaf1f8!important;border:1px solid #0e1823!important;border-radius:6px!important}.stTextInput label{color:#91a5b8!important;font-size:11px!important}
.stButton>button{background:#1269d8!important;color:white!important;border:1px solid #388cff!important;border-radius:5px!important;font-weight:700!important;min-height:34px}.stButton>button:hover{background:#1c79eb!important}
.camera-card{border:1px solid #1a3951;border-radius:6px;overflow:hidden;background:#030c14}.camera-head{height:32px;background:#091622;display:flex;align-items:center;justify-content:space-between;padding:0 9px;font-size:12px;font-weight:700}.ready{border:1px solid var(--green);color:var(--green);border-radius:4px;padding:2px 6px;font-size:10px}.camera-foot{display:flex;justify-content:space-between;padding:6px 8px;color:#9dafc0;font-size:11px}
.empty-camera{height:180px;display:flex;align-items:center;justify-content:center;background:#020b14;color:#536a7d;font-size:12px}.empty-camera span{border:1px dashed #28465d;padding:34px 18px;border-radius:5px}
.pipe-row{display:flex;align-items:center;justify-content:space-between;padding:4px 0;font-size:11px}.pipe-ok{color:var(--green)}.pipe-wait{color:var(--yellow)}.pipe-muted{color:#71879a}
.result-table{width:100%;border-collapse:collapse;font-size:12px}.result-table th{padding:9px 7px;background:#0a1724;border:1px solid #1b3449;color:#eaf2fa;text-align:left}.result-table td{padding:9px 7px;border:1px solid #142b3d}
.status-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #11283a;font-size:11px}.status-row:last-child{border-bottom:0}.status-ok{color:var(--green);font-weight:700}.status-wait{color:var(--yellow);font-weight:700}
.recon-wrap{border:1px solid #1a344a;border-radius:6px;background:#020a12;overflow:hidden}.recon-svg{width:100%;height:302px;display:block;background:#020a12}.view-tabs{text-align:center;margin-top:7px}.view-tab{display:inline-block;padding:7px 14px;margin:0 2px;border-radius:4px;color:#9eb0c2;font-size:11px}.view-tab.active{background:#1559b5;color:#fff}
.bottom{border:1px solid var(--line);border-radius:7px;background:#061421;padding:9px 13px;display:flex;justify-content:space-between;align-items:center;font-size:12px;margin-top:8px}
</style>
""", unsafe_allow_html=True)


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def clear_video_workspace():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for p in VIDEO_DIR.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass


def save_uploads(uploaded):
    clear_video_workspace()
    paths = []
    for f in uploaded:
        target = VIDEO_DIR / Path(f.name).name
        with open(target, "wb") as out:
            out.write(f.getbuffer())
        paths.append(target)
    return paths


def preview(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    idx = max(0, count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, idx, fps
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx, fps


def relative_rows():
    rows = {}
    multi = load_json(MULTI_REPORT)
    for pair in multi.get("cross_camera_pairs", []):
        tri = pair.get("triangulation", {})
        a, b = pair.get("camera_a"), pair.get("camera_b")
        if a and tri.get("camera_a_distance_relative") is not None:
            rows[a] = float(tri["camera_a_distance_relative"])
        if b and tri.get("camera_b_distance_relative") is not None:
            rows[b] = float(tri["camera_b_distance_relative"])
    return rows


def metric_rows():
    report = load_json(METRIC_REPORT)
    rows = []
    data = report.get("metric_distances", [])
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                rows.append((x.get("camera") or x.get("video") or "Camera", x.get("relative_distance"), x.get("metric_distance_m")))
    elif isinstance(data, dict):
        for k, v in data.items():
            rows.append((k, None, v))
    if not rows:
        scale_report = load_json(SCALE_REPORT)
        scale = scale_report.get("scale_m_per_relative_unit") or scale_report.get("scale")
        if scale:
            rows = [(k, v, v * float(scale)) for k, v in relative_rows().items()]
    return rows


def run_pipeline(video_dir: Path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        [sys.executable, str(PIPELINE), str(video_dir)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode == 0, (p.stdout or "") + "\n" + (p.stderr or "")

def clear_old_results():
    """Prevent a previous run from appearing for a newly uploaded dataset."""
    for p in [
        GEOMETRY_REPORT, MULTI_REPORT, SCALE_REPORT,
        METRIC_REPORT, FINAL_REPORT
    ]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def read_frame(path: Path, frame_index=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0, 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_index is None:
        frame_index = max(0, count // 2)
    frame_index = max(0, min(frame_index, max(0, count - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, frame_index, fps
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_index, fps


def draw_marker(image, point, color=(255, 60, 70), label="OBJ_001"):
    out = image.copy()
    if point is None:
        return out
    x, y = int(point[0]), int(point[1])
    cv2.circle(out, (x, y), 12, color, 3)
    cv2.line(out, (x - 20, y), (x + 20, y), color, 2)
    cv2.line(out, (x, y - 20), (x, y + 20), color, 2)
    cv2.putText(
        out, label, (x + 14, max(22, y - 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
    )
    return out


def camera_card(path: Path, number: int, processed: bool, selectable=False):
    frame, idx, fps = read_frame(path)
    st.markdown('<div class="camera-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="camera-head"><span>📹 Camera {number:02d}</span>'
        f'<span class="ready">{"LIVE" if processed else "READY"}</span></div>',
        unsafe_allow_html=True,
    )

    selected = None
    if frame is not None:
        if selectable and not processed:
            st.caption("Click the fixed object/point in Camera 01.")
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                click = streamlit_image_coordinates(
                    frame,
                    key="constant_object_selector",
                    use_column_width=True,
                )
                if click:
                    selected = (float(click["x"]), float(click["y"]))
            except Exception as exc:
                st.warning(f"Object selector unavailable: {exc}")
                st.image(frame, use_container_width=True)
        else:
            selected = st.session_state.get("selected_point")
            display = draw_marker(frame, selected, label=st.session_state.get("object_id", "OBJ_001"))
            st.image(display, use_container_width=True)
    else:
        st.markdown(
            '<div class="empty-camera"><span>Camera preview unavailable</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="camera-foot"><span>Frame: {idx}</span>'
        f'<span>FPS: {fps:.1f}</span></div></div>',
        unsafe_allow_html=True,
    )
    return selected


def metric_rows():
    report = load_json(METRIC_REPORT)
    rows = []
    data = report.get("metric_distances", [])
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                rows.append(
                    (
                        x.get("camera") or x.get("video") or "Camera",
                        x.get("relative_distance"),
                        x.get("metric_distance_m"),
                    )
                )
    elif isinstance(data, dict):
        for k, v in data.items():
            rows.append((k, None, v))

    if not rows:
        scale_report = load_json(SCALE_REPORT)
        scale = scale_report.get("scale_m_per_relative_unit") or scale_report.get("scale")
        if scale:
            rows = [
                (k, v, v * float(scale))
                for k, v in relative_rows().items()
            ]
    return rows


def relative_rows():
    rows = {}
    multi = load_json(MULTI_REPORT)
    for pair in multi.get("cross_camera_pairs", []):
        tri = pair.get("triangulation", {})
        a, b = pair.get("camera_a"), pair.get("camera_b")
        if a and tri.get("camera_a_distance_relative") is not None:
            rows[a] = float(tri["camera_a_distance_relative"])
        if b and tri.get("camera_b_distance_relative") is not None:
            rows[b] = float(tri["camera_b_distance_relative"])
    return rows


def reconstruction_svg(camera_count, object_id, processed, selected=False):
    n = max(2, min(camera_count, 4))
    positions = [(80,235,"#18e86b"), (650,235,"#ffbd27"),
                 (260,45,"#398cff"), (510,55,"#ff4e59")]
    cameras = []
    rays = []
    for i in range(n):
        x, y, c = positions[i]
        cameras.append(
            f'<g transform="translate({x} {y})">'
            f'<rect x="0" y="0" width="64" height="38" rx="3" fill="#071827" stroke="{c}"/>'
            f'<circle cx="64" cy="19" r="7" fill="none" stroke="{c}"/>'
            f'<line x1="64" y1="19" x2="78" y2="12" stroke="{c}"/>'
            f'</g><text x="{x}" y="{y+55}" fill="{c}" font-size="11" '
            f'font-weight="700">Camera {i+1:02d}</text>'
        )
        rays.append(
            f'<line x1="{x+32}" y1="{y+18}" x2="365" y2="154" '
            f'stroke="#9fb5c6" stroke-width="1" stroke-dasharray="6 5" opacity=".75"/>'
        )

    multi = load_json(MULTI_REPORT) if processed else {}
    successful = int(multi.get("successful_3d_pairs", 0) or 0)
    points = 0
    for pair in multi.get("cross_camera_pairs", []):
        valid = [
            p for p in pair.get("triangulation", {}).get("points", [])
            if p.get("valid")
        ]
        points = max(points, len(valid))

    status = "Success" if successful else ("Object selected" if selected else "Waiting")
    point_label = "(Reconstructed Point)" if processed else (
        "(Selected Target)" if selected else "(Waiting for Target)"
    )

    info = (
        '<g transform="translate(485 15)">'
        '<rect width="165" height="105" rx="6" fill="#030d16" stroke="#1d3a52"/>'
        '<text x="10" y="19" fill="#53a3ff" font-size="11" font-weight="700">Reconstruction Info</text>'
        f'<text x="10" y="39" fill="#a8b8c7" font-size="9">Triangulated Points</text>'
        f'<text x="150" y="39" fill="#eef4fa" font-size="9" text-anchor="end">'
        f'{"—" if not processed else (points or "—")}</text>'
        f'<text x="10" y="57" fill="#a8b8c7" font-size="9">Target</text>'
        f'<text x="150" y="57" fill="#eef4fa" font-size="9" text-anchor="end">'
        f'{"Selected" if selected else "Not selected"}</text>'
        f'<text x="10" y="75" fill="#a8b8c7" font-size="9">Status</text>'
        f'<text x="150" y="75" fill="#12ee6a" font-size="9" text-anchor="end">{status}</text>'
        '</g>'
    )

    return (
        '<div class="recon-wrap"><svg class="recon-svg" viewBox="0 0 680 302" '
        'preserveAspectRatio="none">'
        '<defs><pattern id="grid" width="34" height="34" patternUnits="userSpaceOnUse">'
        '<path d="M34 0H0V34" fill="none" stroke="#2d5674" stroke-opacity=".18"/>'
        '</pattern></defs>'
        '<rect width="680" height="302" fill="url(#grid)"/>'
        + ''.join(rays)
        + '<circle cx="365" cy="154" r="9" fill="#dbe6ef"/>'
        '<circle cx="365" cy="154" r="18" fill="none" stroke="#dbe6ef" stroke-opacity=".15"/>'
        f'<text x="365" y="183" fill="#eaf1f7" font-size="11" font-weight="700" text-anchor="middle">'
        f'{object_id or "OBJ_001"}</text>'
        f'<text x="365" y="198" fill="#91a5b8" font-size="9" text-anchor="middle">{point_label}</text>'
        + ''.join(cameras)
        + info
        + '<g transform="translate(10 255)">'
        '<rect width="250" height="34" rx="5" fill="#020a12" stroke="#24425a"/>'
        '<line x1="12" y1="12" x2="40" y2="12" stroke="#b6c4d1" stroke-dasharray="5 4"/>'
        '<text x="48" y="15" fill="#b6c4d1" font-size="9">Camera to Object Rays</text>'
        '<circle cx="178" cy="12" r="5" fill="#dbe6ef"/>'
        '<text x="190" y="15" fill="#b6c4d1" font-size="9">Reconstructed Point</text>'
        '</g></svg></div>'
    )


# ------------------------- session state -----------------------
defaults = {
    "processed": False,
    "log": "",
    "selected_point": None,
    "object_selected": False,
    "processing_seconds": None,
    "local_folder": DEFAULT_LOCAL_FOLDER,
    "loaded_signature": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

now = time.localtime()
st.markdown(
    f'<div class="hero"><div class="hero-left"><div class="hero-icon">📹</div>'
    f'<div><div class="hero-title">CCTV Distance Estimation</div>'
    f'<div class="hero-sub">Multi-Camera | Constant Object | 3D Distance Measurement | Local CCTV</div></div></div>'
    f'<div class="project-box"><b>Project:</b> <span class="project-name">CCTV Surveillance</span><br>'
    f'<span class="muted">📅 Date: {time.strftime("%d %b %Y", now)} &nbsp;&nbsp; '
    f'🕒 Time: {time.strftime("%H:%M:%S", now)}</span></div></div>',
    unsafe_allow_html=True,
)

left, main = st.columns([0.92, 3.35], gap="small")

# ------------------------- left controls ----------------------
with left:
    st.markdown(
        '<div class="panel"><div class="section-title">1. INPUT VIDEOS</div>'
        '<div style="font-size:12px;margin-bottom:7px">Load Local CCTV Videos</div>',
        unsafe_allow_html=True,
    )

    st.caption("Files are read directly from this computer — no browser upload and no 200 MB limit.")

    folder_input = st.text_input(
        "Local CCTV folder",
        value=st.session_state.local_folder,
        key="local_folder_input",
        label_visibility="collapsed",
        placeholder=r"C:\path\to\CCTV\videos",
    )

    browse_col, load_col = st.columns([1, 1], gap="small")
    with browse_col:
        if st.button("📁 Browse Folder", use_container_width=True, key="browse_folder"):
            chosen = choose_local_folder()
            if chosen:
                st.session_state.local_folder = chosen
                st.session_state.local_folder_input = chosen
                st.session_state.processed = False
                st.session_state.selected_point = None
                st.session_state.object_selected = False
                st.session_state.log = ""
                clear_old_results()
                st.rerun()

    with load_col:
        load_local = st.button("Load Videos", use_container_width=True, key="load_local")

    if load_local:
        candidate = Path(folder_input.strip().strip('"'))
        if candidate.exists() and candidate.is_dir():
            files_found = discover_local_videos(candidate)
            if len(files_found) < 2:
                st.error("Select a folder containing at least 2 CCTV videos.")
            else:
                st.session_state.local_folder = str(candidate.resolve())
                st.session_state.loaded_signature = "|".join(
                    f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files_found
                )
                st.session_state.processed = False
                st.session_state.selected_point = None
                st.session_state.object_selected = False
                st.session_state.log = ""
                st.session_state.processing_seconds = None
                clear_old_results()
                st.rerun()
        else:
            st.error("That local folder does not exist.")

    active_folder = Path(st.session_state.local_folder)
    videos = discover_local_videos(active_folder)

    if videos:
        st.markdown(
            f'<div class="green" style="font-size:12px;margin-top:7px">'
            f'✓ {len(videos)} local video(s) loaded</div>'
            f'<div class="muted" style="font-size:10px;word-break:break-all;margin-top:4px">'
            f'{active_folder}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#ffb31b;font-size:11px;margin-top:7px">'
            'No local CCTV videos loaded.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="panel"><div class="section-title">2. CONSTANT OBJECT</div>',
        unsafe_allow_html=True,
    )
    object_id = st.text_input(
        "Object ID", value="OBJ_001", label_visibility="collapsed", key="object_id"
    )
    description = st.text_input(
        "Description",
        value="Fixed / common physical object",
        label_visibility="collapsed",
        key="description",
    )

    if st.session_state.object_selected:
        st.markdown(
            '<div class="green" style="font-size:12px">● Constant / Fixed Object — SELECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#ffb31b;font-size:12px">● Select a fixed object in Camera 01</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Processing button is intentionally disabled until a target is selected.
    run = st.button(
        "▶  Start / Resume Processing",
        use_container_width=True,
        key="run_pipeline",
        disabled=(len(videos) < 2 or not st.session_state.object_selected),
    )

    if len(videos) < 2:
        st.caption("Upload at least 2 overlapping CCTV videos.")
    elif not st.session_state.object_selected:
        st.caption("First click the constant/fixed object in Camera 01.")

# ------------------------- main content -----------------------
with main:
    st.markdown(
        '<div class="panel"><div class="section-title">4. MULTI-CAMERA VIEWS '
        '<span class="white">(Select Constant Object)</span></div>',
        unsafe_allow_html=True,
    )

    if videos:
        cols = st.columns(min(4, len(videos)), gap="small")
        for i, path in enumerate(videos):
            with cols[i % len(cols)]:
                clicked = camera_card(
                    path,
                    i + 1,
                    st.session_state.processed,
                    selectable=(i == 0 and not st.session_state.processed),
                )
                if i == 0 and clicked is not None:
                    st.session_state.selected_point = clicked
                    st.session_state.object_selected = True
                    st.rerun()
    else:
        st.markdown(
            '<div style="height:205px;display:flex;align-items:center;justify-content:center;'
            'color:#91a5b8;border:1px dashed #28465d;border-radius:5px">'
            'Upload CCTV videos to begin.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.object_selected and not st.session_state.processed:
        st.markdown(
            '<div style="margin-top:7px;color:#12ee6a;font-size:12px">'
            '✓ Constant object selected in Camera 01. '
            'Now start processing to locate the corresponding point across the other views.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Processing happens only after an explicit object selection.
    if run:
        start = time.perf_counter()
        with st.spinner("Running CCTV distance analysis..."):
            ok, log = run_pipeline(active_folder)
        st.session_state.processing_seconds = time.perf_counter() - start
        st.session_state.processed = ok
        st.session_state.log = log
        if ok:
            st.rerun()
        else:
            st.error("Pipeline failed. Check diagnostics below.")

    multi = load_json(MULTI_REPORT) if st.session_state.processed else {}
    scale_report = load_json(SCALE_REPORT) if st.session_state.processed else {}
    scale = scale_report.get("scale_m_per_relative_unit") or scale_report.get("scale")
    rows = metric_rows() if st.session_state.processed else []

    recon_col, result_col = st.columns([1.5, 1.08], gap="small")

    with recon_col:
        st.markdown(
            f'<div class="panel"><div class="section-title">5. 3D RECONSTRUCTION '
            f'<span class="white">(Relative View)</span></div>'
            f'{reconstruction_svg(len(videos), object_id, st.session_state.processed, st.session_state.object_selected)}'
            f'<div class="view-tabs"><span class="view-tab active">Top View</span>'
            f'<span class="view-tab">Perspective View</span><span class="view-tab">Side View</span></div></div>',
            unsafe_allow_html=True,
        )

    with result_col:
        table = '<table class="result-table"><tr><th>Camera</th><th>Relative</th><th>3D Distance (m)</th></tr>'
        if rows:
            colors = ["#12ee6a", "#ff4e59", "#398cff", "#ffc338"]
            for i, (cam, rel_d, met) in enumerate(rows):
                table += (
                    f'<tr><td style="color:{colors[i%4]};font-weight:700">Camera {i+1:02d}</td>'
                    f'<td>{f"{rel_d:.2f}" if isinstance(rel_d,(int,float)) else "—"}</td>'
                    f'<td><b>{f"{met:.3f}" if isinstance(met,(int,float)) else "—"}</b></td></tr>'
                )
        else:
            table += '<tr><td>—</td><td>—</td><td>—</td></tr>'
        table += '</table>'

        note = (
            '<div style="color:#12ee6a;font-size:10px;margin-top:7px">'
            'Metric distances calculated by the analysis pipeline.</div>'
            if rows else
            '<div style="color:#ffb31b;font-size:10px;margin-top:7px">'
            'Select the constant object and start processing to calculate distance.</div>'
        )

        st.markdown(
            f'<div class="panel"><div class="section-title">6. DISTANCE RESULTS</div>'
            f'{table}{note}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="panel"><div class="section-title">7. SCALE RECOVERY</div>'
            f'<div class="status-row"><span>Method</span><span class="yellow">Visual Reference</span></div>'
            f'<div class="status-row"><span>Reference Used</span><span>{"Visual cue" if scale else "—"}</span></div>'
            f'<div class="status-row"><span>Estimated Scale</span><span>'
            f'{"%.6f m / relative unit" % float(scale) if scale else "—"}</span></div>'
            f'<div class="status-row"><span>Status</span><span class="'
            f'{"status-ok" if scale else "status-wait"}">'
            f'{"Scale recovered" if scale else "Waiting for processing..."}</span></div></div>',
            unsafe_allow_html=True,
        )

        recon_done = bool(multi.get("successful_3d_pairs")) if st.session_state.processed else False
        st.markdown(
            f'<div class="panel"><div class="section-title">8. SYSTEM STATUS</div>'
            f'<div class="status-row"><span>✓ Cameras detected</span><span class="status-ok">{len(videos)}</span></div>'
            f'<div class="status-row"><span>✓ Constant object</span><span class="status-ok">'
            f'{"Selected" if st.session_state.object_selected else "Waiting"}</span></div>'
            f'<div class="status-row"><span>✓ Feature matching</span><span class="status-ok">'
            f'{"Completed" if multi else "Waiting"}</span></div>'
            f'<div class="status-row"><span>✓ 3D reconstruction</span><span class="status-ok">'
            f'{"Completed" if recon_done else "Waiting"}</span></div>'
            f'<div class="status-row"><span>◌ Metric distance</span><span class="'
            f'{"status-ok" if rows else "status-wait"}">'
            f'{"Available" if rows else "Waiting"}</span></div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.log:
        with st.expander("Pipeline diagnostics", expanded=False):
            st.code(st.session_state.log[-12000:])

processing = (
    "Complete" if st.session_state.processed
    else ("Ready" if st.session_state.object_selected else "Waiting")
)
elapsed = (
    f'{st.session_state.processing_seconds:.1f}s'
    if st.session_state.processing_seconds is not None else "—"
)
st.markdown(
    f'<div class="bottom"><div>🛡️ <span class="muted">System Status:</span> '
    f'<span class="green">{processing}</span></div>'
    f'<div>📹 Cameras Connected: <b>{len(videos)} / {len(videos)}</b></div>'
    f'<div>◷ Total Processing Time: <b>{elapsed}</b></div></div>',
    unsafe_allow_html=True,
)

if FINAL_REPORT.exists() and st.session_state.processed:
    with open(FINAL_REPORT, "rb") as f:
        st.download_button(
            "⬇ Export Report",
            data=f.read(),
            file_name="cctv_distance_final_report.json",
            mime="application/json",
        )
