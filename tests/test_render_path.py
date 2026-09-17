"""Render-path test: execute the real main.py via Streamlit AppTest.

Verifies the upload-first flow: nothing is shown before upload; after a
simulated upload the auto-selection populates sections 4/6/7.
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

PATH = Path(__file__).resolve().parent.parent
MAIN = PATH / "main.py"

# ------------------------------------------------------------------
# Phase 1: no upload -> nothing pre-loaded
# ------------------------------------------------------------------
at = AppTest.from_file(str(MAIN), default_timeout=300)
at.run()

assert not at.exception, f"Streamlit raised (phase 1): {at.exception}"
joined1 = "\n".join(md.value for md in at.markdown)

checks1 = {
    "Section 4 header present": "4. MULTI-CAMERA VIEWS" in joined1,
    "No frames before upload": "Frames appear here after you upload" in joined1,
    "Sidebar asks for upload": len(at.file_uploader) > 0,
}
failed = [n for n, ok in checks1.items() if not ok]

print("PHASE 1 (no upload):")
for n, ok in checks1.items():
    print(f"  [{'OK ' if ok else 'MISS'}] {n}")

# ------------------------------------------------------------------
# Phase 2: simulate an upload of the bundled demo videos
# ------------------------------------------------------------------
video_paths = sorted((PATH / "input_videos").glob("*.mp4"))
assert video_paths, "demo videos missing"

at2 = AppTest.from_file(str(MAIN), default_timeout=600)
at2.run()

# Simulate the uploader by writing the saved-upload state the app expects.
at2.session_state["upload_sig"] = tuple((p.name, p.stat().st_size) for p in video_paths)
at2.session_state["uploaded_paths"] = [str(p.resolve()) for p in video_paths]
at2.session_state["videos_loaded"] = True
at2.run()

assert not at2.exception, f"Streamlit raised (phase 2): {at2.exception}"
joined2 = "\n".join(md.value for md in at2.markdown)

checks2 = {
    "Uploaded count shown": "video(s) uploaded" in joined2,
    "Camera 01 card rendered": "Camera 01" in joined2,
    "Camera 02 card rendered": "Camera 02" in joined2,
    "Auto frame label": "auto" in joined2,
    "Object locked hint": ("Constant object" in joined2) or ("OBJ" in joined2),
    "Section 6 header": "6. DISTANCE RESULTS" in joined2,
    "Section 7 header": "7. SCALE RECOVERY" in joined2,
}

print("\nPHASE 2 (after simulated upload):")
for n, ok in checks2.items():
    print(f"  [{'OK ' if ok else 'MISS'}] {n}")

failed += [n for n, ok in checks2.items() if not ok]

# Optional deep check: if the pipeline reports exist, the metric table must render.
target_report = PATH / "results" / "target_point_report.json"
if target_report.exists():
    deep = {
        "Camera->Ground column": "Camera → Ground" in joined2,
        "Camera->Object column": "Camera → Object" in joined2,
        "Single-view source label": "Single-view (ground-plane)" in joined2,
    }
    # Every report camera must appear as a friendly "Camera NN" label or its file name.
    import json as _json
    _layer = PATH / "results" / "visual_metric_layer.json"
    if _layer.exists():
        _names = [c.get("camera") for c in _json.loads(_layer.read_text()).get("cameras", [])]
        _shown = sum(1 for n in _names if n and ("Camera 0" in joined2 or n in joined2))
        deep["All cameras listed in section 6"] = _shown >= len(_names)
    print("\nDEEP (with reports present):")
    for n, ok in deep.items():
        print(f"  [{'OK ' if ok else 'MISS'}] {n}")
    failed += [n for n, ok in deep.items() if not ok]

print()
if failed:
    print("RENDER FAILURES:")
    for f in failed:
        print(" -", f)
    sys.exit(1)

print("RENDER TEST PASSED: upload-first flow renders correctly.")
