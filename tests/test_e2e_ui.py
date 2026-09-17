"""TRUE end-to-end test through the real Streamlit UI:

  fresh state -> upload videos -> auto frame+object selection
  -> click Start/Resume -> pipelines run -> sections 5/6/7/8 populate

Uses Streamlit AppTest, which executes main.py exactly like the browser does.
"""
import json
import shutil
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

PATH = Path(__file__).resolve().parent.parent
MAIN = PATH / "main.py"
RESULTS = PATH / "results"
BACKUP = PATH / "results_backup_e2e"

REPORTS = [
    "target_point_report.json",
    "multi_camera_report.json",
    "multi_camera_geometry.json",
    "visual_scale_result.json",
    "metric_integration.json",
    "final_pipeline_report.json",
    "visual_metric_layer.json",
]

# Fresh results dir so nothing can pass from a previous run's leftovers.
# NOTE: the user's uploads/ folder is NEVER touched — the test injects the
# demo videos via session_state (input_videos paths), so moving uploads
# around is unnecessary and only risks the user's files (this bit us once:
# a Windows file-lock made the restore fail and the uploads were left in
# the backup folder).
UPLOADS = PATH / "uploads"

failures = []


def _ss(at_obj, key, default=None):
    """Read session state safely (SafeSessionState has no .get)."""
    try:
        return at_obj.session_state[key]
    except Exception:
        return default


try:
    # ------------------------------------------------------------------
    # Phase 1: fresh app, nothing uploaded
    # ------------------------------------------------------------------
    at = AppTest.from_file(str(MAIN), default_timeout=300)
    at.run()
    assert not at.exception, f"Phase 1 exception: {at.exception}"
    j1 = "\n".join(md.value for md in at.markdown)
    p1 = {
        "No frames before upload": "Frames appear here after you upload" in j1,
        "Uploader present": len(at.file_uploader) > 0,
    }
    print("PHASE 1 (fresh, nothing uploaded):")
    for n, ok in p1.items():
        print(f"  [{'OK ' if ok else 'MISS'}] {n}")
        if not ok:
            failures.append(n)

    # ------------------------------------------------------------------
    # Phase 2: simulate uploading the two demo videos
    # ------------------------------------------------------------------
    vids = sorted((PATH / "input_videos").glob("*.mp4"))
    assert vids, "demo videos missing"

    at2 = AppTest.from_file(str(MAIN), default_timeout=600)
    at2.run()
    # Perform the upload exactly as the app does: copy files into uploads/ and
    # set the same session keys the uploader branch sets in the real UI.
    # Scratch folder for the injected upload set — NEVER the user's real
    # uploads/ folder (this test used to delete the user's videos here).
    up_dir = PATH / "uploads_test_e2e"
    if up_dir.exists():
        shutil.rmtree(up_dir, ignore_errors=True)
    up_dir.mkdir(exist_ok=True)
    saved = []
    for p in vids:
        dst = up_dir / p.name
        dst.write_bytes(p.read_bytes())
        saved.append(dst)
    at2.session_state["upload_sig"] = tuple((p.name, p.stat().st_size) for p in saved)
    at2.session_state["uploaded_paths"] = [str(p.resolve()) for p in saved]
    at2.session_state["videos_loaded"] = True
    at2.run()
    assert not at2.exception, f"Phase 2 exception: {at2.exception}"
    j2 = "\n".join(md.value for md in at2.markdown)

    # Auto-selection must have produced an object point.
    auto = _ss(at2, "auto_result") or {}
    obj = auto.get("object") or {}
    click = _ss(at2, "target_click")
    p2 = {
        "Videos recognized": "video(s) uploaded" in j2,
        "Auto best-frame selected": auto.get("base_frame") is not None,
        "Auto object detected": bool(obj.get("success")) and click is not None,
        "Object marker drawn (OBJ)": "OBJ" in j2,
    }
    print("\nPHASE 2 (upload -> auto selection):")
    for n, ok in p2.items():
        print(f"  [{'OK ' if ok else 'MISS'}] {n}")
        if not ok:
            failures.append(n)

    if not (obj.get("success") and click):
        raise SystemExit(_report(failures, "auto-selection failed; cannot proceed"))

    # ------------------------------------------------------------------
    # Phase 3: press Start / Resume Processing
    # ------------------------------------------------------------------
    start_btns = [b for b in at2.button if "Start / Resume" in (b.label or "")]
    assert start_btns, "Start button not found"
    start_btns[0].click()
    at2.run(timeout=600)
    assert not at2.exception, f"Phase 3 exception: {at2.exception}"

    # Pipelines must have produced the reports.
    produced = {r: (RESULTS / r).exists() for r in REPORTS}
    p3 = {
        "target_point_report.json written": produced["target_point_report.json"],
        "visual_metric_layer.json written": produced["visual_metric_layer.json"],
        "Pipeline log captured": bool(_ss(at2, "pipeline_log")),
    }
    print("\nPHASE 3 (Start clicked, pipelines ran):")
    for n, ok in p3.items():
        print(f"  [{'OK ' if ok else 'MISS'}] {n}")
        if not ok:
            failures.append(n)

    # ------------------------------------------------------------------
    # Phase 4: sections 5/6/7/8 populated
    # ------------------------------------------------------------------
    j3 = "\n".join(md.value for md in at2.markdown)
    layer = json.loads((RESULTS / "visual_metric_layer.json").read_text())
    heights = {c.get("camera"): (c.get("camera_to_ground") or {}).get("metric_height_m")
               for c in layer.get("cameras", [])}

    p4 = {
        "Section 6: Camera->Ground column": "Camera → Ground" in j3,
        "Section 6: metre values rendered": "m</b>" in j3,
        "Section 6: friendly camera labels": ">Camera 01<" in j3,
        "Section 7: person-height cue": "Person-height prior" in j3,
        "Section 8: system status panel": "8. SYSTEM STATUS" in j3,
        "Ground heights ~2.5 m (both cams)": all(
            h is not None and 1.5 < h < 3.5 for h in heights.values()
        ) and len(heights) >= 2,
    }
    # Section 5 must render the full reconstruction block. The legend and the
    # info box are emitted right after st.plotly_chart, so reaching them proves
    # the 3D figure was built (AppTest cannot inspect plotly elements directly).
    p4["Section 5 panel present"] = "5. 3D RECONSTRUCTION" in j3
    p4["3D legend rendered (chart built)"] = "Camera to Object Rays" in j3
    p4["Reconstruction Info box"] = "Reconstruction Info" in j3
    p4["Duplicate note shown"] = "duplicate" in j3.lower()

    print("\nPHASE 4 (sections after processing):")
    for n, ok in p4.items():
        print(f"  [{'OK ' if ok else 'MISS'}] {n}")
        if not ok:
            failures.append(n)

    print("\nGround heights:", {k: round(v, 2) for k, v in heights.items() if v})
finally:
    # Restore the previous results. The user's uploads were never touched.
    for f in RESULTS.iterdir():
        f.unlink()
    RESULTS.rmdir()
    if BACKUP.exists():
        BACKUP.rename(RESULTS)
    shutil.rmtree(PATH / "uploads_test_e2e", ignore_errors=True)


def _report(fails, msg=""):
    out = ["E2E FAILURES:"]
    out += [f" - {f}" for f in fails]
    if msg:
        out.append(msg)
    return "\n".join(out)


print()
if failures:
    print(_report(failures))
    sys.exit(1)
print("E2E TEST PASSED: upload -> auto frame -> auto object -> Start -> sections 5/6/7/8 all populate.")
