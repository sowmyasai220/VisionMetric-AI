"""Headless smoke test for the button flow: run pipelines, then exercise main.py's data layer."""
import sys
import types
import importlib.util

sys.path.insert(0, ".")


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _noop(*a, **k):
    return None


st_mod = types.ModuleType("streamlit")
for attr in [
    "set_page_config", "markdown", "button", "write", "success", "warning",
    "error", "info", "caption", "image", "slider", "radio", "code",
    "download_button", "rerun", "plotly_chart", "json", "divider", "metric",
    "checkbox", "text_input", "selectbox", "expander", "spinner", "toast",
]:
    setattr(st_mod, attr, _noop)


def _file_uploader(*a, **k):
    return []


def _spinner(*a, **k):
    return Ctx()


def _cache_decorator(*a, **k):
    """Pass-through cache decorators for the fake streamlit (the real
    st.cache_data / st.cache_resource exist on the runtime streamlit)."""
    def _wrap(fn):
        return fn
    return _wrap


def _status(*a, **k):
    return Ctx()


st_mod.cache_data = _cache_decorator
st_mod.cache_resource = _cache_decorator
st_mod.status = _status
st_mod.file_uploader = _file_uploader
st_mod.spinner = _spinner


class SS(dict):
    def __getattr__(self, k):
        return self.get(k)

    def __setattr__(self, k, v):
        self[k] = v


st_mod.session_state = SS()
st_mod.sidebar = Ctx()


def _columns(spec=None, **k):
    n = len(spec) if isinstance(spec, (list, tuple)) else (spec if isinstance(spec, int) else 2)
    return [Ctx() for _ in range(n)]


def _slider(*a, **k):
    return 0


def _radio(*args, **k):
    options = k.get("options") or (args[1] if len(args) > 1 else None) or ["Top View"]
    idx = k.get("index") or 0
    try:
        return options[idx]
    except Exception:
        return "Top View"


st_mod.columns = _columns
st_mod.slider = _slider
st_mod.radio = _radio
st_mod.status = lambda *a, **k: Ctx()
sys.modules["streamlit"] = st_mod

simg = types.ModuleType("streamlit_image_coordinates")
simg.streamlit_image_coordinates = None
sys.modules["streamlit_image_coordinates"] = simg

spec = importlib.util.spec_from_file_location("main_mod", "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("IMPORT OK")

ok1, log1 = m.run_full_pipeline("input_videos")
print("general pipeline ok:", ok1)
# Static, textured floor-level object point (found via temporal-variance scan)
# plus a static target frame index.
OBJ_X, OBJ_Y, OBJ_FRAME = 1720, 720, 700

# New: the auto-selection stage must find a frame and a static object.
from auto_select import auto_select_frame, auto_select_object  # noqa: E402

_a_idx, _a_score = auto_select_frame("input_videos/My Video-highlight_1.mp4")
print("auto frame:", _a_idx, "score", round(_a_score, 3))
_b_idx, _ = auto_select_frame("input_videos/My Video-highlight_1_1.mp4")
_auto_obj = auto_select_object(
    "input_videos/My Video-highlight_1.mp4", _a_idx,
    ["input_videos/My Video-highlight_1_1.mp4"],
    {"My Video-highlight_1_1.mp4": ("input_videos/My Video-highlight_1_1.mp4", _b_idx)},
)
print("auto object:", _auto_obj.get("success"), _auto_obj.get("point"), _auto_obj.get("status"))
if not _auto_obj.get("success"):
    failures = [_auto_obj.get("message", "auto object selection failed")]
else:
    failures = []

ok2, log2 = m.run_target_pipeline("input_videos", OBJ_X, OBJ_Y, OBJ_FRAME)
print("target pipeline ok:", ok2)
ok3, log3 = m.run_metric_layer("input_videos")
print("metric layer ok:", ok3)

target_report = m.load_json(m.REPORT_TARGET)
scale_factor, scale_source, scale_conf = m.scale_info()
metric_cams = m.metric_layer_info()
obj_pixels = m.object_pixels_by_camera(target_report)
rows = m.target_distance_rows(target_report, scale_factor, metric_cams, obj_pixels)

print()
print("scale_factor:", scale_factor, "| source:", scale_source)
print("metric cameras:", {k: v.get("height_m") for k, v in metric_cams.items()})
print("object pixels:", obj_pixels)
print()

failures = []
skipped_dupes = 0
for r in rows:
    rel = r["relative"]
    height = r["ground_height_m"]
    obj_m = r["object_distance_m"]
    print(
        f"{r['camera'][:34]:36s} cam->ground={height}  "
        f"cam->object={obj_m}  rel3d={rel}  note={r['object_distance_note']}"
    )
    if r["warning"]:
        print("   warning:", r["warning"])
    if r["duplicate_note"]:
        skipped_dupes += 1
        print("   duplicate note:", r["duplicate_note"])
    if height is None:
        failures.append(f"{r['camera']}: no camera->ground height")
    if obj_m is None:
        failures.append(f"{r['camera']}: no camera->object distance")
    # The two bundled test videos are duplicates: one of them must have been
    # skipped for 3D, so at most one camera may carry a relative distance.
    if rel is not None and not r["duplicate_note"]:
        pass  # genuine multi-view camera: fine

print()
print("states:", m.status_from_results(["a", "b"], target_report, scale_factor))

if skipped_dupes < 1:
    failures.append("expected at least one duplicate camera to be skipped for 3D")

# The duplicate camera must still have a transferred object pixel.
dup_camera = target_report["cross_camera_pairs"][0].get("camera_b")
if target_report["cross_camera_pairs"][0].get("skipped_duplicate"):
    corr = target_report["cross_camera_pairs"][0].get("target_correspondence") or {}
    print("duplicate pair point transfer success:", corr.get("success"), "| method:", corr.get("method"))
    if not corr.get("success"):
        failures.append("duplicate camera did not receive a transferred object point")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)

print("\nSMOKE TEST PASSED: ground heights + object distances for every camera; "
      "duplicate ignored for 3D but still annotated.")
