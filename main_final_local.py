import streamlit as st
import cv2, json, subprocess, sys
from pathlib import Path

st.set_page_config(page_title="CCTV Distance Analyser", page_icon="📹", layout="wide")

st.markdown("""
<style>
.stApp{background:#07111f;color:#e8eef7}
.block-container{max-width:1500px;padding:1.2rem 2rem 4rem}
.hero,.section{background:#0b1829;border:1px solid #1a304b;border-radius:15px}
.hero{padding:24px 28px;margin-bottom:18px}
.hero h1{margin:0;color:#f5f8ff;font-size:31px}
.hero p,.small{color:#8fa6c2}
.section{padding:17px;margin-bottom:14px}
.section-title{color:#cbd9eb;font-size:13px;font-weight:800;letter-spacing:1px;margin-bottom:13px}
.info,.result{background:#0a1524;border:1px solid #19304a;border-radius:11px;padding:12px}
.result{margin:7px 0}
.ok{color:#65e6ae;font-weight:700}
.wait{color:#91a8c2}
.metric{font-size:25px;font-weight:800;color:#f1f6ff}
div.stButton>button{width:100%;border-radius:9px;background:#10263f;color:#eaf2ff;border:1px solid #294b70;font-weight:700}
[data-testid="stTextInput"] input{background:#081523;color:#eaf2ff;border:1px solid #23415f}
</style>
""", unsafe_allow_html=True)

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
EXT={".mp4",".avi",".mov",".mkv",".wmv",".m4v"}

def videos(folder):
    p=Path(folder)
    return sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in EXT],key=lambda x:str(x).lower()) if p.is_dir() else []

def preview(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None,0,0
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n//2)); ok,f=cap.read(); cap.release()
    return (cv2.cvtColor(f,cv2.COLOR_BGR2RGB) if ok else None),n,fps

def clear_results():
    for p in RESULTS.glob("*.json"):
        try:p.unlink()
        except:pass

if "folder" not in st.session_state: st.session_state.folder=""
if "vids" not in st.session_state: st.session_state.vids=[]
if "point" not in st.session_state: st.session_state.point=None
if "output" not in st.session_state: st.session_state.output=""

st.markdown("""<div class="hero"><h1>📹 CCTV Distance Analyser</h1>
<p>Multi-Camera | Constant Object | 3D Distance Measurement</p></div>""",unsafe_allow_html=True)

a,b,c=st.columns(3)
a.markdown('<div class="section"><b>PROJECT</b><br><span class="small">CCTV Distance Estimation</span></div>',unsafe_allow_html=True)
b.markdown('<div class="section"><b>VIDEO LIMIT</b><br><span class="ok">NO 200 MB BROWSER LIMIT</span><br><span class="small">Direct local-disk access</span></div>',unsafe_allow_html=True)
c.markdown('<div class="section"><b>PRIVACY</b><br><b>LOCAL ONLY</b><br><span class="small">Footage stays on this computer</span></div>',unsafe_allow_html=True)

left,right=st.columns([.31,.69],gap="large")

with left:
    st.markdown('<div class="section"><div class="section-title">1. INPUT VIDEOS</div>',unsafe_allow_html=True)
    st.markdown('<div class="info">Select the folder containing CCTV videos. This app does NOT use Streamlit file upload, so multi-GB files are supported.</div>',unsafe_allow_html=True)
    folder=st.text_input("Local CCTV folder",value=st.session_state.folder,placeholder=r"C:\Users\HP\Desktop\cctv-distance-project\videos")
    x,y=st.columns(2)
    with x: browse=st.button("📁 Browse Folder")
    with y: load=st.button("🔄 Load Videos")
    if browse:
        try:
            import tkinter as tk
            from tkinter import filedialog
            r=tk.Tk();r.withdraw();r.attributes("-topmost",True)
            chosen=filedialog.askdirectory(title="Select CCTV Video Folder");r.destroy()
            if chosen:
                st.session_state.folder=chosen;st.session_state.vids=videos(chosen);st.session_state.point=None;st.session_state.output="";clear_results();st.rerun()
        except Exception as e: st.error(str(e))
    if load:
        found=videos(folder.strip())
        if not found: st.error("No CCTV video files found in that folder.")
        else:
            st.session_state.folder=folder.strip();st.session_state.vids=found;st.session_state.point=None;st.session_state.output="";clear_results();st.rerun()
    if st.session_state.vids:
        st.markdown(f'<p class="ok">✓ {len(st.session_state.vids)} video(s) loaded</p>',unsafe_allow_html=True)
        for i,p in enumerate(st.session_state.vids,1):
            st.markdown(f'<div class="result"><b>Camera {i:02d}</b><br><span class="small">{p.name}<br>{p.stat().st_size/1024**3:.2f} GB</span></div>',unsafe_allow_html=True)
    else: st.markdown('<p class="wait">Waiting for local CCTV folder…</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title">2. CONSTANT OBJECT</div>',unsafe_allow_html=True)
    if st.session_state.vids:
        st.info("Camera 01 is shown on the right. Click the fixed/common point there.")
        if st.session_state.point:
            st.markdown(f'<p class="ok">✓ Point selected: {st.session_state.point}</p>',unsafe_allow_html=True)
        else: st.markdown('<p class="wait">Select a point in Camera 01.</p>',unsafe_allow_html=True)
    else: st.markdown('<p class="wait">Load videos first.</p>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title">3. PROCESSING PIPELINE</div>',unsafe_allow_html=True)
    for s in ["Camera geometry","Common-point correspondence","3D reconstruction","Metric scale recovery","Distance calculation"]:
        st.markdown(f"<b>{s}</b><br><span class='small'>Automatic</span><hr>",unsafe_allow_html=True)
    run=st.button("▶ RUN ANALYSIS",disabled=not(st.session_state.vids and st.session_state.point))
    if run:
        script=ROOT/"src"/"end_to_end_pipeline.py"
        if not script.exists(): st.error(f"Backend not found: {script}")
        else:
            with st.spinner("Processing local CCTV footage…"):
                p=subprocess.run([sys.executable,str(script),st.session_state.folder],cwd=str(ROOT),capture_output=True,text=True,timeout=1800)
            st.session_state.output=(p.stdout or "")+"\n"+(p.stderr or "")
            if p.returncode==0: st.success("Analysis completed.")
            else: st.error("Pipeline error. Open diagnostics below.")
    st.markdown('</div>',unsafe_allow_html=True)

with right:
    st.markdown('<div class="section"><div class="section-title">4. MULTI-CAMERA VIEWS</div>',unsafe_allow_html=True)
    if not st.session_state.vids:
        st.markdown('<div class="info" style="height:250px;text-align:center;padding-top:110px">No CCTV footage loaded.<br>Choose a local folder to begin.</div>',unsafe_allow_html=True)
    else:
        cols=st.columns(min(3,len(st.session_state.vids)))
        for i,p in enumerate(st.session_state.vids):
            with cols[i%len(cols)]:
                f,n,fps=preview(p)
                st.markdown(f"**CAMERA {i+1:02d}**")
                if f is not None: st.image(f,use_container_width=True);st.caption(f"{p.name} • {n:,} frames • {fps:.2f} FPS")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title">5. 3D RECONSTRUCTION</div>',unsafe_allow_html=True)
    st.markdown('<div class="info" style="height:170px;text-align:center;padding-top:70px">3D reconstruction will appear here after the selected target point is processed.</div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title">6. DISTANCE RESULTS</div>',unsafe_allow_html=True)
    report=RESULTS/"final_pipeline_report.json"
    if report.exists():
        try:
            data=json.loads(report.read_text(encoding="utf-8"))
            st.success("Pipeline report available.")
            st.json(data)
        except Exception: st.info("Report created but could not be displayed.")
    else: st.info("No distance result yet. Select a constant point and run analysis.")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-title">7. SCALE RECOVERY</div><div class="info">Metric scale status will appear after analysis.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section"><div class="section-title">8. SYSTEM STATUS</div>',unsafe_allow_html=True)
    if not st.session_state.vids: st.markdown('<span class="wait">● READY — waiting for local CCTV videos</span>',unsafe_allow_html=True)
    elif not st.session_state.point: st.markdown('<span class="wait">● READY — select the constant object / point</span>',unsafe_allow_html=True)
    else: st.markdown('<span class="ok">● READY — target selected</span>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

    if st.session_state.output:
        with st.expander("Pipeline diagnostics"): st.code(st.session_state.output)

st.markdown('<div class="small" style="text-align:center">CCTV Distance Analyser • Local processing • No browser upload</div>',unsafe_allow_html=True)
