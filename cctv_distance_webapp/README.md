# CCTV Distance Estimation — Local Web Application

## Start

From the project root:

```cmd
venv\Scripts\activate
py -m pip install -r cctv_distance_webapp\requirements.txt
py -m uvicorn cctv_distance_webapp.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

Place any number of recorded CCTV files in `input_videos`.

## Workflow

1. Refresh videos.
2. Select a fixed/common target point in Camera 01.
3. Start Analysis.
4. The server estimates cross-camera correspondence, relative pose, target triangulation, reprojection validity and relative distance.
5. A metric estimate is shown only when the optional visual scale cue succeeds.

The application does not upload CCTV footage.
