**VisionMetric-AI**

### Computer Vision-Based CCTV Distance Estimation

VisionMetric-AI is a computer vision system that analyzes CCTV footage to estimate **camera height** and **camera-to-object distance** using geometric reasoning, visual scale estimation, and object detection.

The project explores how conventional CCTV cameras can be used to derive approximate real-world spatial measurements without requiring dedicated depth sensors.

## Key Features

- **CCTV Video Analysis** — Process recorded surveillance footage.
- **Camera Height Estimation** — Estimate camera elevation using scene geometry.
- **Object Distance Estimation** — Estimate the distance between a camera and a selected object.
- **Geometric Vision** — Uses perspective geometry, vanishing points, and ground-plane analysis.
- **Object Detection** — Integrates YOLO for visual object identification.
- **Metric Scale Recovery** — Converts visual information into real-world spatial estimates.
- **Multi-Camera Support** — Supports analysis across multiple CCTV viewpoints.
- **Web Interface** — Provides an accessible interface for running the analysis.

## How It Works

```text
CCTV Video
     ↓
Frame Extraction
     ↓
Computer Vision Analysis
     ↓
Camera & Ground Geometry
     ↓
Metric Scale Estimation
     ↓
Object Localization
     ↓
Distance Estimation

```

## Tech Stack

**Python • OpenCV • YOLO • Computer Vision • Projective Geometry • HTML • CSS • JavaScript**

## Project Structure

```text
VisionMetric-AI/
├── src/                    # Core computer vision & geometry
├── calibration/            # Camera/scene calibration
├── tests/                  # Testing & validation
├── cctv_distance_webapp/   # Web application
├── app.py                  # Application entry point
├── main.py                 # Main pipeline
├── camera_config.json      # Camera configuration
└── yolo11n.pt              # YOLO model

```

## Author

**SAI SOMWYA S**
Computer Science & Engineering (Data Science)

Exploring the intersection of AI, computer vision, and geometric spatial reasoning.
