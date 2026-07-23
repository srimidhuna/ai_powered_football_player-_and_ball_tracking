<div align="center">
  <h1>⚽ AI-Powered Football Tracker</h1>
  <p>
    <strong>A production-ready football analytics pipeline with YOLOv8, RF-DETR Large, and ByteTrack.</strong>
  </p>
  <p>
    <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/YOLOv8-8A2BE2" alt="YOLOv8"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://roboflow.com/"><img src="https://img.shields.io/badge/RF--DETR-6706CE" alt="RF-DETR"></a>
  </p>
</div>

---

## 🌟 Overview

This repository provides an advanced, end-to-end computer vision pipeline for football match analysis. It tracks players, referees, and the ball, classifies teams using jersey colors, calculates ball possession stats in real-time, and generates minimap visualizations.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    Input[Input Video] -->|Frames| YOLO[YOLO v8m]
    Input -->|Frames| RFDETR[RF-DETR Large]
    
    YOLO -->|Detections| Merged[Merge Detections]
    RFDETR -->|Ball Detection| Merged
    
    Merged --> Tracker[ByteTrack / Supervision]
    Tracker --> Teams[Team Classifier<br>(KMeans + DBSCAN)]
    Tracker --> BallTrack[Ball Tracker<br>(Kalman Filter)]
    Tracker --> Referee[Referee Detector]
    
    Teams --> Posses[Possession Engine]
    BallTrack --> Posses
    BallTrack --> Traj[Trajectory Engine]
    
    Posses --> Vis[Visualizer]
    Traj --> Vis
    Referee --> Vis
    
    Vis --> Output[Output Video]
```

---

## ✨ Key Features

| Feature | Technology | Description |
|---|---|---|
| **Player & Referee Detection** | YOLOv8m (COCO) | Specialized validation to filter out noise and distinguish referees from players. |
| **Football Detection** | RF-DETR Large | Custom Roboflow model specifically for sports balls. |
| **Multi-Object Tracking** | ByteTrack | Assigns and maintains consistent IDs for players via Supervision. |
| **Team Classification** | KMeans + DBSCAN | Automatically separates teams based on jersey color extraction and clustering. |
| **Ball Tracking** | Kalman Filter | Tracks the ball smoothly, predicting trajectory even during brief occlusions. |
| **Ball Possession Stats** | Proximity Algorithms | Attributes possession based on player proximity and time metrics. |
| **Minimap & Trajectory** | OpenCV | Generates a live minimap and draws ball trajectories. |
| **Automated Testing** | Pytest | Includes comprehensive unit tests and validation scripts (`validate_tracking.py`). |

---

## 🚀 Quick Start

### 1. Install dependencies
Clone the repository and install the required packages:
```bash
pip install ultralytics supervision filterpy opencv-python numpy scipy scikit-learn
pip install -r requirements.txt
```

### 2. Run the tracking pipeline
Process a video file and generate the analytics overlay:
```bash
python main.py --input input/your_video.mp4 --output output/result.mp4
```

### 3. Generate a Web-Friendly Format
Convert the processed video to H.264 format for smooth playback in IDEs and browsers:
```bash
python transcode.py output/result.mp4 output/result_h264.mp4
```

---

## 📁 Project Structure

```text
football_tracker/
├── main.py                 # Pipeline orchestration & execution
├── config.py               # Configurable thresholds and hyperparameters
├── detector.py             # YOLO inference + player validation
├── referee_detector.py     # Logic to detect and filter referees
├── tracker.py              # ByteTrack wrapper for persistence
├── ball_tracker.py         # Kalman filter ball tracking engine
├── team_classifier.py      # KMeans + DBSCAN team assignment logic
├── possession.py           # Ball possession statistics calculation
├── trajectory.py           # Ball trail and path rendering
├── visualizer.py           # HUD, stats, and annotation rendering
├── minimap.py              # 2D pitch minimap generation
├── validate_tracking.py    # Analytics validation tools
├── transcode.py            # mp4v → H.264 conversion utility
├── utils.py                # Logger and system helpers
└── test_*.py               # Unit tests for core components
```

---

## 🧠 Deep Dive: How It Works

### Player & Referee Logic (YOLO)
- Detects: `player`, `goalkeeper`, `referee`.
- Excludes YOLO's native ball class to avoid false positives.
- Implements strict validation criteria: aspect ratios, pitch masking, top-frame exclusion (to ignore overhead mics/cameras), and motion consistency checks.

### Ball Tracking (RF-DETR & Kalman Filter)
- The pipeline uses a specialized RF-DETR model purely for ball tracking.
- All non-ball classes detected by RF-DETR are immediately rejected.
- A **Kalman Filter** tracks the ball's momentum and handles interpolation when the ball is obscured by players or leaves the camera's view temporarily.

---

## ⚙️ Configuration (`config.py`)

Key parameters that can be tuned based on the camera angle and lighting of the input video:

| Parameter | Default | Description |
|---|---|---|
| `CONF_PLAYER` | `0.3` | Confidence threshold for YOLO player detection. |
| `CONF_BALL` | `0.2` | Confidence threshold for ball detection. |
| `PLAYER_MIN_AREA` | `500` | Minimum bounding box area for players. |
| `BROADCAST_TOP_REGION` | `0.10` | Top 10% of frame excluded (ignores broadcast overlay). |
| `PLAYER_PERSISTENCE_FRAMES`| `3` | Frames required before a player detection is deemed active. |
| `POSSESSION_MAX_DISTANCE` | `100` | Max pixel distance to attribute ball possession. |

---

## 🛠️ Testing

The project is equipped with unit tests for each major component to ensure robust tracking and logic.
Run tests using:
```bash
python -m unittest discover -p "test_*.py"
```

You can also validate the full tracking sequence over sample files:
```bash
python validate_tracking.py
```
