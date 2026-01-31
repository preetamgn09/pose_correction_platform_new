# Installation Guide

Complete setup instructions for the **Yoga Pose Correction Platform**.

This project supports **multiple pose models**, but due to ecosystem constraints, **not all models can coexist in the same Python environment**. This guide explains the **correct and stable setup**.

---

## System Requirements

### Hardware

* **Minimum**

  * CPU: Intel i5 / Ryzen 5
  * RAM: 8 GB
  * Webcam (720p)

* **Recommended**

  * CPU: Intel i7 / Ryzen 7
  * RAM: 16 GB
  * GPU: NVIDIA GPU (optional, for YOLO acceleration)
  * Webcam: 1080p

---

## Software

* Python **3.10 recommended**
* OS:

  * Windows 10/11
  * Ubuntu 20.04+
  * macOS 12+

---

## Important Design Note (READ THIS)

⚠️ **MediaPipe and TensorFlow (MoveNet) are NOT compatible in the same Python environment on Windows.**

This is a known dependency conflict involving `protobuf`.

### Correct approach:

* Use **separate virtual environments per model**
* The **pose graph (JSON)** is reusable across all environments

---

## Project Files

Required files:

```
pose_features.py
state_discovery.py
pose_graph.py
offline_graph_generator.py
online_monitor_updated.py
pose_adapters.py
```

---

## Environment Setup (RECOMMENDED)

### Environment 1: YOLOv8 (Stable, All-in-One)

Use this if you want the **simplest setup**.

```bash
python -m venv poseEnv_yolo
poseEnv_yolo\Scripts\activate  # Windows
```

```bash
pip install ultralytics opencv-python numpy
```

YOLO models auto-download on first use.

Run:

```bash
python online_monitor_updated.py --graph your_graph.json --model yolo --camera 0
```

---

### Environment 2: MoveNet (Mobile-Aligned)

```bash
python -m venv poseEnv_movenet
poseEnv_movenet\Scripts\activate
```

```bash
pip install tensorflow opencv-python numpy
```

#### Download MoveNet (Required)

Download **TFLite**, not SavedModel:

* Thunder (accurate):
  [https://tfhub.dev/google/movenet/singlepose/thunder/4?lite-format=tflite](https://tfhub.dev/google/movenet/singlepose/thunder/4?lite-format=tflite)

Rename to:

```
movenet_thunder.tflite
```

Place it **in the same directory as `pose_adapters.py`**.

Run:

```bash
python online_monitor_updated.py --graph your_graph.json --model movenet --camera 0
```

---

### Environment 3: MediaPipe (Optional)

```bash
python -m venv poseEnv_mediapipe
poseEnv_mediapipe\Scripts\activate
```

```bash
pip install mediapipe==0.10.9
pip install protobuf==3.20.3
pip install opencv-python numpy
```

Run:

```bash
python online_monitor_updated.py --graph your_graph.json --model mediapipe --camera 0
```

---

## Offline Pose Graph Generation

Pose graphs are **model-agnostic** and stored as **JSON**.

```bash
python offline_graph_generator.py reference_video.mp4 --output graph.json
```

You do **NOT** need to regenerate graphs for different models.

---

## Common Issues & Fixes

### MediaPipe crash: `GetPrototype` error

✔ Fix:

```bash
pip install protobuf==3.20.3
```

### MoveNet error: `.tflite not found`

✔ Fix:

* Download **TFLite model**
* Do **NOT** rename `saved_model.pb`
* Place `.tflite` file correctly

### Skeleton not visible

✔ Cause:

* Keypoints were normalized
  ✔ Fix:
* Convert keypoints to **pixel coordinates in adapters**
* Shared `draw_skeleton()` handles all models

### `PoseGraph.load` missing

✔ Fix:

* Add `load()` alias method in `pose_graph.py`

---

## Platform Notes

### Windows

* Use Python 3.10
* Avoid mixing MediaPipe + TensorFlow
* Camera permission must be enabled

### Linux

```bash
sudo apt install libgl1
```

### macOS (Apple Silicon)

* YOLO uses CPU
* MoveNet works well
* MediaPipe supported via native SDKs (not Python)

---

## Performance Expectations

| Model     | FPS (CPU) | Notes                 |
| --------- | --------- | --------------------- |
| YOLOv8-n  | 20–30     | Best desktop choice   |
| MoveNet   | 25–40     | Best mobile alignment |
| MediaPipe | 20–30     | Lightweight           |

---

## Mobile Deployment (Important)

⚠️ **Python code does NOT run on mobile OS.**

For Android / iOS:

* Use **MoveNet TFLite / MediaPipe SDK**
* Reuse:

  * Pose graph JSON
  * State logic
  * Accuracy / Quality definitions

The current codebase is a **reference implementation**.

---

## Summary (Read This Once)

* ✔ YOLO → easiest, stable
* ✔ MoveNet → mobile-ready
* ❌ MediaPipe + TensorFlow in same env → not supported
* ✔ Graph JSON reusable everywhere
* ✔ Architecture is mobile-safe

---

If you want, next I can:

* split this into **INSTALL_YOLO.md / INSTALL_MOVENET.md**
* generate **requirements.txt per environment**
* prepare **Android MoveNet mapping notes**

Just say what you want next.
