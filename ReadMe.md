# Real-Time Pose Correction System

A production-ready real-time pose correction platform for yoga and exercise guidance.
The system automatically generates **pose-state graphs from reference videos** and provides **live accuracy, quality, and biomechanical feedback** during user sessions.

---

## Overview

The system operates in two phases:

1. **Offline Phase**
   Builds a **model-agnostic pose graph (JSON)** from a reference video.

2. **Online Phase**
   Uses the pose graph to guide users through correct pose sequences using live camera input.

The same pose graph can be reused across **YOLOv8, MoveNet, and MediaPipe**.

---

## Key Features

* Automatic pose graph discovery (no manual labeling)
* Deterministic state-machine execution
* Multi-model pose estimation via adapters
* Two-metric evaluation system:

  * **Accuracy** (binary correctness)
  * **Quality** (per-state refinement)
* Intelligent biomechanical feedback
* Single-inference-per-frame (performance safe)
* Mobile-aligned architecture (MoveNet ready)

---

## Architecture

```
Reference Video → Pose Graph Generator → pose_graph.json
                                             ↓
Camera → Pose Adapter → Feature Extraction → State Machine → Feedback
                                                   ↓
                                              Visualization
```

---

## Core Components

### Offline

* `offline_graph_generator.py` – generates pose graphs (JSON)
* `state_discovery.py` – pose state segmentation & clustering
* `pose_graph.py` – graph data structure & loader

### Online

* `online_monitor_updated.py` – real-time monitoring engine
* `pose_adapters.py` – YOLO / MoveNet / MediaPipe adapters
* `pose_features.py` – biomechanical feature extraction

---

## Metrics (Frozen Design)

### Accuracy — Binary Correctness

* Frame-level GREEN / RED classification
* Starts after first GREEN
* Paused during transitions
* Formula:

  ```
  Accuracy = (GREEN frames / total evaluated frames) × 100
  ```

### Quality — Per-State Refinement

* Computed **only during GREEN**
* Reset on:

  * leaving GREEN
  * changing pose state
* Sliding-window smoothing (recoverable)
* One quality score per completed state
* Final quality = average of all state qualities
* **Does not decay during correct holds**

### Feedback — Coaching Layer

* Triggered on:

  * RED poses
  * sustained low-quality GREEN
* Biomechanical priority:

  1. Balance / stability
  2. Torso alignment
  3. Lower body
  4. Upper body
* Rate-limited and paused during transitions

---

## Multi-Model Pose Support

All models are normalized to **COCO-17 keypoints in pixel space**, enabling:

* identical skeleton rendering
* shared feature extraction
* reusable pose graphs

| Model            | Best Use                       |
| ---------------- | ------------------------------ |
| YOLOv8 Pose      | Desktop / GPU                  |
| MoveNet (TFLite) | Mobile & cross-platform        |
| MediaPipe Pose   | Lightweight CPU (separate env) |

---

## Project Structure

```
pose-correction-system/
├── offline_graph_generator.py
├── online_monitor_updated.py
├── pose_graph.py
├── state_discovery.py
├── pose_features.py
├── pose_adapters.py
├── reference_videos/
└── pose_graphs/
    └── exercise_graph.json
```

---

## Usage

### 1️⃣ Generate Pose Graph (Offline)

```bash
python offline_graph_generator.py reference_videos/pushup.mp4 --output pose_graphs/pushup_graph.json
```

**Notes**

* Output is always **JSON**
* Graph is **model-agnostic**
* No `--model` flag required

---

### 2️⃣ Run Online Monitoring

```bash
python online_monitor_updated.py --graph pose_graphs/pushup_graph.json --model yolo --camera 0 --max-reps 10
```

Supported models:

```bash
--model yolo
--model movenet
--model mediapipe
```

---

## Installation (IMPORTANT)

⚠️ **MediaPipe and TensorFlow (MoveNet) cannot coexist in the same Python environment on Windows** due to protobuf conflicts.

### Recommended Setup

#### Environment A — YOLO / MoveNet

```bash
pip install -r requirements.txt
```

#### Environment B — MediaPipe (separate)

```bash
pip install mediapipe==0.10.9 protobuf==3.20.3 opencv-python numpy
```

Pose graphs (`.json`) are reusable across environments.

---

## MoveNet Model Requirement

MoveNet requires a **TFLite file** (not SavedModel).

Download:

* [https://tfhub.dev/google/movenet/singlepose/thunder/4?lite-format=tflite](https://tfhub.dev/google/movenet/singlepose/thunder/4?lite-format=tflite)

Rename to:

```
movenet_thunder.tflite
```

Place it in the same directory as `pose_adapters.py`.

---


## Mobile Deployment

⚠️ Python code **does not run on mobile OS**.

Mobile implementation:

* Android → MoveNet TFLite / MediaPipe SDK
* iOS → MoveNet / MediaPipe iOS

Reusable assets:

* pose graph JSON
* state logic
* accuracy / quality definitions

Current Python code serves as a **reference implementation**.

---

## License

MIT License

---

## Status

**Version**: 1.0.0
**State**: Frozen & Stable
**Last Updated**: January 2026


