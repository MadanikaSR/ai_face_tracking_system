# 🎯 AI Face Tracking & Visitor Analytics System

> **Hackathon Project** — Built for the [Katomaran Hackathon](https://katomaran.com)

A production-grade, real-time face tracking and visitor analytics system. Detects, recognizes, and tracks faces from video files or live RTSP streams — assigning each unique person a persistent ID and logging every entry and exit event exactly once.

---

## ✨ Key Features

| Capability               | Detail                                                                        |
| ------------------------ | ----------------------------------------------------------------------------- |
| **Face Detection**       | YOLOv8-based pipeline with InsightFace SCRFD (fast, landmark-aware)           |
| **Face Recognition**     | ArcFace (w600k_r50) — 512-d embeddings using cosine similarity                |
| **Tracking**             | ByteTrack — stable multi-object tracking across frames                        |
| **Auto Registration**    | New faces get a unique ID after accumulating multiple high-quality embeddings |
| **Re-identification**    | Lost tracks can re-inherit their original ID within a short temporal window   |
| **Entry / Exit Logging** | Each face is logged exactly once per session (entry + exit)                   |
| **Duplicate Prevention** | IoU overlap + embedding similarity prevent duplicate IDs                      |
| **Unique Visitor Count** | Accurate count from SQLite, exposed via dashboard                             |
| **Input Sources**        | Video file (`.mp4`, `.avi`, `.mkv`) + RTSP stream                             |

---

## 🏗️ Project Structure

```
ai_face_tracking_system/
├── app.py
├── config.json
├── download_models.py
├── requirements.txt
│
├── core/
│   ├── detector.py
│   ├── recognizer.py
│   ├── tracker.py
│   ├── identity_manager.py
│   ├── pipeline.py
│   └── event_manager.py
│
├── database/
│   └── db.py
│
├── logging_system/
│   └── logger.py
│
├── ui/
│   ├── viewer.py
│   └── launcher.py
│
├── utils/
│   ├── helpers.py
│   └── alignment.py
│
├── logs/
│   ├── entries/
│   └── exits/
│
└── models/
```

---

## 🚀 Quick Start

### 1 — Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
```

### 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 — Download AI Models

```bash
python download_models.py
```

### 4 — Configure Input Source

Edit `config.json`:

```json
{
  "input_type": "video",
  "video_path": "path/to/video.mp4",
  "rtsp_url": "rtsp://username:password@ip:port/stream",
  "reset_on_start": false,
  "detection_frame_skip": 2,
  "recognition_threshold": 0.55,
  "exit_timeout_seconds": 120
}
```

Set `"input_type"` to `"video"` or `"rtsp"`.

---

### 5 — Run the System

```bash
python app.py
```

---

### 6 — Open Analytics Dashboard

```bash
streamlit run ui/viewer.py
```

---

## ⚙️ Configuration Reference

| Key                     | Description                         |
| ----------------------- | ----------------------------------- |
| `input_type`            | `"video"` or `"rtsp"`               |
| `detection_frame_skip`  | Frame skipping for performance      |
| `recognition_threshold` | Face similarity threshold           |
| `exit_timeout_seconds`  | Time before exit event is triggered |

---

## 🧠 System Architecture

```
Video / RTSP
     │
     ▼
┌─────────────┐    bboxes + landmarks
│  SCRFD Det. │ ──────────────────────►┐
└─────────────┘                        │
                                       ▼
                               ┌──────────────┐
                               │  ByteTrack   │
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │   ArcFace    │
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────────────┐
                               │  Identity Manager    │
                               │  · Matching          │
                               │  · Re-identification │
                               │  · Duplicate control │
                               └──────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   ┌────────────┐          ┌──────────────┐
                   │  SQLite DB │          │  Event Logs  │
                   │  faces     │          │  logs/entries│
                   │  events    │          │  logs/exits  │
                   └────────────┘          └──────────────┘
```

---

## 📊 Dashboard

The Streamlit dashboard provides:

* Unique visitor count
* Entry / exit event statistics
* Event logs (timestamp, ID, event type)
* Registered faces grid (image + metadata)

---

## 🛠️ Tech Stack

| Layer            | Technology                 |
| ---------------- | -------------------------- |
| Face Detection   | YOLOv8 + InsightFace SCRFD |
| Face Recognition | InsightFace ArcFace (ONNX) |
| Tracking         | ByteTrack                  |
| Backend          | Python                     |
| Inference        | ONNX Runtime               |
| Database         | SQLite                     |
| UI               | Streamlit                  |
| Launcher         | Tkinter                    |

---

## ⚡ Hardware Requirements

| Mode     | Minimum      | Recommended      |
| -------- | ------------ | ---------------- |
| CPU-only | i5 / Ryzen 5 | i7 / Ryzen 7     |
| GPU      | —            | NVIDIA GTX 1060+ |
| RAM      | 4 GB         | 8 GB+            |

---

## ⚠️ Known Limitations

* Full face occlusion (e.g., niqab/purdha) may reduce detection accuracy
* Very small or low-quality faces may be ignored
* Performance depends on hardware

---

## 📂 Sample Outputs

* `logs/entries/` → entry images
* `logs/exits/` → exit logs
* `data.db` → database
* `events.log` → system logs

---

## 🎥 Demo

👉 https://www.loom.com/share/40a729ec88764ad38503f75bfa2b86ca

---

## 📢 Hackathon Note

This project is a part of a hackathon run by https://katomaran.com
