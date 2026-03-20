# AI Face Tracking System

## Overview

A real-time system for detecting, tracking, and recognizing faces using AI models. The system assigns unique IDs to faces and logs their entry and exit events.

## Status

Initial version — core pipeline implemented. Recognition and tracking improvements in progress.

## Tech Stack

* Python
* YOLOv8
* ONNX Runtime (InsightFace-based models)
* OpenCV
* SQLite
* Streamlit (for database visualization)

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models**:
   ```bash
   python download_models.py
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **View Database**:
   ```bash
   streamlit run ui/viewer.py
   ```
