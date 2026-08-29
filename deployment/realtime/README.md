# Jetson Real-Time Camera Demo

This directory preserves the real-time Jetson camera programs and performance
evidence used during deployment testing of the selected exp004 YOLO11n model.

## Programs

### `scripts/realtime_yolo_record_web.py`

Initial browser-based real-time detector and recorder.

It provides:

- USB camera inference
- bounding-box visualization
- class and confidence display
- MJPEG browser preview
- video recording
- CSV detection logging

### `scripts/realtime_yolo_record_cases_web.py`

Extended version used during the later deployment test.

It additionally records:

- preprocess latency
- inference latency
- postprocess latency
- end-to-end FPS
- optional success/error snapshots and metadata

Both files are preserved from the actual Jetson deployment workflow.

## Deployment Configuration

- Model: `best_DZH.pt` from exp004
- Camera: `/dev/video0`
- Capture backend: V4L2
- Camera mode: YUYV, 640x480, requested 30 FPS
- YOLO image size: 640
- Confidence threshold: 0.25
- Preprocessing: `rect=False`
- Inference device: CUDA device 0
- Browser service: port 8080

The original scripts use the Jetson deployment path
`/home/nvidia/yolo_deploy/models/best_DZH.pt`.

## Recorded Demo

Selected run:

`demo_20260828_175009`

| Metric | Value |
|---|---:|
| Unique processed frames | 1742 |
| Duration | 87.832 s |
| Recorded frame rate | 19.833 FPS |
| Mean end-to-end FPS | 20.548 |
| Mean preprocess | 2.622 ms |
| Mean inference | 25.783 ms |
| P95 inference | 26.125 ms |
| Mean postprocess | 3.029 ms |

Course requirement: at least 5 FPS.

**Result: PASS**

These measurements correspond to the PyTorch/Ultralytics CUDA real-time
pipeline. TensorRT measurements are reported separately.

## Case-Capture Index

The enhanced program supports manually saving successful and error examples.

The archived `case_index.csv` contains **0 manually saved case
rows**. The file is retained as original evidence and is not interpreted as
containing cases when it only contains its header.

Typical success/failure examples for the coursework are instead preserved in
the independent acceptance evaluation under `evaluation/acceptance/annotated/`.

## Video Artifacts

The MP4 files are intentionally excluded from normal Git history because they
are large binary artifacts.

Their SHA256 checksums are stored in:

`evidence/video_checksums.sha256`

The final video files are intended for the final coursework package or GitHub
Release.

## Evidence Files

- `evidence/demo_20260828_175009.csv` - real per-frame/detection runtime log
- `evidence/demo_20260828_175009.summary.json` - summary computed per unique frame
- `evidence/case_index.csv` - case-capture index from the enhanced recorder
- `evidence/video_checksums.sha256` - checksums of the archived MP4 files
