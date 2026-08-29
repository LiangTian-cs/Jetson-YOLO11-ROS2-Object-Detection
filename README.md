# Jetson YOLO11 ROS2 Object Detection

Real-time desktop object detection using YOLO11, NVIDIA Jetson and ROS2.

## Target Classes

- bottle
- mouse
- keyboard

## Selected Model

The final deployment model is YOLO11n from experiment `exp004`,
trained on the frozen `v004` dataset.

| Metric | Result |
|---|---:|
| Precision | 0.8095 |
| Recall | 0.6839 |
| mAP50 | 0.7818 |
| mAP50-95 | 0.5861 |

Dataset v004 contains 388 images and 609 annotated bounding boxes.

Later adaptation and public hard-case experiments are preserved in Git
history. They were not selected for formal deployment.

## Deployment Pipeline

```text
YOLO11n exp004
      |
      v
  PyTorch .pt
      |
      v
     ONNX
      |
      v
 NVIDIA Jetson
      |
      +--> Real-time USB camera inference
      |
      +--> ROS2 detection publishing
      |
      v
 TensorRT FP16
```

## Deployment Contract

- Input image size: `640`
- Rectangular preprocessing: `rect=False`
- Default confidence threshold: `0.25`
- Classes: `bottle`, `mouse`, `keyboard`

## Repository Structure

- `model_selection/` - final model-selection evidence
- `dataset_metadata/` - frozen dataset metadata
- `deployment/` - export, validation and deployment utilities

Earlier training experiments are preserved through Git commit history instead
of being duplicated in the latest repository tree.

## Binary Artifacts

Large binary artifacts are intentionally excluded from normal Git history.

The final coursework release/submission will provide:

- trained PyTorch model (`.pt`)
- exported ONNX model (`.onnx`)
- Jetson-generated TensorRT FP16 engine (`.engine`)
- real-time result video
- final dataset package

## Jetson Real-Object Evaluation

The frozen predefined acceptance test achieved:

| Evaluation | Correct / Total | Accuracy |
|---|---:|---:|
| Predefined 20-object manifest | 17 / 20 | 85.00% |
| Actual-scene visual audit | 19 / 22 | 86.36% |

The official coursework score is **17/20 = 85.00%**, exceeding the required
80% threshold.

Manual inspection additionally identified two real mouse objects that were
visible in the photographs but absent from the predefined manifest. Detailed
results and annotated evidence are stored under `evaluation/`.

The measured PyTorch/CUDA inference-only throughput during this evaluation was
approximately **34.10 FPS**.

## Real-Time Jetson Camera Demo

A recorded Jetson PyTorch/CUDA camera run processed
**1742 frames over 87.832 seconds**.

The measured mean end-to-end processing rate was
**20.55 FPS**, exceeding the coursework requirement
of 5 FPS.

The browser-based recording programs and runtime evidence are available under
`deployment/realtime/`. Large MP4 files are kept outside normal Git history
and are intended for the final Release/coursework package.

## ROS2 Integration

A pre-deployment ROS2 prototype is preserved under `ros2/prototype/`.

The prototype implements a three-node camera, YOLO detector and visualization
pipeline using `sensor_msgs/Image` and `vision_msgs/Detection2DArray`.

It represents an earlier development stage and is intentionally kept separate
from the final Jetson-specific ROS2 interface.

## Current Status

Completed:

- dataset construction and dataset versioning
- YOLOv8n / YOLO11n experiment comparison
- final exp004 YOLO11n model selection
- ONNX export and validation utilities
- independent real-object acceptance evaluation
- Jetson PyTorch/CUDA real-time camera demonstration
- preservation of the pre-deployment ROS2 prototype

Remaining deployment stages:

- recover and preserve the final Jetson-specific ROS2 publisher
- TensorRT FP16 engine generation and benchmarking on Jetson
- final ROS2 topic evidence and deployment documentation
- final coursework report and release artifacts
