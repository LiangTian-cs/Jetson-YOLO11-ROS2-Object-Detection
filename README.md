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

- Jetson-specific DZH ROS2 publisher validated on hardware
- TensorRT FP16 engine generated and benchmarked on Jetson
- final ROS2 topic and deployment evidence preserved
- final coursework report and release artifacts


## Final Jetson Deployment Validation

The final DZH ROS2 detector was built and validated on the Jetson Orin platform.

- ROS2 node: `dzh_yolo_camera_node`
- Topic: `/DZH/yolo/detections`
- Message: `yolo_interfaces/msg/DetectionArray`
- Camera: `/dev/video0`, YUYV, 640x480 @ 30 FPS
- Inference configuration: `conf=0.25`, `imgsz=640`, `rect=False`
- Hardware ROS2 topic rate: approximately **12.49 Hz**, exceeding the 5 FPS coursework requirement
- A recorded ROS2 message contained simultaneous `keyboard` and `mouse` detections with class, confidence and bounding-box fields

ROS2 runtime evidence is preserved under `ros2/final_dzh/evidence/`.

A TensorRT FP16 engine was generated directly on the Jetson using TensorRT 10.3.0.

- Engine: `best_DZH_fp16.engine`
- Engine size: approximately 8.5 MB
- SHA256: `c97a0b14083db07d03cb8f4afc52e7f9959d8492a39885a6216b7fbe79d0a14a`
- TensorRT camera benchmark: 25 measured frames after 5 warm-up frames
- Mean inference pipeline latency: **23.45 ms**
- Mean TensorRT inference pipeline throughput: **42.64 FPS**

The TensorRT number above is the inference-pipeline throughput after a camera frame has been acquired; it is not reported as complete camera end-to-end FPS. The separately recorded PyTorch/CUDA real-time camera demonstration achieved approximately **20.55 FPS end-to-end**.

The frozen 20-object acceptance manifest achieved **17/20 = 85%**, satisfying the required accuracy threshold. A supplementary visual audit of all physical objects visible in the captured scenes counted **19/22 = 86.36%**.

TensorRT evidence is preserved under `deployment/tensorrt/evidence/`. The platform-specific `.engine` binary is intentionally excluded from normal Git history and is intended for the final Release/course submission.
