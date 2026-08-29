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

## Current Status

Model training and model selection are complete.

The selected exp004 model has been exported to ONNX. The next development
stages cover Jetson real-time inference, independent real-object evaluation,
ROS2 integration and TensorRT acceleration.

> Work in progress.
