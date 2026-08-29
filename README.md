# Jetson YOLO11 ROS2 Object Detection

Desktop object detection coursework project.

## Current Stage

Baseline model comparison.

Three target classes are used:

- bottle
- mouse
- keyboard

Two initial experiments were evaluated on dataset v003:

| Experiment | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| exp001 | YOLOv8n | 0.7095 | 0.7560 | 0.7732 | 0.5366 |
| exp002 | YOLO11n | 0.8167 | 0.5487 | 0.7064 | 0.5139 |

YOLO11n improved precision but produced substantially lower recall.
Changing the model architecture alone was therefore not sufficient.

**Next step:** improve the training dataset and repeat the comparison.

> Work in progress.
