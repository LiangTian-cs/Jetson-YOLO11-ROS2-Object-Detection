# Jetson YOLO11 ROS2 Object Detection

Desktop object detection coursework project.

## Current Stage

Dataset improvement and model comparison.

The initial v003 dataset was expanded after analysing errors from the
baseline experiments.

### Dataset Development

| Dataset | Images | Bounding Boxes |
|---|---:|---:|
| v003 | 140 | 249 |
| v004 | 388 | 609 |

The expanded v004 dataset was frozen before the next model comparison.

## v004 Experiments

| Experiment | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| exp003 | YOLOv8n | 0.7754 | 0.6702 | 0.7565 | 0.5352 |
| exp004 | YOLO11n | 0.8095 | 0.6839 | 0.7818 | 0.5861 |

YOLO11n on the expanded dataset achieved the strongest overall validation
performance in this stage.

**Current best model:** exp004 - YOLO11n trained on frozen dataset v004.

**Next step:** investigate deployment-domain behaviour and difficult
real-world examples.

> Work in progress.
