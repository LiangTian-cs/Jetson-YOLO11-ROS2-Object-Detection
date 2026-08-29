# Jetson YOLO11 ROS2 Object Detection

Desktop object detection coursework project.

## Current Stage

Formal deployment model selected and frozen.

After baseline comparison, dataset expansion, deployment-domain adaptation,
and public hard-case experiments, the clean exp004 model was selected for
formal deployment.

## Final Model Selection

| Item | Selection |
|---|---|
| Model | YOLO11n |
| Experiment | exp004 |
| Dataset | frozen v004 |
| Target classes | bottle, mouse, keyboard |
| Training images | 388 |
| Bounding boxes | 609 |

### Validation Performance

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.8095 | 0.6839 | 0.7818 | 0.5861 |

## Selection Rationale

exp004 was retained as the final deployment model because it provided the
strongest clean validation performance without using deployment-domain
captured samples for training.

Later experiments were useful for analysis but were not selected:

- exp005 and exp006 improved performance on previously observed deployment
  examples, but included deployment-domain adaptation data.
- exp007 and exp008 used public hard-case augmentation but reduced overall
  validation and deployment performance.

The final decision was therefore to return to and freeze exp004.

## Model Artifact

The trained `best.pt` binary is not stored directly in the Git source tree.

Its SHA256 checksum is provided under:

`model_selection/exp004/best.pt.sha256`

The final model binary will be provided as a release/submission artifact.

## Next Step

Export and validate the frozen model for deployment, then integrate real-time
Jetson inference, ROS2 result publishing, and TensorRT acceleration.

> Work in progress.
