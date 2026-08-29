# Jetson YOLO11 ROS2 Object Detection

Desktop object detection coursework project.

## Current Stage

Public hard-case augmentation experiments.

Previous deployment-adaptation experiments showed that deployment-domain
samples could improve performance on previously observed objects, but those
experiments were not suitable as independent generalization evidence.

The next step was therefore to investigate whether publicly sourced hard-case
samples could improve difficult deployment scenarios without using captured
deployment samples.

## Hard-case Experiments

| Experiment | Main Change | Validation mAP50-95 | Deployment Check |
|---|---|---:|---:|
| exp007 | Added 15 public horizontal-bottle hard cases | 0.484 | 1/8 |
| exp008 | Added 48 public deployment hard cases across multiple classes | 0.521 | 0/8 |

exp007 showed that adding a small number of horizontal-bottle examples caused
a substantial degradation in validation and deployment performance.

exp008 increased the diversity of public hard cases, but performance still
remained below the clean v004 model. A bottle-to-mouse class confusion was
also observed during deployment testing.

These experiments indicate that simply adding difficult samples from a
different public-data distribution does not necessarily improve deployment
generalization.

The public hard-case augmentation strategy was therefore rejected.

**Next step:** return to the strongest clean model, exp004, and freeze it for
formal deployment.

> Work in progress.
