# Jetson YOLO11 ROS2 Object Detection

Desktop object detection coursework project.

## Current Stage

Deployment-domain adaptation experiments.

After selecting exp004 as the strongest clean model in the previous stage,
real camera testing revealed a deployment-domain gap, particularly for some
bottle appearances and viewpoints.

Two exploratory adaptation experiments were therefore conducted.

## Adaptation Experiments

| Experiment | Model | Main Change | Deployment Observation |
|---|---|---|---|
| exp005 | YOLO11n | Added deployment-domain captured samples | Strong improvement on previously observed Mac examples |
| exp006 | YOLO11n | Added additional diversity to the adaptation data | Maintained strong performance on the same deployment check |

Both experiments achieved 8/8 on the previously observed Mac bottle check.

However, these results are **not treated as independent generalization
evidence**, because deployment-domain captured samples were included in the
adaptation training process.

The experiments demonstrate that deployment-specific data can rapidly
improve same-domain performance, but they also introduce a risk of instance
overfitting and data leakage.

For this reason, exp005 and exp006 are treated as exploratory adaptation
experiments rather than final deployment-model candidates.

**Next step:** investigate whether public hard-case data can improve difficult
poses without relying on deployment-domain training samples.

> Work in progress.
