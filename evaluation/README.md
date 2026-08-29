# Jetson Real-Object Acceptance Evaluation

This directory contains the frozen real-object evaluation evidence collected
on the NVIDIA Jetson using the selected exp004 YOLO11n model.

## Configuration

- Model: exp004 YOLO11n
- Artifact: `best_DZH.pt`
- Confidence threshold: `0.25`
- Image size: `640`
- Preprocessing: `rect=False`
- Device: NVIDIA Jetson CUDA
- Number of scenes: 10

## Predefined 20-Object Acceptance Test

The original frozen manifest defines 20 target objects.

| Class | Correct | Total | Accuracy |
|---|---:|---:|---:|
| bottle | 5 | 7 | 71.43% |
| mouse | 6 | 6 | 100.00% |
| keyboard | 6 | 7 | 85.71% |
| **Overall** | **17** | **20** | **85.00%** |

The coursework requirement is at least 80% correct.

**Result: PASS**

The machine-generated evaluation files are preserved without modification.

## Actual-Scene Visual Audit

The automatic evaluator counted detections outside the predefined manifest as
unmatched or false-positive boxes.

Manual review of the saved annotated photographs showed that two of these
mouse detections corresponded to real additional physical objects visible in
the scenes but not listed in the original 20-object manifest.

The photographed scenes therefore contained 22 relevant objects in total.

| Class | Correct | Actual Objects | Accuracy |
|---|---:|---:|---:|
| bottle | 5 | 7 | 71.43% |
| mouse | 8 | 8 | 100.00% |
| keyboard | 6 | 7 | 85.71% |
| **Overall** | **19** | **22** | **86.36%** |

After visual review, one unmatched detection remained a genuine false
positive.

The official coursework acceptance result remains **17/20 = 85.00%** because
the predefined evaluation protocol contains 20 objects.

The **19/22 = 86.36%** result is reported separately as an actual-scene visual
audit.

## Runtime

The original evaluation recorded:

| Stage | Mean Time |
|---|---:|
| Preprocess | 4.77 ms |
| Inference | 29.33 ms |
| Postprocess | 4.98 ms |

Inference-only throughput was approximately **34.10 FPS**.

These values correspond to the PyTorch/CUDA evaluation path and are separate
from the later TensorRT benchmark.

## Manifest Misses

Three predefined targets were missed by the automatic evaluation:

- S04: bottle
- S06: keyboard
- S10: bottle

## Evidence

- `acceptance/capture_manifest.csv` - predefined 20-object manifest
- `acceptance/FROZEN.sha256` - frozen capture checksums
- `acceptance/object_results.csv` - original automatic object scoring
- `acceptance/scene_results.csv` - original scene detections and timing
- `acceptance/summary.json` - original machine-generated summary
- `acceptance/annotated/` - saved annotated photographs
- `visual_audit.csv` - supplementary actual-scene visual review

No machine-generated evaluation file has been manually edited.
