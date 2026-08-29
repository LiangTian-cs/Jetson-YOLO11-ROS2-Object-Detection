# Deployment

Deployment utilities for the frozen exp004 YOLO11n detector.

## Model Contract

- Classes: `bottle`, `mouse`, `keyboard`
- Input image size: `640`
- Rectangular preprocessing: `False`
- Default confidence threshold: `0.25`

The same preprocessing contract is maintained across model validation,
benchmarking and deployment inference.

## Utilities

- `scripts/export_onnx.py` - export the selected PyTorch checkpoint to ONNX
- `scripts/test_onnx.py` - compare ONNX output against the PyTorch reference
- `scripts/model_inference.py` - unified `.pt`, `.onnx` and `.engine` inference
- `scripts/benchmark_inference.py` - repeatable inference benchmark
- `scripts/check_environment.sh` - deployment environment inspection

## Artifacts

Large binary model files are intentionally excluded from normal Git history.

`artifacts/` contains metadata and checksums for the deployment artifacts.

The final submission/release will contain:

- PyTorch `.pt`
- ONNX `.onnx`
- Jetson-generated TensorRT FP16 `.engine`

## Next Stage

Validate the frozen model on NVIDIA Jetson, perform real-time camera
inference, publish detections through ROS2, and benchmark TensorRT FP16.
