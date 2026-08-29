# Deployment backend benchmark procedure

Run this procedure only after a model has been trained and the ONNX/TensorRT
artifacts have been validated for equivalent detections.

## Fixed benchmark contract

- Same representative image for every backend
- Same class contract: `0 bottle`, `1 mouse`, `2 keyboard`
- Batch 1 and input size 640
- Same confidence threshold
- 20 warmup iterations minimum
- 300 measured iterations minimum
- No simultaneous training or unrelated GPU workload
- Record device, software versions, precision mode, power mode, and temperature

## PyTorch

```bash
.venv/bin/python deployment/benchmark_inference.py \
  --weights runs/<experiment_id>/training/weights/best.pt \
  --source /path/to/benchmark_desk.jpg \
  --output runs/<experiment_id>/benchmark_pytorch.json \
  --imgsz 640 --warmup 20 --iterations 300 --device 0
```

## ONNX

```bash
.venv/bin/python deployment/benchmark_inference.py \
  --weights runs/<experiment_id>/deployment/best.onnx \
  --source /path/to/benchmark_desk.jpg \
  --output runs/<experiment_id>/benchmark_onnx.json \
  --imgsz 640 --warmup 20 --iterations 300 --device 0
```

## TensorRT on Jetson

```bash
python3 deployment/benchmark_inference.py \
  --weights runs/<experiment_id>/deployment/best_fp16.engine \
  --source /path/to/benchmark_desk.jpg \
  --output runs/<experiment_id>/benchmark_tensorrt.json \
  --imgsz 640 --warmup 20 --iterations 300 --device 0
```

## Recorded metrics

- End-to-end mean, median, and P95 latency
- Model-reported inference latency
- FPS derived from mean end-to-end latency
- Mean GPU utilization when `nvidia-smi` is available
- Maximum observed GPU memory use

Jetson commonly lacks `nvidia-smi`; collect `tegrastats` separately and enter
GPU, memory, temperature, and power values in `testing/jetson_test_record.md`.
Do not compare FPS values collected with different images, input sizes, warmup,
iteration counts, or active background workloads.
