#!/usr/bin/env python3
"""Benchmark one YOLO artifact on one image without changing model or data."""

import argparse
import json
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_NAMES = {0: "bottle", 1: "mouse", 2: "keyboard"}
BACKENDS = {".pt": "PyTorch", ".onnx": "ONNX", ".engine": "TensorRT"}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def query_nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first_line = result.stdout.strip().splitlines()[0]
    utilization, memory = [float(value.strip()) for value in first_line.split(",")[:2]]
    return {"gpu_utilization_percent": utilization, "memory_used_mib": memory}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help=".pt, .onnx, or .engine artifact")
    parser.add_argument("--source", required=True, help="Representative image")
    parser.add_argument("--output", required=True, help="Benchmark JSON output")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not source.is_file():
        raise FileNotFoundError(source)
    if weights.suffix.lower() not in BACKENDS:
        raise ValueError("weights must be .pt, .onnx, or .engine")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >=0 and iterations must be >=1")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(index): name for index, name in model.names.items()}
    if names != EXPECTED_NAMES:
        raise ValueError(f"Model class contract mismatch: expected {EXPECTED_NAMES}, got {names}")
    predict_args = {
        "source": str(source),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "rect": False,
        "verbose": False,
    }
    for _ in range(args.warmup):
        model.predict(**predict_args)

    wall_latencies = []
    inference_latencies = []
    gpu_samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = model.predict(**predict_args)[0]
        wall_latencies.append((time.perf_counter() - started) * 1000)
        inference_latencies.append(float(result.speed.get("inference", 0.0)))
        gpu_sample = query_nvidia_smi()
        if gpu_sample:
            gpu_samples.append(gpu_sample)

    mean_latency = statistics.fmean(wall_latencies)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": BACKENDS[weights.suffix.lower()],
        "weights": str(weights),
        "source": str(source),
        "class_names": EXPECTED_NAMES,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "device": args.device,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "fps": 1000 / mean_latency if mean_latency else 0.0,
        "latency_ms": {
            "mean_end_to_end": mean_latency,
            "median_end_to_end": statistics.median(wall_latencies),
            "p95_end_to_end": percentile(wall_latencies, 0.95),
            "mean_model_inference": statistics.fmean(inference_latencies),
        },
        "gpu": {
            "sampling": "nvidia-smi after each timed inference; query time excluded",
            "samples": len(gpu_samples),
            "mean_utilization_percent": statistics.fmean(s["gpu_utilization_percent"] for s in gpu_samples) if gpu_samples else None,
            "max_memory_used_mib": max((s["memory_used_mib"] for s in gpu_samples), default=None),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
