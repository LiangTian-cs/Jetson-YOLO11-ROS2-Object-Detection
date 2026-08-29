#!/usr/bin/env python3
"""Unified model deployment inference tool.

Loads a PyTorch (.pt), ONNX (.onnx), or TensorRT (.engine) artifact with the same
configuration (model path / confidence / imgsz), runs inference on images, and
prints/records bbox, class, and confidence plus timing.
"""

import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import cv2

EXPECTED_NAMES = {0: "bottle", 1: "mouse", 2: "keyboard"}
SUPPORTED = {".pt", ".onnx", ".engine"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help=".pt / .onnx / .engine artifact")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default="", help="optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    source = Path(args.source).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() not in SUPPORTED:
        raise ValueError("model must be .pt, .onnx, or .engine")
    if source.is_dir():
        images = sorted(glob.glob(str(source) + "/*.jpg") + glob.glob(str(source) + "/*.png"))
    elif source.is_file():
        images = [str(source)]
    else:
        raise FileNotFoundError(source)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError("no images found")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = {int(index): name for index, name in model.names.items()}
    if names != EXPECTED_NAMES:
        raise ValueError("class contract mismatch: expected " + repr(EXPECTED_NAMES) + ", got " + repr(names))

    rows = []
    total_dets = 0
    for path in images:
        frame = cv2.imread(path)
        if frame is None:
            continue
        started = time.perf_counter()
        result = model.predict(
            source=frame,
            conf=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            rect=False,
            verbose=False,
        )[0]
        wall_ms = (time.perf_counter() - started) * 1000.0
        infer_ms = float(result.speed.get("inference", 0.0)) if result.speed else wall_ms
        dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                dets.append({
                    "class_id": int(cls),
                    "class_name": result.names[int(cls)],
                    "confidence": float(conf),
                    "xyxy": [round(float(v), 2) for v in xyxy],
                })
        total_dets += len(dets)
        rows.append({
            "image": path,
            "detections": dets,
            "wall_ms": round(wall_ms, 2),
            "infer_ms": round(infer_ms, 2),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    wall = [r["wall_ms"] for r in rows]
    infer = [r["infer_ms"] for r in rows]
    summary = {
        "model": str(model_path),
        "backend": model_path.suffix.lower().lstrip("."),
        "source": str(source),
        "images": len(rows),
        "detections": total_dets,
        "confidence": args.confidence,
        "imgsz": args.imgsz,
        "device": args.device,
        "latency_ms": {
            "mean_wall": round(statistics.fmean(wall), 2) if wall else None,
            "mean_model": round(statistics.fmean(infer), 2) if infer else None,
            "p95_wall": round(sorted(wall)[min(len(wall) - 1, int((len(wall) - 1) * 0.95))], 2) if wall else None,
        },
        "fps": round(1000.0 / statistics.fmean(wall), 2) if wall else None,
        "per_image": rows,
    }
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "per_image"}, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved:", args.output)


if __name__ == "__main__":
    main()
