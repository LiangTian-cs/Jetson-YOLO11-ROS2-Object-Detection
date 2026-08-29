#!/usr/bin/env python3
"""Export the three-class Ultralytics YOLO checkpoint to ONNX."""

import argparse
from pathlib import Path

EXPECTED_NAMES = {0: "bottle", 1: "mouse", 2: "keyboard"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.model).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.half and args.device == "cpu":
        raise ValueError("--half requires a CUDA device")

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    model_names = {int(index): name for index, name in model.names.items()}
    if model_names != EXPECTED_NAMES:
        raise ValueError(f"Checkpoint class contract mismatch: {model_names}")
    output = model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        device=args.device, dynamic=args.dynamic, simplify=args.simplify,
        half=args.half,
    )
    print(f"ONNX export: {output}")


if __name__ == "__main__":
    main()
