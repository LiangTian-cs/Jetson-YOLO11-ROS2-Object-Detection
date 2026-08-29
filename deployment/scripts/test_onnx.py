#!/usr/bin/env python3
"""ONNX Runtime deployment check for the three-class YOLO11n model.

Runs the exported ONNX on image input, parses its raw output with the same
preprocessing/NMS pipeline as Ultralytics, and compares bbox / class /
confidence against the PyTorch checkpoint on the same images.
"""

import argparse
import glob
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

EXPECTED_NAMES = {0: "bottle", 1: "mouse", 2: "keyboard"}
MATCH_IOU = 0.5
CONF_TOL = 0.05


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Equivalent to Ultralytics LetterBox with auto=False."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_onnx_output(raw, r, pad, conf_thr, iou_thr):
    """Use Ultralytics exact NMS on the raw ONNX tensor, then inverse-letterbox."""
    import torch
    from ultralytics.utils.nms import non_max_suppression

    prediction = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    processed = non_max_suppression(
        prediction,
        conf_thres=conf_thr,
        iou_thres=iou_thr,
        agnostic=False,
        max_det=300,
    )[0]
    dets = []
    if processed is not None and len(processed):
        for values in processed.tolist():
            x1, y1, x2, y2, conf, cls = values
            dets.append({
                "xyxy": [(x1 - pad[0]) / r, (y1 - pad[1]) / r,
                         (x2 - pad[0]) / r, (y2 - pad[1]) / r],
                "class_id": int(cls),
                "class_name": EXPECTED_NAMES[int(cls)],
                "confidence": float(conf),
            })
    return dets


def parse_ultralytics_result(result):
    dets = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = result.names
        for xyxy, cls, conf in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            dets.append({
                "xyxy": [float(v) for v in xyxy],
                "class_id": int(cls),
                "class_name": names[int(cls)],
                "confidence": float(conf),
            })
    return dets


def match_detections(pred, ref):
    matched = []
    ref_used = [False] * len(ref)
    for j, r in enumerate(ref):
        best = None
        best_iou = 0.0
        for p in pred:
            if r["class_id"] != p["class_id"]:
                continue
            value = iou(r["xyxy"], p["xyxy"])
            if value > best_iou:
                best_iou = value
                best = p
        if best is not None and best_iou >= MATCH_IOU:
            matched.append({
                "ref": r,
                "pred": best,
                "iou": best_iou,
                "conf_diff": abs(r["confidence"] - best["confidence"]),
            })
            ref_used[j] = True
    unmatched_ref = [r for j, r in enumerate(ref) if not ref_used[j]]
    return matched, unmatched_ref


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--onnx", required=True, help="Exported ONNX model")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    pt_path = Path(args.pt).resolve()
    onnx_path = Path(args.onnx).resolve()
    source = Path(args.source).resolve()
    assert pt_path.is_file(), pt_path
    assert onnx_path.is_file(), onnx_path

    import onnxruntime as ort
    from ultralytics import YOLO

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    expected_input = session.get_inputs()[0].shape
    if args.imgsz not in expected_input:
        raise ValueError("onnx input shape %s does not match imgsz=%d" % (expected_input, args.imgsz))

    model = YOLO(str(pt_path))
    names = {int(i): n for i, n in model.names.items()}
    if names != EXPECTED_NAMES:
        raise ValueError("class contract mismatch: %s" % (names,))

    if source.is_dir():
        images = sorted(glob.glob(str(source) + "/*.jpg") + glob.glob(str(source) + "/*.png"))
    else:
        images = [str(source)]
    if args.max_images:
        images = images[: args.max_images]
    assert images, "no images found in source"

    all_matched = []
    all_unmatched_ref = []
    unmatched_pred_total = 0
    summary_rows = []

    for path in images:
        frame = cv2.imread(path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # ONNX path (letterbox + ultralytics NMS + inverse letterbox)
        t0 = time.perf_counter()
        padded, r, pad = letterbox(frame, (args.imgsz, args.imgsz))
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        raw = session.run(None, {input_name: blob})[0]
        onnx_dets = parse_onnx_output(raw, r, pad, args.conf, args.iou)
        onnx_ms = (time.perf_counter() - t0) * 1000.0

        # PyTorch path
        result = model.predict(
            source=frame, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, rect=False, verbose=False,
        )[0]
        torch_dets = parse_ultralytics_result(result)

        matched, unmatched_ref = match_detections(onnx_dets, torch_dets)
        all_matched.extend(matched)
        all_unmatched_ref.extend(unmatched_ref)
        unmatched_pred_total += len(onnx_dets) - len(matched)

        summary_rows.append({
            "image": os.path.basename(path),
            "size": "%dx%d" % (w, h),
            "torch_dets": len(torch_dets),
            "onnx_dets": len(onnx_dets),
            "matched": len(matched),
            "unmatched_ref": len(unmatched_ref),
            "onnx_ms": round(onnx_ms, 2),
        })
        print(json.dumps(summary_rows[-1], ensure_ascii=False))

    ious = [m["iou"] for m in all_matched]
    conf_diffs = [m["conf_diff"] for m in all_matched]
    torch_total = sum(r["torch_dets"] for r in summary_rows)
    report = {
        "pt": str(pt_path),
        "onnx": str(onnx_path),
        "source": str(source),
        "images": len(summary_rows),
        "torch_detections": torch_total,
        "onnx_detections": sum(r["onnx_dets"] for r in summary_rows),
        "matched": len(all_matched),
        "unmatched_ref": len(all_unmatched_ref),
        "unmatched_pred": unmatched_pred_total,
        "match_rate": round(len(all_matched) / max(1, torch_total), 4),
        "bbox_iou": {
            "mean": round(statistics.fmean(ious), 4) if ious else None,
            "min": round(min(ious), 4) if ious else None,
        },
        "conf_diff": {
            "mean": round(statistics.fmean(conf_diffs), 4) if conf_diffs else None,
            "max": round(max(conf_diffs), 4) if conf_diffs else None,
            "tolerance": CONF_TOL,
            "exceeded": sum(1 for d in conf_diffs if d > CONF_TOL),
        },
        "per_image": summary_rows,
    }
    print("SUMMARY " + json.dumps({k: v for k, v in report.items() if k != "per_image"}, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("saved:", args.json_out)


if __name__ == "__main__":
    main()
