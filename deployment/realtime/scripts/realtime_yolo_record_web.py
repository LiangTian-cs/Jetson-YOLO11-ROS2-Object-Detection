#!/usr/bin/env python3
import csv
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import torch
from ultralytics import YOLO

HOST = "0.0.0.0"
PORT = 8080

MODEL_PATH = Path("/home/nvidia/yolo_deploy/models/best_DZH.pt")
OUT_DIR = Path("/home/nvidia/yolo_deploy/results/DZH_realtime_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF = 0.25
IMGSZ = 640

frame_lock = threading.Lock()
state_lock = threading.Lock()

latest_annotated = None
latest_status = {
    "fps": 0.0,
    "inference_ms": 0.0,
    "detections": [],
    "recording": False,
    "video_path": "",
    "csv_path": "",
}

recording = False
writer = None
csv_file = None
csv_writer = None
record_start = None
record_video_path = None
record_csv_path = None
stop_event = threading.Event()


def set_recording(enable):
    global recording, writer, csv_file, csv_writer
    global record_start, record_video_path, record_csv_path

    with state_lock:
        if enable and not recording:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_video_path = OUT_DIR / f"demo_{stamp}.mp4"
            record_csv_path = OUT_DIR / f"demo_{stamp}.csv"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(record_video_path),
                fourcc,
                20.0,
                (640, 480),
            )
            if not writer.isOpened():
                writer = None
                return False, "failed to open video writer"

            csv_file = open(
                record_csv_path,
                "w",
                newline="",
                encoding="utf-8",
            )
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame_index",
                "time_sec",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "inference_ms",
                "end_to_end_fps",
            ])

            record_start = time.perf_counter()
            recording = True
            latest_status["recording"] = True
            latest_status["video_path"] = str(record_video_path)
            latest_status["csv_path"] = str(record_csv_path)
            latest_status["_frame_index"] = 0
            return True, f"recording started: {record_video_path.name}"

        if (not enable) and recording:
            recording = False
            latest_status["recording"] = False

            if writer is not None:
                writer.release()
                writer = None

            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
                csv_file = None

            csv_writer = None
            return True, "recording stopped"

        return True, "no state change"


def detector_worker():
    global latest_annotated

    if not MODEL_PATH.is_file():
        print(f"ERROR: model not found: {MODEL_PATH}", flush=True)
        stop_event.set()
        return

    print("Loading model:", MODEL_PATH, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: cannot open /dev/video0", flush=True)
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        "Camera:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "@",
        cap.get(cv2.CAP_PROP_FPS),
        "FPS",
        flush=True,
    )

    prev = time.perf_counter()
    ema_fps = 0.0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        loop_start = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            device=0,
            rect=False,
            verbose=False,
        )

        r = results[0]
        annotated = r.plot()

        now = time.perf_counter()
        dt = now - prev
        prev = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        ema_fps = inst_fps if ema_fps == 0 else (0.9 * ema_fps + 0.1 * inst_fps)

        inference_ms = float(r.speed.get("inference", 0.0))

        detections = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": cls_id,
                    "class_name": r.names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })

        cv2.putText(
            annotated,
            f"FPS {ema_fps:.1f} | infer {inference_ms:.1f} ms",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        with state_lock:
            latest_status["fps"] = round(ema_fps, 2)
            latest_status["inference_ms"] = round(inference_ms, 2)
            latest_status["detections"] = detections

            if recording and writer is not None:
                writer.write(annotated)
                latest_status["_frame_index"] += 1
                idx = latest_status["_frame_index"]
                t = time.perf_counter() - record_start

                if detections:
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        csv_writer.writerow([
                            idx,
                            f"{t:.3f}",
                            d["class_id"],
                            d["class_name"],
                            f"{d['confidence']:.6f}",
                            f"{x1:.2f}",
                            f"{y1:.2f}",
                            f"{x2:.2f}",
                            f"{y2:.2f}",
                            f"{inference_ms:.3f}",
                            f"{ema_fps:.3f}",
                        ])
                else:
                    csv_writer.writerow([
                        idx,
                        f"{t:.3f}",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"{inference_ms:.3f}",
                        f"{ema_fps:.3f}",
                    ])

        with frame_lock:
            latest_annotated = annotated

        elapsed = time.perf_counter() - loop_start
        if elapsed < 0.001:
            time.sleep(0.001)

    cap.release()
    set_recording(False)


HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DZH Real-time YOLO Demo</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;text-align:center}
.wrap{max-width:1000px;margin:auto;padding:18px}
img{width:min(100%,900px);border:2px solid #555;border-radius:10px;background:#000}
.stats{font-size:22px;margin:12px}
button{font-size:20px;padding:12px 22px;margin:7px;border:0;border-radius:8px;cursor:pointer}
.card{background:#222;padding:12px;border-radius:10px;margin:12px 0}
.rec{font-weight:700}
</style>
</head>
<body>
<div class="wrap">
<h2>DZH YOLO11n Real-time Demo</h2>
<img src="/video_feed">
<div class="stats" id="stats">Loading...</div>
<div>
<button onclick="act('start')">Start recording</button>
<button onclick="act('stop')">Stop recording</button>
</div>
<div class="card" id="record">Not recording</div>
<div class="card" id="detections"></div>
</div>

<script>
async function status(){
  const r=await fetch('/status',{cache:'no-store'});
  const s=await r.json();
  document.getElementById('stats').textContent =
    `FPS ${s.fps.toFixed(1)} | inference ${s.inference_ms.toFixed(1)} ms`;
  document.getElementById('record').textContent =
    s.recording ? `RECORDING → ${s.video_path}` : 'Not recording';
  document.getElementById('detections').innerHTML =
    s.detections.length
    ? s.detections.map(d =>
        `${d.class_name} ${(d.confidence*100).toFixed(1)}%`
      ).join(' &nbsp; | &nbsp; ')
    : 'No detections';
}
async function act(cmd){
  const r=await fetch(`/action?cmd=${cmd}`,{method:'POST'});
  const x=await r.json();
  alert(x.message);
  await status();
}
setInterval(status,500);
status();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = urlparse(self.path)

        if p.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if p.path == "/status":
            with state_lock:
                out = {
                    "fps": latest_status["fps"],
                    "inference_ms": latest_status["inference_ms"],
                    "detections": latest_status["detections"],
                    "recording": latest_status["recording"],
                    "video_path": latest_status["video_path"],
                    "csv_path": latest_status["csv_path"],
                }
            self.send_json(out)
            return

        if p.path == "/video_feed":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            try:
                while not stop_event.is_set():
                    with frame_lock:
                        frame = (
                            None
                            if latest_annotated is None
                            else latest_annotated.copy()
                        )

                    if frame is None:
                        time.sleep(0.05)
                        continue

                    ok, jpg = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                    )
                    if not ok:
                        continue

                    payload = jpg.tobytes()

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(payload)}\r\n\r\n".encode()
                    )
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / 30)

            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/action":
            self.send_error(404)
            return

        q = parse_qs(p.query)
        cmd = q.get("cmd", [""])[0]

        if cmd == "start":
            ok, msg = set_recording(True)
        elif cmd == "stop":
            ok, msg = set_recording(False)
        else:
            ok, msg = False, "unknown action"

        self.send_json({"ok": ok, "message": msg}, 200 if ok else 400)


def main():
    t = threading.Thread(target=detector_worker, daemon=True)
    t.start()

    for _ in range(200):
        if stop_event.is_set():
            raise RuntimeError("detector failed to start")
        with frame_lock:
            if latest_annotated is not None:
                break
        time.sleep(0.05)

    print("======================================")
    print("DZH YOLO11n REAL-TIME DEMO")
    print("Model:", MODEL_PATH)
    print("imgsz=640 rect=False conf=0.25")
    print("Jetson: http://127.0.0.1:8080")
    print("LAN:    http://<JETSON_IP>:8080")
    print("Output:", OUT_DIR)
    print("======================================")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        set_recording(False)
        server.server_close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
