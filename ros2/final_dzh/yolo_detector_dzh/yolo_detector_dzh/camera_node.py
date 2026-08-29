#!/usr/bin/env python3
"""DZH Jetson camera YOLO detector.

Reads a USB camera directly with OpenCV/V4L2, runs Ultralytics YOLO inference,
and publishes yolo_interfaces/DetectionArray.

This implementation is reconstructed from the verified Jetson deployment
contract. It is not claimed to be a byte-for-byte copy of the original
Jetson source file.
"""

import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from yolo_interfaces.msg import Detection, DetectionArray


EXPECTED_NAMES = {
    0: "bottle",
    1: "mouse",
    2: "keyboard",
}


def normalize_names(names):
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def clip_int(value, low, high):
    return max(low, min(high, int(round(float(value)))))


class DzhYoloCameraNode(Node):
    def __init__(self):
        super().__init__("dzh_yolo_camera_node")

        self.declare_parameter(
            "model_path",
            "/home/nvidia/yolo_deploy/models/best_DZH.pt",
        )
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "0")
        self.declare_parameter("topic", "/DZH/yolo/detections")
        self.declare_parameter("frame_id", "dzh_camera")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("log_every", 30)

        self.model_path = str(self.get_parameter("model_path").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.topic = str(self.get_parameter("topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.log_every = max(1, int(self.get_parameter("log_every").value))

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_file}")

        from ultralytics import YOLO

        self.get_logger().info(f"Loading YOLO model: {model_file}")
        self.model = YOLO(str(model_file))

        model_names = normalize_names(self.model.names)
        if model_names != EXPECTED_NAMES:
            raise ValueError(
                "Model class contract mismatch: "
                f"expected={EXPECTED_NAMES}, got={model_names}"
            )

        self.publisher = self.create_publisher(
            DetectionArray,
            self.topic,
            10,
        )

        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_V4L2,
        )
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index} "
                f"(/dev/video{self.camera_index})"
            )

        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"YUYV"),
        )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.get_logger().info(
            "Camera ready: "
            f"/dev/video{self.camera_index} "
            f"{actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )
        self.get_logger().info(
            "YOLO contract: "
            f"conf={self.confidence} imgsz={self.imgsz} "
            f"rect=False device={self.device}"
        )
        self.get_logger().info(f"Publishing detections: {self.topic}")

        self.frame_count = 0
        self.ema_fps = 0.0
        self.last_frame_time = None
        self.read_failures = 0

        # The callback runs serially under the normal single-threaded executor.
        # The camera/inference path therefore determines the actual processing rate.
        self.timer = self.create_timer(0.001, self.process_frame)

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.read_failures += 1
            if self.read_failures % 30 == 1:
                self.get_logger().warning("Camera frame read failed")
            return

        self.read_failures = 0
        started = time.perf_counter()

        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            rect=False,
            verbose=False,
        )[0]

        wall_ms = (time.perf_counter() - started) * 1000.0

        now = time.perf_counter()
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
            if dt > 0:
                instantaneous_fps = 1.0 / dt
                if self.ema_fps == 0.0:
                    self.ema_fps = instantaneous_fps
                else:
                    self.ema_fps = (
                        0.9 * self.ema_fps
                        + 0.1 * instantaneous_fps
                    )
        self.last_frame_time = now

        height, width = frame.shape[:2]

        output = DetectionArray()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.frame_id
        output.image_width = int(width)
        output.image_height = int(height)
        output.fps = float(self.ema_fps)

        if result.boxes is not None:
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                class_id = int(class_id)
                if class_id not in EXPECTED_NAMES:
                    self.get_logger().warning(
                        f"Unexpected class id: {class_id}"
                    )
                    continue

                x1, y1, x2, y2 = xyxy

                detection = Detection()
                detection.class_id = class_id
                detection.class_name = EXPECTED_NAMES[class_id]
                detection.confidence = float(confidence)
                detection.xmin = clip_int(x1, 0, width)
                detection.ymin = clip_int(y1, 0, height)
                detection.xmax = clip_int(x2, 0, width)
                detection.ymax = clip_int(y2, 0, height)

                output.detections.append(detection)

        self.publisher.publish(output)
        self.frame_count += 1

        if self.frame_count % self.log_every == 0:
            speed = result.speed or {}
            preprocess_ms = float(speed.get("preprocess", 0.0))
            inference_ms = float(speed.get("inference", 0.0))
            postprocess_ms = float(speed.get("postprocess", 0.0))

            self.get_logger().info(
                f"frames={self.frame_count} "
                f"detections={len(output.detections)} "
                f"fps={self.ema_fps:.2f} "
                f"pre={preprocess_ms:.2f}ms "
                f"infer={inference_ms:.2f}ms "
                f"post={postprocess_ms:.2f}ms "
                f"wall={wall_ms:.2f}ms"
            )

    def destroy_node(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DzhYoloCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
