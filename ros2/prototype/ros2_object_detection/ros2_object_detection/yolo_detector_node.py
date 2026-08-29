#!/usr/bin/env python3
"""ROS2 YOLO detector node.

Subscribes to /camera/image_raw (sensor_msgs/Image), runs YOLO inference with the
trained YOLO11n weights, and publishes vision_msgs/Detection2DArray on /detections.
Logs inference time, processing FPS, and (optionally) per-detection details.
"""

import statistics
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

EXPECTED_NAMES = {0: "bottle", 1: "mouse", 2: "keyboard"}
STATS_EVERY = 30  # log summary every N processed frames


def _bbox_center(bbox):
    if hasattr(bbox.center, "position"):
        return bbox.center.position.x, bbox.center.position.y
    return bbox.center.x, bbox.center.y


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")
        self.declare_parameter("weights", "best.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detection_topic", "/detections")
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "0")
        self.declare_parameter("log_detections", False)

        self.weights = str(self.get_parameter("weights").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.log_detections = bool(self.get_parameter("log_detections").value)

        from ultralytics import YOLO

        self.model = YOLO(self.weights)
        model_names = {int(index): name for index, name in self.model.names.items()}
        if model_names != EXPECTED_NAMES:
            raise ValueError(
                "Model class contract mismatch: expected " + repr(EXPECTED_NAMES)
                + ", got " + repr(model_names)
            )
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Detection2DArray, self.detection_topic, 10)
        self.subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos_profile_sensor_data
        )

        self.frame_count = 0
        self.inference_ms = []
        self.stats_window_start = time.perf_counter()
        self.get_logger().info(
            "detector ready: weights=" + self.weights
            + " classes=" + repr(list(EXPECTED_NAMES.values()))
            + " subscribe=" + self.image_topic
            + " publish=" + self.detection_topic
            + " conf=" + str(self.confidence)
            + " imgsz=" + str(self.imgsz)
            + " device=" + self.device
        )

    def image_callback(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

        started = time.perf_counter()
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            rect=False,
            device=self.device,
            verbose=False,
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        self.inference_ms.append(inference_ms)

        output = Detection2DArray()
        output.header = message.header
        if result.boxes is not None:
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                x1, y1, x2, y2 = xyxy
                detection = Detection2D()
                detection.header = message.header
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if hasattr(detection.bbox.center, "position"):
                    detection.bbox.center.position.x = float(cx)
                    detection.bbox.center.position.y = float(cy)
                else:
                    detection.bbox.center.x = float(cx)
                    detection.bbox.center.y = float(cy)
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(result.names[int(class_id)])
                hypothesis.hypothesis.score = float(confidence)
                detection.results.append(hypothesis)
                output.detections.append(detection)

                if self.log_detections:
                    self.get_logger().info(
                        "detection class=" + str(result.names[int(class_id)])
                        + " conf=" + format(float(confidence), ".3f")
                        + " bbox=(" + format(x1, ".1f") + "," + format(y1, ".1f")
                        + "," + format(x2, ".1f") + "," + format(y2, ".1f") + ")"
                    )
        self.publisher.publish(output)

        self.frame_count += 1
        if self.frame_count % STATS_EVERY == 0:
            elapsed = time.perf_counter() - self.stats_window_start
            fps = STATS_EVERY / elapsed if elapsed > 0 else 0.0
            mean_ms = statistics.fmean(self.inference_ms) if self.inference_ms else 0.0
            self.get_logger().info(
                "published=" + str(len(output.detections))
                + " infer_ms=" + format(mean_ms, ".1f")
                + " fps=" + format(fps, ".1f")
                + " frames=" + str(self.frame_count)
            )
            self.inference_ms = []
            self.stats_window_start = time.perf_counter()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
