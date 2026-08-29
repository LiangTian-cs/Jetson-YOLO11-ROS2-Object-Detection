#!/usr/bin/env python3
"""ROS2 display node.

Subscribes to /detections and visualizes bounding boxes, class labels and confidence
scores. When the corresponding image topic (/camera/image_raw) is available the detections
are overlaid on the frame and shown in a live OpenCV window; otherwise the detections are
printed to the log. display_mode=log is available for headless runs.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

# BGR colors per class for consistent visualization.
COLORS = {
    "bottle": (0, 255, 0),
    "mouse": (0, 255, 255),
    "keyboard": (255, 0, 0),
}
WINDOW_NAME = "ros2_object_detection"


def _bbox_center(bbox):
    """Return (cx, cy) of a BoundingBox2D across older/newer vision_msgs layouts."""
    if hasattr(bbox.center, "position"):
        return bbox.center.position.x, bbox.center.position.y
    return bbox.center.x, bbox.center.y


class DisplayNode(Node):
    def __init__(self):
        super().__init__("display_node")
        self.declare_parameter("detection_topic", "/detections")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("subscribe_image", True)
        self.declare_parameter("display_mode", "window")  # window | log

        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.subscribe_image = bool(self.get_parameter("subscribe_image").value)
        self.display_mode = str(self.get_parameter("display_mode").value)
        if self.display_mode not in ("window", "log"):
            raise ValueError("display_mode must be window or log, got " + repr(self.display_mode))

        self.bridge = CvBridge()
        self._latest_image = None
        self._window_ok = None  # None = not tried yet

        self.subscription = self.create_subscription(
            Detection2DArray, self.detection_topic, self._detections_callback, 10
        )
        if self.subscribe_image:
            self.image_subscription = self.create_subscription(
                Image, self.image_topic, self._image_callback, qos_profile_sensor_data
            )
        self.get_logger().info(
            "display ready: subscribe=" + self.detection_topic + " mode=" + self.display_mode
            + " subscribe_image=" + str(self.subscribe_image)
        )

    def _image_callback(self, message):
        self._latest_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def _show_overlay(self, overlay):
        if self._window_ok is False:
            return False
        try:
            cv2.imshow(WINDOW_NAME, overlay)
            cv2.waitKey(1)
            self._window_ok = True
            return True
        except cv2.error as exc:
            self._window_ok = False
            self.get_logger().warn("OpenCV window unavailable (" + str(exc) + "); falling back to log mode")
            return False

    def _detections_callback(self, message):
        frame = self._latest_image
        overlay = frame.copy() if frame is not None else None
        lines = []
        for detection in message.detections:
            hyp = detection.results[0] if detection.results else None
            class_id = hyp.hypothesis.class_id if hyp else "?"
            score = hyp.hypothesis.score if hyp else 0.0
            center_x, center_y = _bbox_center(detection.bbox)
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y
            x1 = int(center_x - size_x / 2.0)
            y1 = int(center_y - size_y / 2.0)
            x2 = int(center_x + size_x / 2.0)
            y2 = int(center_y + size_y / 2.0)
            color = COLORS.get(class_id, (0, 0, 255))
            if overlay is not None:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = class_id + " " + format(score, ".2f")
                cv2.putText(overlay, label, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            lines.append("class=" + class_id + " conf=" + format(score, ".2f")
                         + " bbox=(" + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2) + ")"
                         + " size=(" + format(size_x, ".1f") + "x" + format(size_y, ".1f") + ")")

        self.get_logger().info("detections=" + str(len(message.detections)))
        for line in lines:
            self.get_logger().info(line)

        if self.display_mode == "window":
            if overlay is not None:
                self._show_overlay(overlay)
            else:
                self.get_logger().warn("No image received yet; running in text-only mode")


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
