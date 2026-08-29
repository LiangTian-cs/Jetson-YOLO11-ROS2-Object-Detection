#!/usr/bin/env python3
"""ROS2 camera node.

Publishes sensor_msgs/Image on /camera/image_raw either from a USB camera
(source_type=usb, default) or from local image files / a directory
(source_type=image) for offline testing. Logs publish FPS statistics.
"""

import glob
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
_STATS_EVERY = 30


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("source_type", "usb")  # usb | image
        self.declare_parameter("device_index", 0)
        self.declare_parameter("device_url", "")
        self.declare_parameter("image_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("loop", True)

        self.source_type = str(self.get_parameter("source_type").value)
        self.image_path = str(self.get_parameter("image_path").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.fps = float(self.get_parameter("fps").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.image_topic = str(self.get_parameter("image_topic").value)

        self.publisher = self.create_publisher(Image, self.image_topic, qos_profile_sensor_data)
        self.bridge = CvBridge()
        self._capture = None
        self._images = []

        if self.source_type == "usb":
            self._setup_usb()
        elif self.source_type == "image":
            self._setup_image()
        else:
            raise ValueError("source_type must be usb or image, got " + repr(self.source_type))

        self._index = 0
        self._published = 0
        self._stats_window_start = time.perf_counter()
        period = 1.0 / self.fps if self.fps > 0 else 0.033
        self.timer = self.create_timer(period, self._publish_frame)
        frames = "live" if self._images is None else len(self._images)
        self.get_logger().info(
            "camera ready: source_type=" + self.source_type
            + " topic=" + self.image_topic
            + " fps=" + format(self.fps, ".1f")
            + " frames=" + str(frames)
        )

    def _setup_usb(self):
        device_index = int(self.get_parameter("device_index").value)
        device_url = str(self.get_parameter("device_url").value)
        source = device_url if device_url else device_index
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError("Could not open USB camera source=" + repr(source))
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        if width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._images = None  # marker: live camera stream

    def _setup_image(self):
        if not self.image_path:
            raise RuntimeError("source_type=image requires the image_path parameter")
        self._images = self._load_images(self.image_path)
        if not self._images:
            raise RuntimeError("No images found at image_path=" + repr(self.image_path))

    @staticmethod
    def _load_images(image_path):
        if os.path.isdir(image_path):
            files = []
            for ext in _IMAGE_EXTS:
                files.extend(glob.glob(os.path.join(image_path, ext)))
            files.sort()
            images = []
            for path in files:
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
            return images
        img = cv2.imread(image_path)
        return [img] if img is not None else []

    def _publish_frame(self):
        if self._capture is not None:
            ok, frame = self._capture.read()
            if not ok:
                self.get_logger().warn("Failed to read a frame from the camera")
                return
        else:
            if self._index >= len(self._images):
                if not self.loop:
                    self.get_logger().info("Finished publishing all local images")
                    self.timer.cancel()
                    return
                self._index = 0
            frame = self._images[self._index]
            self._index += 1

        message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)

        self._published += 1
        if self._published % _STATS_EVERY == 0:
            elapsed = time.perf_counter() - self._stats_window_start
            publish_fps = _STATS_EVERY / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(
                "published=" + str(self._published)
                + " publish_fps=" + format(publish_fps, ".1f")
            )
            self._stats_window_start = time.perf_counter()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if getattr(node, "_capture", None) is not None:
            node._capture.release()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
