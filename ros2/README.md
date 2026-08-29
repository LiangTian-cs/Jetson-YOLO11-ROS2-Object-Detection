# ROS2 Integration

This directory records the ROS2 development path for the object-detection
course project.

## Prototype

`prototype/ros2_object_detection/` preserves the pre-deployment ROS2
pipeline developed before the final Jetson-specific integration.

The prototype consists of three ROS2 nodes:

1. `camera_node`
   - publishes `sensor_msgs/Image`
   - default topic: `/camera/image_raw`
   - supports USB camera and local-image input

2. `yolo_detector_node`
   - subscribes to `/camera/image_raw`
   - performs YOLO11n inference
   - confidence threshold: 0.25
   - image size: 640
   - uses `rect=False`
   - publishes `vision_msgs/Detection2DArray`
   - default topic: `/detections`

3. `display_node`
   - subscribes to the detection topic
   - displays or logs class, confidence and bounding boxes

## Development Status

This package is intentionally preserved as a development prototype.

It is not presented as the final Jetson ROS2 interface.

The final deployment was designed around a Jetson-specific DZH package and a
custom detection-message interface. That final source will be preserved
separately when the original Jetson workspace is available.

Keeping the prototype separate avoids rewriting historical code and makes the
transition from the generic ROS2 pipeline to the final deployment interface
explicit.

## Prototype Data Flow

```text
USB camera / image files
        |
        v
camera_node
sensor_msgs/Image
/camera/image_raw
        |
        v
yolo_detector_node
YOLO11n
        |
        v
vision_msgs/Detection2DArray
/detections
        |
        v
display_node
```

## Notes

The prototype uses cv_bridge and vision_msgs.

It should not be confused with the later custom Jetson detection-message
interface.
