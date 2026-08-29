# ros2_object_detection

ROS2 object detection pipeline for desktop scenes (bottle / mouse / keyboard).

- `camera_node` publishes `sensor_msgs/Image` on `/camera/image_raw` from a **USB camera** or from **local image files** (for offline testing).
- `yolo_detector_node` subscribes to `/camera/image_raw`, runs a trained **YOLO11n** model and publishes `vision_msgs/Detection2DArray` on `/detections`.
- `display_node` subscribes to `/detections`, draws **bbox / class / confidence** over the latest image in a live OpenCV window (or logs them in `log` mode).

## Package layout

```
ros2_object_detection/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── ros2_object_detection
├── config/
│   └── ros2_object_detection.yaml      # canonical parameter defaults
├── launch/
│   └── ros2_object_detection.launch.py # launches all three nodes
├── ros2_object_detection/
│   ├── camera_node.py
│   ├── yolo_detector_node.py
│   └── display_node.py
└── README.md
```

## Prerequisites

- ROS 2 (Ubuntu 22.04 → **Humble**; other distros work with matching packages).
- Packages: `rclpy`, `sensor_msgs`, `vision_msgs`, `cv_bridge`, `python3-opencv`, `python3-numpy`, `launch`, `launch_ros`, `ament_index_python`.
  ```bash
  sudo apt update
  sudo apt install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-cv-bridge \
                   ros-$ROS_DISTRO-launch ros-$ROS_DISTRO-launch-ros \
                   python3-opencv python3-numpy
  ```
- Ultralytics + PyTorch to load and run the YOLO model:
  ```bash
  pip install ultralytics torch
  ```
- The weights file path is set in `config/ros2_object_detection.yaml` under `yolo_detector_node.ros__parameters.weights`. Point it to your trained model, e.g. `runs/exp004_yolo11n_v004/weights/best.pt` (on this project) or a copied `best.pt` on the deployment machine.

## Build

```bash
cd /root/autodl-tmp/ros2_object_detection/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install --packages-select ros2_object_detection
source install/setup.bash
```

## Run

The launch file reads `config/ros2_object_detection.yaml` by default and exposes common overrides as launch arguments.

### USB camera input

```bash
ros2 launch ros2_object_detection ros2_object_detection.launch.py \
    source_type:=usb device_index:=0
```

To use a specific device path:

```bash
ros2 launch ros2_object_detection ros2_object_detection.launch.py \
    source_type:=usb device_url:=/dev/video1
```

### Local image input (offline test)

Publish a single image, or all images in a directory, in a loop:

```bash
ros2 launch ros2_object_detection ros2_object_detection.launch.py \
    source_type:=image image_path:=/root/autodl-tmp/ros2_object_detection/dataset_versions/dataset_v004_three_class_final/yolo/images/val
```

For a single file:

```bash
ros2 launch ros2_object_detection ros2_object_detection.launch.py \
    source_type:=image image_path:=/path/to/test.jpg
```

### Useful overrides

| Argument | Default | Meaning |
|---|---|---|
| `source_type` | `usb` | `usb` or `image` |
| `image_path` | `` | file/directory for `source_type=image` |
| `device_index` | `0` | USB camera index |
| `fps` | `30.0` | publish rate |
| `confidence` | `0.25` | detection confidence threshold |
| `imgsz` | `640` | YOLO inference size |
| `device` | `0` | `0`=GPU, `cpu`=CPU |
| `log_detections` | `false` | true logs every detection (class/conf/bbox) |
| `display_mode` | `window` | `window` for OpenCV overlay, `log` for headless text |
| `image_topic` | `/camera/image_raw` | input image topic |
| `detection_topic` | `/detections` | output detections topic |

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | camera → detector / display |
| `/detections` | `vision_msgs/Detection2DArray` | detector → display |

## Report metric baseline

Trained and evaluated on the frozen `dataset_v004_three_class_final` split (388 images / 609 bbox, train/val = 308/80). YOLO11n v004: P=0.8095, R=0.6839, mAP50=0.7818, mAP50-95=0.5861.

## Notes

- Detector loads the model once at startup and verifies the class contract `{0: bottle, 1: mouse, 2: keyboard}`.
- `display_node` only logs when no image has arrived yet; it also needs the image topic to draw boxes (it subscribes to `/camera/image_raw` by default).
- For a headless / no-GUI machine, run with `display_mode:=log`.
