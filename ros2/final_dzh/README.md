# DZH Jetson ROS2 Publisher

This directory contains the DZH-specific ROS2 detection publisher for the
Jetson deployment.

## Provenance

The original Jetson workspace was:

`/home/nvidia/yolo_deploy/members/DZH_ws`

The original package contract was reconstructed from preserved deployment
records after the source file was no longer available on the Mac or RTX5090
server.

This repository version is therefore not claimed to be a byte-for-byte copy
of the original Jetson source.

The message definitions, environment layout, camera configuration, model
contract and ROS2 topic contract are preserved from verified project records.

Live Jetson build/run validation should be added as a separate evidence commit.

## Package

- Package: `yolo_detector_dzh`
- Node: `dzh_yolo_camera_node`
- Detection topic: `/DZH/yolo/detections`
- Model: `/home/nvidia/yolo_deploy/models/best_DZH.pt`
- Camera: `/dev/video0`

## Inference Contract

- classes: bottle, mouse, keyboard
- confidence threshold: 0.25
- imgsz: 640
- rect: False
- camera: 640x480 YUYV
- requested camera rate: 30 FPS
- inference device: CUDA device 0

## ROS2 Interface

The deployed system uses the shared package `yolo_interfaces`.

`Detection.msg`:

```text
int32 class_id
string class_name
float32 confidence
int32 xmin
int32 ymin
int32 xmax
int32 ymax
```

`DetectionArray.msg`:

```text
std_msgs/Header header
uint32 image_width
uint32 image_height
float32 fps
yolo_interfaces/Detection[] detections
```

Copies under `interface_contract/` are documentation snapshots only. Do not
replace or rebuild the shared Jetson `yolo_interfaces` package.

## Jetson Build

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces/share/yolo_interfaces/local_setup.bash
export AMENT_PREFIX_PATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces:$AMENT_PREFIX_PATH
export CMAKE_PREFIX_PATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces:$CMAKE_PREFIX_PATH
export PYTHONPATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces/local/lib/python3.10/dist-packages:$PYTHONPATH
source /home/nvidia/yolo_deploy/.venv/bin/activate

cd /home/nvidia/yolo_deploy/members/DZH_ws
colcon build --packages-select yolo_detector_dzh --symlink-install
source install/yolo_detector_dzh/share/yolo_detector_dzh/local_setup.bash
```

## Run

```bash
ros2 run yolo_detector_dzh dzh_yolo_camera_node
```

or:

```bash
ros2 launch yolo_detector_dzh dzh_yolo.launch.py
```

## Verification

```bash
ros2 interface show yolo_interfaces/msg/Detection
ros2 interface show yolo_interfaces/msg/DetectionArray
ros2 topic list | grep DZH
ros2 topic info /DZH/yolo/detections
ros2 topic echo /DZH/yolo/detections --once
ros2 topic hz /DZH/yolo/detections
```

Live command output should be preserved later as Jetson ROS2 evidence.
