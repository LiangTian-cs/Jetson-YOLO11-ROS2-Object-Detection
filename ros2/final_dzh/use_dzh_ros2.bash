#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

source /home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces/share/yolo_interfaces/local_setup.bash

export AMENT_PREFIX_PATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces:${AMENT_PREFIX_PATH:-}
export CMAKE_PREFIX_PATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces:${CMAKE_PREFIX_PATH:-}
export PYTHONPATH=/home/nvidia/yolo_deploy/common_ws/install/yolo_interfaces/local/lib/python3.10/dist-packages:${PYTHONPATH:-}

if [ -f /home/nvidia/yolo_deploy/members/DZH_ws/install/yolo_detector_dzh/share/yolo_detector_dzh/local_setup.bash ]; then
    source /home/nvidia/yolo_deploy/members/DZH_ws/install/yolo_detector_dzh/share/yolo_detector_dzh/local_setup.bash
fi

source /home/nvidia/yolo_deploy/.venv/bin/activate
