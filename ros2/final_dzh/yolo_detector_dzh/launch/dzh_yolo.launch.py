from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("yolo_detector_dzh"))
    config = share / "config" / "dzh_yolo.yaml"

    return LaunchDescription(
        [
            Node(
                package="yolo_detector_dzh",
                executable="dzh_yolo_camera_node",
                name="dzh_yolo_camera_node",
                output="screen",
                parameters=[str(config)],
            )
        ]
    )
