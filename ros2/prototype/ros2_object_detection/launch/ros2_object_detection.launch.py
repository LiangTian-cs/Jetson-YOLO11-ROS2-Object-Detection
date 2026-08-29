"""Launch the full ros2_object_detection pipeline.

Starts camera_node (USB or local image), yolo_detector_node (YOLO11n) and display_node.
Parameter defaults come from config/ros2_object_detection.yaml; run-specific values can be
overridden with the launch arguments below.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = get_package_share_directory("ros2_object_detection")
    default_params = os.path.join(package_share, "config", "ros2_object_detection.yaml")

    declared_arguments = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        # camera
        DeclareLaunchArgument("source_type", default_value="usb"),
        DeclareLaunchArgument("image_path", default_value=""),
        DeclareLaunchArgument("device_index", default_value="0"),
        DeclareLaunchArgument("fps", default_value="30.0"),
        # detector
        DeclareLaunchArgument("confidence", default_value="0.25"),
        DeclareLaunchArgument("imgsz", default_value="640"),
        DeclareLaunchArgument("device", default_value="0"),
        # topics / display
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("detection_topic", default_value="/detections"),
        DeclareLaunchArgument("display_mode", default_value="window"),
    ]

    params_file = LaunchConfiguration("params_file")

    camera_params = [
        params_file,
        {
            "source_type": LaunchConfiguration("source_type"),
            "image_path": LaunchConfiguration("image_path"),
            "device_index": ParameterValue(LaunchConfiguration("device_index"), value_type=int),
            "fps": ParameterValue(LaunchConfiguration("fps"), value_type=float),
            "image_topic": LaunchConfiguration("image_topic"),
        },
    ]
    detector_params = [
        params_file,
        {
            "confidence": ParameterValue(LaunchConfiguration("confidence"), value_type=float),
            "imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int),
            "device": LaunchConfiguration("device"),
            "image_topic": LaunchConfiguration("image_topic"),
            "detection_topic": LaunchConfiguration("detection_topic"),
        },
    ]
    display_params = [
        params_file,
        {
            "display_mode": LaunchConfiguration("display_mode"),
            "image_topic": LaunchConfiguration("image_topic"),
            "detection_topic": LaunchConfiguration("detection_topic"),
        },
    ]

    camera_node = Node(
        package="ros2_object_detection",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=camera_params,
    )
    detector_node = Node(
        package="ros2_object_detection",
        executable="yolo_detector_node",
        name="yolo_detector_node",
        output="screen",
        parameters=detector_params,
    )
    display_node = Node(
        package="ros2_object_detection",
        executable="display_node",
        name="display_node",
        output="screen",
        parameters=display_params,
    )

    return LaunchDescription([*declared_arguments, camera_node, detector_node, display_node])
