from glob import glob
from setuptools import find_packages, setup

package_name = "ros2_object_detection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROS2 course project",
    maintainer_email="student@example.com",
    description="ROS2 object detection pipeline: camera + YOLO detector + display for bottle/mouse/keyboard desktop scenes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = ros2_object_detection.camera_node:main",
            "yolo_detector_node = ros2_object_detection.yolo_detector_node:main",
            "display_node = ros2_object_detection.display_node:main",
        ],
    },
)
