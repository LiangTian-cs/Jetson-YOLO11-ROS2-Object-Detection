from glob import glob

from setuptools import find_packages, setup


package_name = "yolo_detector_dzh"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.launch.py"),
        ),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DZH",
    maintainer_email="LiangTian-cs@users.noreply.github.com",
    description="Jetson YOLO11 camera detector using yolo_interfaces.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dzh_yolo_camera_node = yolo_detector_dzh.camera_node:main",
        ],
    },
)
