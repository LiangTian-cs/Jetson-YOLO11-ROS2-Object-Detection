# 最终项目报告：桌面目标检测（bottle / mouse / keyboard）

> 状态：模板已就绪，已填入当前已验证数据；带「待填」的章节需在 Jetson 到手后完成。

## 1. Dataset construction（数据集构建）

- 冻结数据：`dataset_versions/dataset_v004_three_class_final`（TREE_SHA256 `b0f042958c26a7bfb34fdf6c4f8216a9ad578723f88a3d33b5af41144aa8cc9a`）
- 来源：Open Images V7 + COCO，经两轮人工审核（v004：accepted 248 / removed 33，bbox 修正 1）
- 总图片 388，总 bbox 609；train/val = 308/80（79.4% / 20.6%），划分 seed=42
- 类别分布：bottle 155 图 / 208 bbox；mouse 201 图 / 207 bbox；keyboard 176 图 / 194 bbox
- bbox 面积（相对图）：min 0.0103，median 0.0432，mean 0.0904，p95 0.3226，<1% 数量 0
- 报告：`reports/dataset_v004_final_statistics.md`、`reports/v004_freeze_report.md`、`reports/v004_manual_review.md`

## 2. Model training（模型训练）

- 环境：Ultralytics 8.4.127 / PyTorch 2.11.0+cu128 / RTX 5090；imgsz=640，batch=16，device=0
- 训练实验（v004 final）：
  - `runs/exp003_yolov8n_v004`（YOLOv8n，best epoch 17）
  - `runs/exp004_yolo11n_v004`（YOLO11n，best epoch 77）
- 统一 v004 val 指标（80 图 / 146 bbox）：

| 模型 | Precision | Recall | mAP50 | mAP50-95 | 参数 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n v004 | 0.7754 | 0.6702 | 0.7565 | 0.5352 | 3,006,233 |
| YOLO11n v004 | 0.8095 | 0.6839 | 0.7818 | 0.5861 | 2,582,737 |

## 3. Model comparison（模型比较）

- YOLO11n 在 mAP50-95（+0.0509）、mouse recall（0.7671）、near-small recall（22/32，0.6875）领先，且参数少 423,496。
- YOLOv8n 在 mouse 误检上更优（FP 14→5，-64.3%）。
- 每类指标详见 `reports/v004_model_comparison.md`。
- 结论：**YOLO11n v004 为主要部署模型；YOLOv8n v004 作为 low-mouse-FP 对照**。

## 4. ONNX deployment（ONNX 部署）

- 制品：`deployment/models/yolo11n_v004.onnx`（10,566,605 B / 10.57 MB）
- 输入 `images [1,3,640,640]`、输出 `output0 [1,7,8400]`、opset 17（`deployment/models/yolo11n_v004.meta.json`）
- 一致性（`deployment/test_onnx.py`，80 图 val，rect=False）：**141/141 匹配 100%**，IoU 均值 0.9989 / 最小 0.9621，conf 差均值 0.0004 / 最大 0.0038
- ONNX CPU 延迟：均值 192.3 ms（≈5.2 FPS）；PyTorch GPU / Jetson TensorRT 待测
- 关键点：ultralytics 默认 `rect=True` 会使 PT 用最小矩形输入而 ONNX/Engine 固定 640——已固定 `rect=False` 统一
- TensorRT：Jetson 端流程见 `deployment/build_tensorrt.md`；JP/TensorRT 版本待填

## 5. ROS2 Architecture

The final Jetson deployment uses the personal ROS2 package `yolo_detector_dzh`.

- Node: `dzh_yolo_camera_node`
- Camera: `/dev/video0`, YUYV, 640×480 @ 30 FPS
- Model: `/home/nvidia/yolo_deploy/models/best_DZH.pt`
- Inference parameters: `conf=0.25`, `imgsz=640`, `rect=False`, CUDA device 0
- Published topic: `/DZH/yolo/detections`
- Message type: `yolo_interfaces/msg/DetectionArray`
- Per-object fields: class ID, class name, confidence, xmin, ymin, xmax, ymax

The package was successfully built on the Jetson using `colcon build`, and ROS2
successfully discovered and executed:

`yolo_detector_dzh dzh_yolo_camera_node`

The measured publication rate of `/DZH/yolo/detections` was approximately
**12.49 Hz**, which is well above the coursework requirement of 5 FPS.

A saved ROS2 message contained simultaneous detections of:

- keyboard, confidence = 0.8682
- mouse, confidence = 0.3536

The message also contained the corresponding class IDs, confidence values, and
bounding-box coordinates.

Therefore, the final ROS2 detection publishing pipeline was successfully
validated on the Jetson hardware.

> The repository preserves the earlier three-node `vision_msgs` prototype under
> `ros2/prototype/`. The final DZH-specific implementation is stored under
> `ros2/final_dzh/`.
>
> The final ROS2 runtime validation used the PyTorch/CUDA model. TensorRT FP16
> performance was tested separately and is not presented as ROS2 TensorRT
> performance.

## 6. Jetson Deployment

The final system was deployed and validated on an NVIDIA Jetson Orin platform
with ROS2 Humble.

The verified software environment was:

- Ubuntu 22.04, aarch64
- L4T R36.4.7
- CUDA 12.6
- PyTorch 2.10.0
- TensorRT 10.3.0
- Ultralytics 8.4.127

The final PyTorch model was:

`best_DZH.pt`

SHA256:

`7f084f4d2a106585b4d74f186e78dd4b7ac1b1d7dda0aa93fa5ffda82c5d81eb`

### 6.1 PyTorch/CUDA Real-Time Camera Test

A recorded Jetson camera run processed 1742 frames.

Measured performance:

- Mean end-to-end FPS: **20.55 FPS**
- Mean inference time: approximately **25.78 ms**
- Coursework requirement: **≥ 5 FPS**
- Result: **PASS**

This result represents the complete PyTorch/CUDA real-time camera pipeline.

### 6.2 TensorRT FP16 Deployment

The TensorRT FP16 engine was generated directly on the Jetson using
TensorRT 10.3.0.

Engine information:

- Engine file: `best_DZH_fp16.engine`
- File size: approximately 8.5 MB
- Input shape: 1×3×640×640
- SHA256:
  `c97a0b14083db07d03cb8f4afc52e7f9959d8492a39885a6216b7fbe79d0a14a`

The TensorRT camera benchmark used 5 warm-up frames followed by 25 measured
frames.

Measured results:

- Mean inference-pipeline latency: **23.45 ms**
- Mean inference-pipeline throughput: **42.64 FPS**
- Simultaneous keyboard and mouse detections were observed
- Result: **PASS**

The reported 42.64 FPS represents TensorRT inference-pipeline throughput after
a camera frame has been acquired. It should not be interpreted as complete
camera end-to-end FPS.

## 7. Real-Object Evaluation

The frozen formal acceptance manifest contained 20 predefined target objects.

The automatic evaluation results were:

- Correct detections: **17/20**
- Overall accuracy: **85.00%**
- Bottle: **5/7**
- Mouse: **6/6**
- Keyboard: **6/7**
- Coursework requirement: **≥ 80%**
- Result: **PASS**

Typical failures included missed detections of several bottle and keyboard
instances. The frozen automatic evaluation results were preserved exactly and
were not manually edited.

A supplementary visual audit was also performed by counting all physical
objects actually visible in the captured evaluation images.

The supplementary result was:

- **19/22 = 86.36%**

The two results use different evaluation scopes:

- **17/20 = 85.00%** is the official frozen-manifest automatic evaluation.
- **19/22 = 86.36%** is the supplementary visual audit of all visible physical
  objects.

The supplementary audit does not replace or modify the official automatic
evaluation result.

Overall, the complete pipeline — model training, Jetson real-object detection,
real-time camera inference, ROS2 publishing, and TensorRT FP16 deployment —
was successfully validated on the target hardware.
