# Haikang calibration package 2026-08-28

这是在线测量工具的独立海康标定包，对应第一轮海康结果。源目录实际名称为
`calibration_tool/projects/haikang/otutputs/0828`（`otutputs` 是现有目录名）。

运行时文件：

- `calibration_result.yaml`：相机内参；
- `circular_cone.yaml`：第一轮模型比较后选定的圆锥模型，来自
  `laser_model/models/circular_cone.yaml`；
- `camera_ground_extrinsics.yaml`：第一轮地面外参；
- `manifest.yaml`：在线包的文件清单与 SHA-256 校验。

本包采用 `correction.mode: none` 作为海康量块验证基线。没有加入大恒的 H1/HB2、
laser ray C1 或未重新验证的 ground-U 补偿。后续如使用海康量块拟合修正，应新建
海康专用修正文件并更新 package id，不覆盖本包的基线结果。

模型有效相机深度范围为约 734.775～852.587 mm；运行配置使用该范围外扩后的
725～865 mm 工作门限。在线 ROI、曝光、增益和像素格式改变时，应重新做独立验证。
