# 单帧线激光三维截面测量工具

从单帧线激光图像提取亚像素中心，恢复地面坐标系点云，并测量障碍物高度与
截面长度。当前快照已内置运行所需的相机内参、圆锥激光面模型、地面外参和地面 U
向补偿表，可单独复制或克隆本目录运行。

在线实时工具的界面、取流、FPS、硬件 ROI、单帧测量、结果导出和配置替换说明见
[在线实时工具用户手册](docs/ONLINE_USER_MANUAL.md)。

## 运行

```powershell
python -m pip install -r requirements.txt
python main.py
```

默认配置为 `configs/measure_tool.yaml`，其中所有相对路径都以该配置文件所在
目录为基准。内置标定文件位于 `configs/calibration/`，测量结果写入 `output/`。

```powershell
python main.py --config configs/measure_tool.yaml
$env:PYTHONPATH="$PWD;$PWD\.."
python -m unittest discover -s tests
```

在线实时工具支持海康 MVS（默认）、大恒 Galaxy USB3 和模拟相机：

```powershell
python online_camera.py --camera-backend daheng
python online_camera.py --simulate
```

大恒 backend 会从 `C:\Program Files\Daheng Imaging\GalaxySDK` 或
`DAHENG_GALAXY_ROOT` 加载 SDK 随附的 `gxipy`。切换到大恒相机后，应使用独立的
大恒标定 manifest；当前内置海康标定不能直接用于大恒正式测量。

### 在线实时工具的单帧分析

运行 `online_camera.py`（或在主窗口点击“在线相机”）后，实时窗口保留原有
图像快照和定长录制功能，并新增：

- “导出当前点云/CSV”：保存当前帧的 `laser_center.csv`、`full_points.csv`、
  `full_laser_ground.ply`、`overlay.png` 和 `result.json` 到
  `output/online_measurements/`；
- “单帧测量与区域选择”：打开与离线工具相同的分析界面，可框选基准区域、一个或
  多个障碍物区域，执行三维恢复与高度/长度测量，再用“保存结果”输出完整测量文件。

实时相机的 ROI 偏移会自动转换回标定使用的全幅像素坐标；分析窗口中的框选仍以
当前相机 ROI 图像坐标显示。

通过“加载图像”打开独立保存的硬件 ROI/软件裁剪图时，程序会优先读取同目录的
相邻 JSON、`result.json` 或 `frames.csv` 中的 `OffsetX/OffsetY`。如果没有元数据，
会在加载时要求确认图像在标定全幅中的左上角偏移；不能把 ROI 局部坐标直接当作全幅
像素坐标参与三维恢复。

## 目录

- `configs/calibration/`：当前设备对应的运行标定与补偿数据。
- `laser/`：centroid、Steger 和 shared Steger 激光中心提取实现。
- `reconstruction/`：像素去畸变、射线与激光表面求交、地面坐标转换。
- `measurement/`：局部地面拟合和障碍物高度统计。
- `gui/`：图像、三维点云和截面交互视图。
- `docs/`：配置、补偿和使用说明。

内置标定只适用于生成这些参数时的相机、镜头、激光器及其安装位置。更换设备
或移动相机/激光器后，必须重新标定并替换 `configs/calibration/` 中对应文件。
