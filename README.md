# 线激光三维截面实时测量工具

主程序位于 `laser_measurement_tool/`，支持海康 MVS、大恒 Galaxy USB3 和模拟相机，
包含激光中心提取、三维重建、截面测量、结果导出及 Stage-1 离线扫描功能。

## 快速启动

```powershell
cd laser_measurement_tool
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe online_camera.py --simulate
```

真实相机需要另行安装对应厂商 SDK。正式测量前，必须确认相机、镜头、激光器、安装姿态、
曝光、ROI 和 `configs/` 中标定包一致。

## 验证

在仓库根目录执行：

```powershell
python -m pytest -q
```

详细安装、配置和操作说明见：

- [主程序说明](laser_measurement_tool/README.md)
- [在线工具用户手册](laser_measurement_tool/docs/ONLINE_USER_MANUAL.md)
- [配置字段说明](laser_measurement_tool/docs/USAGE_CONFIG.md)

运行输出默认写入 `laser_measurement_tool/output/`，该目录已被 Git 忽略。
