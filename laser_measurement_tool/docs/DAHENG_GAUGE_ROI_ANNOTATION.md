# 大恒量块人工 ROI 确认操作说明

## 目的

本工具用于对 6 个高度 × 5 个 v 位置的 30 个条件人工确认三段 v 方向 ROI：

1. `baseline_before`
2. `height`
3. `baseline_after`

每个条件只确认一次。界面使用该条件 5 张重复图的 median image，以及 5 次单帧 Steger 输出按 v 分箱后的中位点；界面只提供几何证据，不显示真值、C0/C1 高度或误差。

## 启动前检查

在仓库根目录执行，确认以下输入已存在：

- `outputs/daheng_c1_gauge_blocks_20260819/roi_registry.json`
- `outputs/daheng_c1_gauge_blocks_20260819/roi_candidates.csv`
- `outputs/daheng_c1_gauge_blocks_20260819/pointwise_diagnostics.csv`
- 6 个 `obs_*` 目录下各 25 张 TIFF

如需先只生成 30 张证据图、不打开窗口：

```bash
python tools/annotate_daheng_gauge_rois.py --prepare-only
```

默认预览使用原始灰度范围；如确实需要增强暗部显示，可加 `--contrast-stretch`，该选项只影响显示，不影响 Steger 输入或 ROI 数值。

## 启动人工确认

```bash
python tools/annotate_daheng_gauge_rois.py
```

窗口现在包含两个等像素比例的局部放大视图：左侧为自动三段 ROI 周围的 context，右侧为以中位激光点为中心的 detail。纵向范围自动覆盖 `baseline_before / height / baseline_after` 并留出上下边界观察余量；点击任一视图的水平边界都使用同一个原图 v 坐标。

每个条件按以下顺序操作：

- 当前 auto 边界正确：不点击，直接按 `Enter`。
- 当前边界错误：在图中从上边界拖到下边界形成选区，再按 `Enter`；也可以分别左键点击上、下两条 v 边界，再按 `Enter`。
- 每个条件必须依次确认 3 个区段；窗口标题中的 `1/3`、`2/3`、`3/3` 表示当前区段进度。拖拽完成或两次点击后会立即显示带斜线的临时选区，按 `Enter` 后才固定并进入下一段。
- 点击错了：按 `Esc` 清除当前两次点击后重来。
- 三段 ROI 依次完成后自动保存该条件的 manual overlay，并进入下一条件。
- 需要中止：按 `Q`。已完成的条件保存在 draft 中，重新启动会从第一个未完成条件继续。

### 只重选一个已完成条件

如果验收报告指出某个 `height × position` 的 height ROI 点数不足或边界不合适，不需要重新确认全部 30 个条件。保留 draft 后，使用 `DATASET:POSE` 指定需要重开的条目：

```bash
python tools/annotate_daheng_gauge_rois.py --reselect obs_2mm:001
```

脚本会复用 draft 中其他 29 个已确认条目，只重新打开 `obs_2mm / pose 001`；完成后仍会重新写出完整的 `roi_registry_manual.json`。多个条目可以重复写 `--reselect`。

边界顺序必须满足：

```text
baseline_before < height < baseline_after
```

工具会拒绝越界或相互重叠的选择。

## 确认完成后的文件

30/30 完成后才会写入：

- `outputs/daheng_c1_gauge_blocks_20260819/roi_registry_manual.json`
- `outputs/daheng_c1_gauge_blocks_20260819/manual_overlays/*.png`（30 张）
- `outputs/daheng_c1_gauge_blocks_20260819/roi_auto_vs_manual.csv`
- `outputs/daheng_c1_gauge_blocks_20260819/roi_auto_vs_manual.json`

在 30/30 之前只会写 `roi_registry_manual_draft.json`，不会生成可供验收器使用的 frozen registry。

## 使用冻结 ROI 重跑验收

```bash
python tools/evaluate_daheng_c1_gauge_blocks.py \
  --roi-registry outputs/daheng_c1_gauge_blocks_20260819/roi_registry_manual.json \
  --output outputs/daheng_c1_gauge_blocks_20260819_manual_frozen
```

该参数会严格要求 registry 顶层和每条记录均为 `manual_confirmed=true`，并跳过自动 ROI 选择。C0/C1 仍使用同一次 Steger 结果和同一组 frozen ROI。

验收报告位于新的 output 目录中的 `gauge_block_acceptance_report.md`，同时会生成 CSV、JSON 和图表。

## 与生产 GUI 对比时的注意事项

生产 GUI 默认启动的是 `configs/measure_tool.yaml`；比较本批量块必须使用同一份 Daheng 配置：

```bash
cd D:\Docs\linelaserscan\0704line-laser-3d-scanner
.\.venv\Scripts\python.exe .\laser_measurement_tool\online_camera.py `
  --camera-backend daheng `
  --config .\laser_measurement_tool\configs\measure_tool_daheng_0811.yaml
```

从在线窗口进入“单帧测量与区域选择”后点击“加载图像”读取历史 TIFF，GUI 会用同一个 Daheng `AppConfig`，按同一个 `steger` 参数调用一次 `extract_laser_center`。本批量 TIFF 是 `4096×3000、offset=(0,0)`，因此 GUI 与验收脚本都在同一全幅坐标系中。

本标注窗口显示的是 5 次单帧 Steger 点集按 v 分箱后的中位结果叠加到 5 张图的 median image；标注图只画离散点，不连接相邻点，避免人为跨越无效区段。生产 GUI 加载一张 TIFF 时显示的是该单帧中心点，两者的缺失点和局部断点可能不同，但不能据此判断 Steger 算法不同。若 GUI 加载的是 `480×3000` 相机 ROI 图，比较全幅坐标时要给 GUI 的局部 `u` 加上 `(1760, 0)` 偏移。

当前生产 GUI 没有独立的黑棋盘格识别 mask；黑格边缘是否出现中心点，取决于 Steger 的亮度、Hessian 响应和连续性条件。因此“GUI 没有提取黑格”不能作为批量流程已经使用棋盘格过滤的证据。

在线窗口的“单帧测量与区域选择”按钮对当前在线帧不会重新运行 Steger，而是复用在线帧处理阶段已经生成的 `centers_uv_full`；但在该窗口中点击“加载图像”读取历史 TIFF 时，会按上面的 GUI 单帧路径重新提取。要做严格逐点对比，必须比较同一 TIFF、同一 offset 和同一配置；不能用 5 次 median centerline 与 GUI 单帧中心线直接逐点比较。

当前仓库量块数据的 `frames.csv`/`dataset_manifest.yaml` 记录为 `4096×3000、offset=(0,0)、曝光330µs`；这与当前 Daheng 在线配置的 `480×3000、offset=(1760,0)、曝光2000µs` 不是同一采集协议。若需要验收当前在线配置下的图像，应重新采集或把在线保存的原始帧及元数据纳入数据集。

## 常见问题

- 如果窗口无法启动，确认本机 Python 环境安装了 Qt backend；脚本使用 Matplotlib `qtagg`。
- 如果只看到 draft 而没有 `roi_registry_manual.json`，说明尚未完成 30/30，或中途按了 `Q`。
- 不要手工把 draft 改名为 frozen registry；验收器会校验 30/30 冻结标记。
