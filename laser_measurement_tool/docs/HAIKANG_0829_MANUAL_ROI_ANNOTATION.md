# 海康 0829 三段人工 ROI 冻结

## 目的与边界

`tools/annotate_haikang_0829_manual_rois.py` 是既有大恒三段人工 ROI 工具的海康薄适配器。每个 `h02/h06/h10/h20/h30 × p01...p10` condition 只选择一次：

1. `baseline_before`
2. `height`
3. `baseline_after`

选择结果由该 condition 的 20 帧共用。界面只显示第 10 张原始代表帧和 20 帧 Steger median centerline，不读取或显示真实高度、`h_raw`、误差、MAE/RMSE、Session Ground、board polygon 或自动 ROI 候选。

海康是 column scan（逐列扫描）。图像硬件 ROI 的 `offset_x/offset_y` 会在显示和中心线融合前加回；registry 一律保存 inclusive full-sensor `u` 范围。

## 先准备固定证据

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe .\tools\annotate_haikang_0829_manual_rois.py --prepare-only
```

首次运行对 1000 张源 PNG 各调用一次既有 Steger，并生成：

- `c0_height_audit/manual_roi/representative_view_manifest.json`
- `c0_height_audit/manual_roi/representative_views/<condition>_evidence.npz`
- `c0_height_audit/manual_roi/representative_views/<condition>_selection_view.png`
- `c0_height_audit/manual_roi/manual_roi_registry_draft.json`
- `c0_height_audit/manual_roi/manual_roi_selection_report.md`

输入 PNG 内容、`frames.csv`、海康配置、Steger/median 实现和缓存 fingerprint 一致时，后续运行直接复用 evidence。只替换了某一组源图时使用 `--refresh-condition h02_p01`；只有明确需要重算全部 1000 帧时才使用 `--refresh-evidence`。

## 人工选择

```powershell
.\.venv\Scripts\python.exe .\tools\annotate_haikang_0829_manual_rois.py
```

- 在任一图像面板横向拖拽，按 `Enter`（或点 `Confirm`）提交当前区段。
- 三段均已有范围后，再按一次 `Enter` 确认该 condition。
- `1/2/3`：切换并重选三段之一。
- `Esc` / `Reselect`：清除当前临时选区。
- `U` / `Undo`：撤销上一步。
- `S` / `Skip`：标记当前 condition 不可用。
- `P` / `Left`：上一组；`N` / `Right`：下一组。
- `Q`：保存 draft 并退出；再次启动会从第一个未完成 condition 继续。

边界必须满足：

```text
baseline_before < height < baseline_after
```

三个 inclusive `u` 范围不能重叠，不要求等宽。应避开块体两侧跳变边缘，只选择稳定内部段。

## 重选与冻结

重开一个或多个已完成 condition：

```powershell
.\.venv\Scripts\python.exe .\tools\annotate_haikang_0829_manual_rois.py `
  --reselect h06_p03 `
  --start h06_p03
```

所有 50 个 condition 均为 `selected` 或显式 `unusable` 后，工具才写出：

- `manual_roi_registry.json`
- `manual_roi_registry.csv`
- `overlays/<condition>_manual_roi.png`
- 更新后的 `manual_roi_selection_report.md`

本工具不调用 H1、H-B2 或 C1，也不修改 C0 与 Session Ground。
