# 海康 0829 冻结人工 ROI C0 重放

## 边界

`tools/replay_haikang_manual_roi_c0_0829.py` 执行 `H0-1M-B`：使用 `H0-1M-A` 冻结的三段 full-sensor `u` ROI，绕过 Auto ROI-V2 target detection，保留生产 Steger、circular-cone C0、Session Ground 与 `measure_height_line`。

输出统一标记为 `MANUAL_ROI_DIAGNOSTIC`，不能表述为当前自动 ROI 系统的最终生产精度。脚本不重新标定 C0，不修改 Session Ground，不拟合或应用 H1/H-B2/C1。

## 执行

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe .\tools\replay_haikang_manual_roi_c0_0829.py
```

每帧链路：

```text
PNG
→ FramePipeline.run_frame
→ Steger
→ circular-cone C0
→ Session Ground
→ frozen manual full-sensor u masks
→ concatenate(baseline_before, baseline_after)
→ existing measure_height_line(..., ground_correction_mode="auto")
→ h_raw_mm
```

同一组点另以 `ground_correction_mode="session_reference"` 生成 `session_height_mean_mm` 对照。高度公式没有复制到适配器。

## Truth barrier

脚本先完成全部 50×20 帧重放并写出不含 truth/accuracy-error 的 `manual_h_raw_frames.csv`（仍保留 pipeline/measurement 失败诊断字段），之后才加载目录 ground truth、计算 condition-level accuracy 和画图。20 帧仅作为重复测量；overall accuracy 的统计单元是 50 个 condition median。

## 输出

默认目录：

```text
0829/c0_height_audit/manual_roi_measurement/
```

包含：

- `manual_h_raw_frames.csv`
- `manual_h_raw_position_summary.csv`
- `manual_c0_accuracy_summary.json`
- `manual_c0_height_audit_report.md`
- `height_pred_vs_gt.png`
- `error_vs_height.png`
- `error_vs_position.png`
- `height_position_residual_heatmap.png`
- `temporal_std_vs_condition.png`

约 0.2 mm 目标同时报告两种口径：aggregate/P95（MAE 与 P95 absolute error）以及严格 all-condition Max，避免用 aggregate 结论掩盖单个 condition 的较大误差。
