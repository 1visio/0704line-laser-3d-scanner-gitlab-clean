# 海康 0829 H1 可行性审计

脚本：`tools/audit_haikang_h1_feasibility_0829.py`

本审计只处理 H0-1M-B 的 `MANUAL_ROI_DIAGNOSTIC` condition-level 结果，不改 C0、Session Ground、C1、H-B2、在线配置，也不生成生产补偿文件。

## 复用的 H1 定义

沿用大恒 `tools/validate_height_linear_cv.py` 的 H1：

```text
h_corr = k * h_raw
k = sum(h_raw * truth) / sum(h_raw ** 2)
```

拟合和预测分别直接调用既有 `_fit_parameters(model="H1")`、`_predict(model="H1")`。每个 condition 使用 `manual_h_raw_position_summary.csv` 的 median；20 帧只用于输入完整性核验，不展开为训练样本。

## 分组协议

- LOHO interpolation：留出 6、10、20 mm，各 fold 训练其余高度。
- Endpoint extrapolation：`LOW_END_EXTRAPOLATION` 使用 6/10/20/30 mm 训练、留出 2 mm；`HIGH_END_EXTRAPOLATION` 使用 2/6/10/20 mm 训练、留出 30 mm。端点结果单独保存。
- LOPO：依次留出 p01–p10，每 fold 用其余 9 个 position 训练。
- 所有参数仅由对应训练 fold 拟合；禁止 full-data fit 后回评。
- 可声明数据域为 `valid_height_domain = [2, 30] mm`，不对域外高度作有效性声明。

## 运行与输出

```bash
python tools/audit_haikang_h1_feasibility_0829.py
```

结果写入 `0829/c0_height_audit/h1_feasibility/`：四个 CSV、汇总 JSON、Markdown 报告，以及 raw/H1 error、LOHO、LOPO、端点和 residual heatmap 图。

本目录是 feasibility 诊断产物；即使 H1 在某些 fold 改善，也不得直接复制为在线生产补偿配置。
