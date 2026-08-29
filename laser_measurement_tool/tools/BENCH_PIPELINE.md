# bench_pipeline.py · 流水线性能基准 + AOI 等价性诊断

测量"输入图像 → 亚像素中心 → 地面系点云"这条链路在**本机**上的每帧耗时，分解各段占比，
并诊断把处理范围限制到条纹附近（AOI）之后结果差在哪里、哪个更可信。

回答三个问题：实时在线能跑多少帧率？AOI 能提速多少？AOI 与全幅谁对？

> **第三个问题不是形式检查。** `laser/backends.py` 的逐列提取用的是**整列 `np.argmax`**，
> 视场里任何比条纹更亮的特征都会抢走峰值——那一列要么被静默丢弃、要么给出错误的中心。
> 本脚本会统计"全幅峰落在条纹带之外"的列数与最长连续段：连续段一旦超过
> `segment_min_columns`（默认 42），全幅提取就会输出**一整段位置错误的伪点云**，
> 而下游没有任何环节能发现。这是本脚本最重要的输出。

- 放置位置：`laser_measurement_tool/tools/bench_pipeline.py`
- 依赖：仓库 `.venv` 已有的 `numpy` / `opencv-python`，无需新装包
- 不写任何结果到仓库（除显式 `--json`），不修改任何标定或配置

## 快速开始

```powershell
Set-Location "G:\dev\projects\0704linescan\laser_measurement_tool"

# 1. 合成条纹基准（最快，10 秒出结果，用于跨机器对比）
& "..\.venv\Scripts\python.exe" tools\bench_pipeline.py --repeat 10

# 2. 用真实帧基准（推荐，反映真实条纹宽度与噪声）
& "..\.venv\Scripts\python.exe" tools\bench_pipeline.py `
  --image "..\calibration\外参标定\laser_obs\Image_20260722140055330.tif" `
  --repeat 10

# 3. 带真实标定 + 落 JSON（用于性能回归对比）
& "..\.venv\Scripts\python.exe" tools\bench_pipeline.py `
  --image "..\calibration\外参标定\laser_obs\Image_20260722140055330.tif" `
  --intrinsics "..\calibration\calib02\calibration_fix_k3_exclude_018_026_030_031_matlab\calibration_result.yaml" `
  --laser-plane "..\calibration\外参标定\laser_plane.yaml" `
  --extrinsics "..\outputs\ground_extrinsics\camera_ground_extrinsics.yaml" `
  --repeat 10 --json "..\outputs\bench\bench_YYYYMMDD.json"
```

## 参数说明

### 输入图像（二选一）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--image PATH` | 无 | 真实图像路径。支持中文路径（内部用 `np.fromfile` + `imdecode`）。彩色图自动转灰度。**给了这个就忽略下面三个合成参数。** |
| `--width N` | 2448 | 合成图宽度（像素） |
| `--height N` | 2048 | 合成图高度（像素） |
| `--bit-depth {8,12}` | 8 | 合成图位深。选 12 时峰值与噪声按 Mono12 量级放大 |

合成条纹是一条接近水平、带轻微弯曲的高斯亮线（σ≈2.2 px）+ 高斯噪声，形态接近 `laser_obs` 数据集。用它的好处是**跨机器可比**，坏处是不含真实的饱和、反射率突变与断线。

### 提取参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--method {centroid,steger}` | steger | 提取算法；当前默认使用统一实时 Steger |
| `--background-kernel N` | 51 | 背景抑制高斯核，必须为 ≥3 的奇数。与 `configs/measure_tool.yaml` 同名字段含义一致 |
| `--min-local-contrast-dn F` | 20.0 | 峰值最低局部对比度（DN，按源位深）。Mono12 图像需按位深放大 |
| `--steger-sigma F` | 1.5 | 统一实时 Steger 高斯导数尺度；FWHM 2～3 px 可在 1.2～2.0 间验证 |
| `--steger-threshold F` | 30.0 | Steger 原始灰度下限（DN） |
| `--steger-deriv-thresh F` | 0.5 | Steger 法向二阶导数阈值 |
| `--steger-roi-margin N` | 120 | Steger 自动条纹带扩展量 |
| `--steger-roi-max-height N` | 512 | Steger Hessian 计算带最大宽度 |
| `--scan-axis {column,row}` | column | `column`=条纹接近水平（AOI 是**行带**）；`row`=条纹接近竖直（AOI 是**列带**）。**与图像实际条纹方向不匹配会提取不到点** |
| `--aoi-margin N` | 150 | AOI 相对条纹范围的外扩余量（像素）。脚本用"全幅结果的中心范围 ± 该余量"构造 AOI，模拟实时模式下按上一帧跟踪条纹 |

### 标定（可选）

| 参数 | 说明 |
|---|---|
| `--intrinsics PATH` | 相机内参 YAML（`camera_matrix`/`K` + `dist_coeffs`/`D`） |
| `--laser-plane PATH` | 激光平面 YAML（`coefficients{a,b,c,d}` / `plane` / `plane_abcd` 任一写法） |
| `--extrinsics PATH` | 地面外参 YAML（`R`+`t` 或 `T_ground_from_camera` 4×4） |

三个都给才会走 `calibration/config_loader.load_calibration_files()` 真实加载；否则用内置合成标定。
**合成标定只影响点云数值，不影响耗时结论** —— 重建段耗时与标定内容无关。

### 其他

| 参数 | 默认 | 说明 |
|---|---|---|
| `--repeat N` | 10 | 每项重复次数。每项都先预热一次再计时，排除首次分配与 OpenCV 内部初始化。重建段自动至少跑 20 次（单次太快） |
| `--skip-variants` | 关 | 只测"全幅默认"与"AOI"，跳过参数变体与背景抑制替代方案对比。适合快速回归 |
| `--json PATH` | 无 | 把完整结果（平台、参数、各项耗时统计）写成 JSON。目录不存在会自动创建 |

## 输出怎么读

```
项目                                                       均值ms     中位ms     FPS
------------------------------------------------------------------------------
提取 · 全幅（默认参数）                                           134.9    135.5     7.4  2448 点
提取 · AOI 378×2448（参数不变）                                  38.8     38.5    25.7  行范围 [874, 1252)
重建 · 2448 点 → 地面系                                         1.1      1.0   885.4
提取 · 全幅 + 关闭段内平滑（correction_window=1）                   115.6    110.6     8.6  数值会与默认不同
提取 · 全幅 + background_kernel=25 + 关平滑                      88.8     87.8    11.3  数值会与默认不同
  └ 分解：GaussianBlur(51,51) 单独                             39.7     32.3    25.2
  └ 分解：boxFilter(51,51) 替代方案                               6.1      6.2   164.0
  └ 分解：1/4 下采样 + Gauss13 + 上采样 替代方案                        3.3      3.3   301.2
----------------------------------------------------------------------------------
AOI 等价性（按 u 配对比较，AOI 范围 [605, 1026)）
  点数：全幅 2435  AOI 2441  共同 u 2435  仅全幅有 0  仅 AOI 有 6
  共同 u 上 max|Δv| = 0.090945 px（超过 1e-6 的 u 数：6）
  差异样例：u=1069: 全幅 872.7349 / AOI 872.7519，...
  → AOI 比全幅多出点，且没有丢点。这通常意味着全幅的整列 argmax 被条纹带外
    更亮的特征抢走，那些列被全幅丢弃了。此时 AOI 结果更可信，不是精度妥协。
----------------------------------------------------------------------------------
整列 argmax 干扰诊断（backends 的固有风险）
  全幅峰落在条纹带 [605, 1026) 之外的列数：10  其中对比度达标：6
  这些列的最长连续段：6 列（u 1072–1077），segment_min_columns = 42
  ⚠ 目前连续段短于 segment_min_columns，伪段被侥幸滤掉，但这些列的点被静默
    丢失了。干扰一旦变宽就会产生伪点云。
----------------------------------------------------------------------------------
精测链路（全幅提取+重建）：137.0 ms → 7.3 fps
AOI 链路（AOI 提取+重建）：40.6 ms → 24.6 fps  （提速 3.37×）
```

### 要看的四块

**1. 精测链路 fps** —— 不改任何算法时的实时上限。

| 精测 fps | 结论 |
|---|---|
| ≥ 15 | 全幅即可实时，AOI 只作为余量/正确性手段 |
| 5 ~ 15 | 需要 AOI 才能到 15 fps |
| < 5 | AOI 后仍可能不够，考虑硬件 ROI（同时降 GigE 带宽）或把提取热点向量化/下沉到 C |

**2. AOI 提速倍数** —— 典型 2~3.5×。AOI 是唯一"参数一字不改"的提速手段。

**3. AOI 等价性** —— 按**整数索引轴（列号/行号）配对**比较，不是按数组下标。
下标比较在两边点数不同时会全部错位，在台阶跳变处报出上百像素的假差异（早期版本就踩过这个坑）。
四种判定：

| 判定 | 含义 | 该怎么办 |
|---|---|---|
| `逐点一致` | 完全相同 | AOI 可直接用于精测 |
| `AOI 多出点，没有丢点` | 全幅的整列 argmax 被带外干扰抢走 | **AOI 更可信**；同时说明提取算法需要限制峰值搜索范围 |
| `AOI 少点` | `--aoi-margin` 太小截断了条纹 | 加大 margin |
| `两边各有独有点` | 混合情形 | 按差异样例的 `u`/`v` 去图上逐个核对 |

注意即使判定为"AOI 多出点"，共同列上也可能有零点几像素的偏移——这是段内加权平滑
（`correction_window`）的邻域因为补回了缺失列而改变导致的，属于连带效应而非独立误差。

**4. 整列 argmax 干扰诊断** —— 三种输出：

- `本帧无带外干扰`：这一帧安全。
- `⚠ 连续段短于 segment_min_columns`：伪段被侥幸滤掉，但那些列的点**被静默丢失**。
- `⚠ 最长连续段已达到 segment_min_columns`：**全幅提取正在输出伪点云**，必须立即限制峰值搜索范围。

干扰源通常是激光在台阶侧面/工件表面的二次反射，或视场里另一处饱和亮区。
它随工件、姿态、曝光变化，所以**单帧安全不代表整批安全**——正式验收前应对整个数据集扫一遍。

**5. 背景抑制分解行** —— `GaussianBlur` 通常占全幅耗时的 1/4 到 1/3。`boxFilter` 与下采样方案
快 6–10 倍，但**会改变数值结果**，只能用于 `preview` 档。

## 批量扫一遍数据集（推荐在正式验收前做）

单帧诊断只能说明这一帧。要确认整批数据里有没有"干扰段够宽 → 伪点云"的帧：

```powershell
Get-ChildItem "..\calibration\外参标定\laser_obs\*.tif" | ForEach-Object {
  & "..\.venv\Scripts\python.exe" tools\bench_pipeline.py `
    --image $_.FullName --repeat 1 --skip-variants `
    --json "..\outputs\bench\$($_.BaseName).json"
}
```

然后检查所有 JSON 的 `interference.fake_segment_risk` 字段是否存在 `true`。

## 注意事项

- 结果受 CPU 睿频、后台负载影响。跨机器对比请用**合成条纹**并固定 `--repeat 10` 以上，看**中位数**而不是均值。
- 合成条纹**不含带外干扰**，所以干扰诊断永远显示"本帧无带外干扰"。要看干扰必须用真实图像。
- 用 Mono12 真实图像时记得同步放大 `--min-local-contrast-dn`（例如 8 位用 20 → 12 位用 320），否则会因阈值过低把噪声当峰值，耗时与点数都失真。
- `--scan-axis` 选错时脚本会报"全幅未提取到中心点"并跳过 AOI 测量 —— 这是参数问题，不是性能问题。
- 本脚本只测**单帧**流水线。相机取流、Qt 渲染、3D 点云绘制的开销不在其中，实时模式的端到端帧率会低于这里的数字。
