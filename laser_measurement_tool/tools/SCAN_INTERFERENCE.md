# scan_interference.py · 带外干扰批量扫描

对整个目录逐帧检测"整列 `argmax` 被更亮的带外特征抢走"的风险，输出逐帧 CSV 与汇总 Markdown。

- 放置位置：`laser_measurement_tool/tools/scan_interference.py`
- 依赖：仓库 `.venv` 已有的 `numpy` / `opencv-python`，无需新装包
- 不修改任何输入文件；只在 `--csv` / `--report` / `--json` 指定的位置写结果
- **退出码：检出 `fake_segment_risk` 时返回 1**，可直接用于采集流程的自动准入判断

## 为什么需要它

`laser/backends.py`、`reconstruct_ground_pointcloud_reusable.py`、`src/.../mono.py` 三处提取
都用**整列最亮点**定位条纹：

```python
peak_rows = np.argmax(signal, axis=0)   # 在整列里找最亮
```

视场里任何比条纹更亮的东西（激光在桌面/工件上的镜面高光、二次反射、另一处饱和亮区）
都会抢走峰值。后果分两级：

| 干扰连续段宽度 | 后果 | 可见性 |
|---|---|---|
| < `segment_min_columns`（默认 42） | 那些列被丢弃，点云少点 | **完全静默**，无日志无警告 |
| ≥ `segment_min_columns` | 输出**一整段位置错误的伪点云** | **完全静默** —— 伪点自身连续一致，射线求交、地面变换、异常过滤、障碍物拟合全都不报错 |

干扰是**装配相关**的：换一次装配、换工件、换角度，它就换位置换宽度。
所以"上一批数据没问题"不代表这批没问题 —— 这个脚本要按批跑。

## 快速开始

```powershell
Set-Location "G:\dev\projects\0704linescan\laser_measurement_tool"

# 扫一个目录
& "..\.venv\Scripts\python.exe" tools\scan_interference.py "..\calibration\外参标定\laser_obs"

# 扫多个目录 + 出 CSV 和报告
& "..\.venv\Scripts\python.exe" tools\scan_interference.py `
  "..\calibration\外参标定\laser_obs" `
  "..\calibration\外参标定\laser_board" `
  "..\calibration\calib03" `
  --csv "..\reports\interference_scan.csv" `
  --report "..\reports\interference_scan_report.md"

# 采集后自动准入（检出伪段风险则退出码非 0）
& "..\.venv\Scripts\python.exe" tools\scan_interference.py "<本批采集目录>" --quiet
if ($LASTEXITCODE -ne 0) { Write-Host "检出伪段风险，先处理现场再继续" -ForegroundColor Red }
```

## 参数说明

### 输入

| 参数 | 默认 | 说明 |
|---|---|---|
| `inputs`（位置参数，可给多个） | 必填 | 图像文件或目录。目录默认只扫一层 |
| `--recursive` | 关 | 目录递归查找（深度不限） |

识别的扩展名：`.tif .tiff .png .bmp .jpg .jpeg`。重复路径自动去重。

### 提取参数（**必须与实际使用的提取配置一致**）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--method {centroid}` | centroid | 提取算法，取值来自 `laser/backends.py` 的 `AVAILABLE_METHODS` |
| `--background-kernel N` | 51 | 背景抑制高斯核（≥3 的奇数） |
| `--min-local-contrast-dn F` | 20.0 | 峰值最低局部对比度（DN，按源位深）。**Mono12 图像要按位深放大**（8 位 20 → 12 位约 320） |
| `--segment-min-columns N` | 42 | **伪段判定阈值**。必须与实际提取配置一致，否则判定失真 |
| `--continuity-max-column-gap N` | 2 | 连续段允许的最大列间隔 |
| `--scan-axis {column,row}` | column | `column`=条纹接近水平；`row`=条纹接近竖直 |

上面这些默认值与 `configs/measure_tool.yaml` 的 `extraction.centroid` 段一致。**改了配置就要同步改这里。**

### 检测参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cluster-gap N` | 200 | 簇切分间隔（像素）。**唯一需要按场景调的参数**，见下 |

`cluster_gap` 决定"多远算另一处特征"：

- **要大于条纹自身的合法跨度**（含台阶跳变）。一个 55 mm 台阶在 0.367 mm/px 下跳变约 150 px，所以 200 px 是安全的下限。台阶更高就要加大。
- **要小于干扰与条纹的距离**。本项目实测干扰在 v≈1357、条纹在 v≈870，相距 480 px，200 px 足够区分。
- 设得太小 → 把同一条条纹的台阶切成两簇，误报；设得太大 → 把干扰并进主簇，漏报。
- 拿不准时先跑一遍看 `cluster_count`：正常帧应该是 1（无台阶）或 1（台阶在同簇内）。持续出现 2 且第二簇很小，就是干扰。

### 输出

| 参数 | 说明 |
|---|---|
| `--csv PATH` | 逐帧结果 CSV（`utf-8-sig`，Excel 直接打开不乱码） |
| `--report PATH` | 汇总 Markdown 报告，按"伪段风险 / 输出已分裂 / 静默丢点 / 失败"分节列表 |
| `--json PATH` | 完整结果 JSON，含扫描参数。适合程序化检查 `frames[].fake_segment_risk` |
| `--quiet` | 不打印逐帧进度，只打印汇总 |

## 输出怎么读

### 逐帧进度行

```
[   1/11] · 静默丢点  Image_20260722140055330.tif  点数=2435  带外列=6  最长段=6  簇数=2
[   5/31] ⚠ 伪段风险  Image_xxx.tif                点数=2489  带外列=68 最长段=61 簇数=2
[   2/12]   正常      laser 003.tif                点数=1850  带外列=0  最长段=0  簇数=1
```

| 标记 | 含义 | 处理 |
|---|---|---|
| `正常` | 无带外候选峰 | 无需处理 |
| `· 静默丢点` | 有带外峰，但最长连续段 < `segment_min_columns` | 点云少了点但位置都对。可接受，但说明现场有干扰源 |
| `⚠ 伪段风险` | 最长连续段 ≥ `segment_min_columns` | **这一帧的点云可能含整段错误点**。先处理现场，再考虑软件限带 |

### CSV 关键列

| 列 | 含义 |
|---|---|
| `extracted_points` | 实际提取到的点数 |
| `candidate_columns` | 对比度达标的列数（潜在有效列） |
| `cluster_count` | 候选峰聚成几簇。> 1 就有别的亮特征 |
| `main_cluster_columns` / `main_cluster_range` | 主簇（真实条纹）的列数与位置范围 |
| `outside_columns` | 峰落在主簇之外的列数 |
| `outside_longest_run` / `outside_longest_run_range` | 带外列的最长连续段长度与列范围 |
| `outside_longest_run_cluster_range` | 该连续段所属簇的**位置范围** —— 拿这个去图上找干扰源 |
| `fake_segment_risk` | `True` / `False`，核心判定 |
| `output_cluster_count` / `output_split_detected` | 最终输出点云本身分成几簇。**分裂 ≠ 一定有问题**，高台阶本身就会分簇，要对照 `cluster_gap` 判断 |

### 定位干扰源

拿到 `outside_longest_run_range`（列范围）和 `outside_longest_run_cluster_range`（位置范围），
直接去原图那块区域看：

```powershell
& "..\.venv\Scripts\python.exe" -c @"
import numpy as np, cv2
img = cv2.imdecode(np.fromfile(r'<图像路径>', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
print(img[1350:1370, 1065:1085])   # 换成你的 v 范围 / u 范围
"@
```

再用 `nolaser`（激光关闭）帧查同一位置：

- 激光关时也亮 → 传感器热像素 / 坏点 / 镜头脏污 → 需要坏点表或清洁
- 激光关时全黑 → **激光的高光或二次反射** → 现场遮光 / 改角度 / 消除反光面最有效

## 检测方法（为什么不需要预先知道条纹在哪）

1. 复刻 `_extract_columnwise` 的前两步：`GaussianBlur` 背景抑制 → 逐列 `argmax` → 对比度筛选；
2. 把候选峰按**位置**排序，间隔 > `cluster_gap` 处切开，得到若干簇；
3. **列数最多的簇视为真实条纹**，其余为带外候选；
4. 对每个带外簇统计它在扫描轴上的最长连续段，≥ `segment_min_columns` 即判定 `fake_segment_risk`；
5. 另外跑一次真实提取，检查**输出点云本身**是否已分裂成多簇 —— 这是"伪段已进入结果"的直接证据。

这样不依赖任何预设 AOI，因此对**已经被污染的帧**同样有效。
（如果先用全幅提取结果推 AOI，被伪段污染的帧其 AOI 会把干扰包进去，从而漏报 —— 这是必须避开的循环依赖。）

## 注意事项

- `--segment-min-columns` 与实际提取配置不一致时，判定完全失真。改配置务必同步。
- Mono12 图像要放大 `--min-local-contrast-dn`，否则会把噪声当候选峰，`outside_columns` 虚高。
- `output_split_detected` 为真不一定是问题 —— 请对照 `cluster_gap` 与实际台阶高度判断。
- 本脚本只检测"更亮的带外特征"。**比条纹暗但仍在条纹带内**的干扰（例如条纹旁的宽晕）它检不出，那类问题需要看 FWHM 与对比度分布。
- 扫描耗时约等于每帧一次全幅提取（本项目约 45 ms/帧 @ 用户机，135 ms/帧 @ 云端容器），31 帧数秒完成。
