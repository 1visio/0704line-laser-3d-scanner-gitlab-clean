# 使用与配置说明（USAGE_CONFIG）

单帧线激光高度测量工具的运行方法、配置修改指南与二次开发接口。
**换新标定 / 换新参数时，只需要看第 2 节。**

---

## 1. 如何运行

### 1.1 启动图形界面

PowerShell，在工具目录执行：

```powershell
python main.py                                  # 使用默认配置 configs/measure_tool.yaml
python main.py --config configs/measure_tool.yaml   # 显式指定配置
python main.py --config configs/measure_tool_machine_b.yaml
```

依赖安装（一次性）：

```powershell
pip install -r requirements.txt
```

### 1.2 界面操作流程

1. **加载图像**：选择单通道灰度图（.tif/.tiff/.png/.bmp，支持 Mono8/Mono12）。
2. **提取激光线**：绿色点为亚像素中心，当前只保存在内存中。
   （可先在下拉框切换提取算法；当前配置默认使用 steger。）
3. 添加一个或多个**障碍物区域**（红框）；每个红框按添加顺序独立测量为
   “障碍物 1、2…”。如需逐帧地面校正，可添加多个**基准区域**（蓝框），
   所有蓝框会合并拟合；完全不添加蓝框时使用固定 `Zg=0`。
4. **三维恢复并测量**：重建到地面坐标系，右侧面板显示高度、长度、夹角等；
   图上叠加两条拟合线（青色=基准线，橙色=高度线）。
5. **保存结果**：在输出目录生成 `"图名"_measure[_序号]/`，始终包含
   `laser_center.csv`；完成 ROI 测量后还包含
   `result.json`（含 `obstacles` 逐项结果）、`baseline_points.csv`、
   单障碍物的 `height_points.csv` 或多障碍物的 `obstacle_N_points.csv`、
   `overlay.png`，以及
   整幅激光线补偿后的地面系点云 `full_laser_ground.ply`（含 `Zg` 映射 RGB）。
   未执行 ROI 测量也可保存，此时不生成 baseline/obstacle 点云文件。

### 1.3 运行单元测试

```powershell
cd laser_measurement_tool
$env:PYTHONPATH="$PWD;$PWD\.."       # 同时挂 工具目录 和 仓库根目录
python -m unittest discover -s tests -v
```

---

## 2. 改配置改哪里

**唯一入口：`laser_measurement_tool/configs/measure_tool.yaml`。**
所有相对路径都相对于该 yaml 所在目录（`configs/`）解析。
不想改默认文件时，可复制一份改好后用 `--config` 指定（推荐做法：
每套标定保存一份 yaml，例如 `measure_tool_calib03.yaml`，随时切换）。

### 2.1 换新标定（最常见）

修改 `calibration` 段的三个标定路径；如需地面弯曲补偿，再指定补偿表：

```yaml
calibration:
  intrinsics: calibration/camera_intrinsics.yaml
  laser_model: calibration/circular_cone.yaml
  extrinsics: calibration/camera_ground_extrinsics.yaml
  ground_u_compensation: calibration/ground_u_compensation.csv
```

三个文件接受的格式（加载器会自动识别并校验，单位不对会直接报错）：

| 文件 | 必需字段 | 兼容写法 |
| --- | --- | --- |
| 内参 | 3×3 矩阵 + 畸变(4/5/8/12/14 个) | `camera_matrix`/`K`，`dist_coeffs`/`D` |
| 激光表面模型 | 相机系参数（mm） | 旧平面 `coefficients/plane/plane_abcd`；或 `model_type: global_plane`、`quadratic_graph`、`circular_cone` |
| 地面外参 | 旋转+平移（mm） | `R`(3×3)+`t`(3) 或 `T_ground_from_camera`(4×4) |
| 地面 U 补偿 | `column_u_px`, `bias_mm` | validation v2 的 `ground_bias_table.csv`，或 YAML `sample_table: [[u,bias], ...]` |

圆锥模型至少包含 `axis_unit_camera`、`apex_camera_mm`、
`half_apex_angle_deg`；本实时包的默认文件是
`calibration/circular_cone.yaml`。旧的 `laser_plane` 配置键仍可读取，
便于回放历史平面标定。

补偿在重建到地面坐标后按 `Zg_corrected = Zg_raw - bias(u)` 应用。
表内缺失列按相邻采样线性插值，表范围外使用最近端点值。关闭补偿时将
`ground_u_compensation` 改为 `null`。完整说明见
[`GROUND_U_COMPENSATION_GUIDE.md`](GROUND_U_COMPENSATION_GUIDE.md)。

### 2.2 提取参数（换相机/曝光/激光方向时可能要调）

```yaml
extraction:
  method: steger            # 标定与在线统一使用实时 Steger
  profile: realtime_steger.yaml
  centroid:
    background_kernel: 51             # 背景抑制高斯核，奇数；噪声大调大
    min_local_contrast_dn: 20.0       # 峰值最低对比度（DN）；Mono12 需按位深放大
    centroid_window_radius: 5         # 重心窗口半径
    segment_min_columns: 42           # 有效分段最少列数（过滤杂散短段）
    continuity_max_column_gap: 2      # 分段最大列间隔
    continuity_max_vertical_jump: 14.0  # 分段最大纵向跳变（px）
    correction_window: 7              # 段内平滑窗口，1=关闭
    correction_max_shift: 3.5         # 平滑最大位移（px）
    scan_axis: column   # ★ column=条纹接近水平（laser_obs 数据集）
                        #   row=条纹接近竖直（如老工具截图那种竖线）
  steger:
    {}                       # 实际参数由中央 profile 提供
```

`steger` 通过 Hessian 主曲率和二阶泰勒展开得到亚像素中心；当前标定和在线
测量都调用 `calibration/src/realtime_steger.py`。FWHM 约 2～3 px 的线建议从
`sigma=1.5` 开始，再对照 `1.2/1.8/2.0`。`shared_steger` 只是历史兼容别名。
GUI 下拉框切换算法只对当前会话生效；要改变启动默认值，修改
`extraction.method`。

### 2.3 重建与测量参数

```yaml
reconstruction:
  quadratic_epsilon: 1.0e-12  # 二次求交判据
  min_camera_depth_mm: 600.0    # 当前圆锥标定工作范围（相机系 Zc）
  max_camera_depth_mm: 725.0
  model_range_margin_mm: 10.0   # 模型 z_valid_range_mm 的边界外扩
  image_roi_polygon: null       # 可选：固定姿态棋盘格内部像素四边形 [[u,v], ...]
measurement:
  outlier_sigma_multiplier: 2.0 # 残差>该倍数稳健σ的点剔除；想更宽松调大
  outlier_max_iterations: 5
  min_baseline_points: 20       # 框选点太少时的报错门槛
  min_height_points: 20
output:
  dir: ../output                # 结果输出目录
  save_pointcloud_csv: true
  save_overlay_png: true
  save_full_pointcloud_ply: true  # 整幅激光线的 Xg/Yg/Zg，ASCII PLY，mm
```

### 2.4 高度修正模式（A-12）

高度修正是最终 `height_raw` 标量之后的互斥模式，取值只有
`none`、`h1`、`hb2`。旧配置中的 `stage_a_height_scale` 会兼容映射到
`h1`，但新配置和 GUI 使用 `h1`。H1 与 Frozen H-B2 只会有一个作为
`active_height_correction` 生效；另一个仅写入 shadow logging。

```yaml
correction:
  mode: h1
  stage_a_height_scale_enabled: true
  stage_a_height_scale_config: calibration_daheng_0811/stage_a_height_scale.json
  hb2_height_correction_config: calibration_daheng_0811/hb2_height_correction.json
  hb2_q2_policy: extrapolate_diagnostic  # 域外仍计算，但标记 Stage-B OOD
```

Frozen H-B2 的 `q2` 来自 Frozen-C0 的
`P_c0=lambda_c0*[xn,yn,1]`，不使用 C1 后点、Ground 点或 corrected height。
`reject` 策略在硬域外输出 `HB2_Q2_OOD` 且不返回 H-B2 高度；
`extrapolate_diagnostic` 使用实际 q2 继续计算高度，但仍输出
`HB2_Q2_OOD`、`active_height_valid=false`，界面显示“Stage-B 状态: 超出有效域”。
该值仅作诊断参考，不应视为有效域内的正式标定结果；
`clamp_diagnostic` 则使用域边界截断 q2，并单独标记为诊断 clamp。
在线和单帧结果会记录 `height_raw`、`height_h1`、`height_hb2`、active mode、
q1/q2、q2 domain、v 范围、point count、C1 clamp 和 Ground Reference status。
在线原始帧录制目录还会尽力写入独立的 `height_shadow.csv`；它按已处理结果
保留相机帧号，不假设每个采集帧都有一条重建结果，因此与 `frames.csv`
分开保存。

在线工具还支持可选的 Session 基准标定。开发阶段默认 `optional`，成功后只替换
当前进程的 ground `R/t`，不覆盖 reference 外参 YAML：

```yaml
session_ground_calibration:
  mode: optional              # disabled / optional / required
  pattern_cols: 11
  pattern_rows: 8
  square_size_mm: 20.0
  detector: sb_then_classic
  output: null                # 默认写入 output.dir/session_ground_calibration.json
  quality:
    target_frames: 5
    max_capture_attempts: 8
    max_reprojection_rmse_px: 0.5
    saturation_ratio_warn: 0.05
    dynamic_range_p95_p5_warn: 20.0
    edge_margin_warn_px: 20.0
  sanity:
    mask_enabled: true
    mask_inset_mm: 0.0        # 完整物理边界；0 mm，不做腐蚀/膨胀
    min_valid_points: 20
    max_abs_bias_mm: 2.0
    max_rmse_mm: 2.0
    max_p95_abs_mm: 3.0
    max_abs_mm: 5.0
    max_abs_slope_mm_per_mm: 0.02
```

`required` 模式下必须先连接相机、进入“Session 基准标定”预览，调好曝光后点击“采集 PnP 棋盘格（5 帧）”并获得 `VALID`，再开始
在线重建；`disabled` 始终使用 reference。在线结果 JSON 会记录
`ground_extrinsic_source: reference/session`。

点击“激光地面一致性检查”前保持棋盘不动并打开激光。该检查会复用 Session PnP 的
pose、内参和畸变，投影完整棋盘物理边界生成 mask，先筛选源像素位于 mask 内的
重建点，再统计当前正式链路的原始 `Zg`，默认至少 20 个有限点；阈值超限时报警并写入
`session_ground_calibration.json.laser_ground_sanity`，不会自动做 bias offset、
`a*S+b` 拟合或 Surface correction。检查要求当前在线处理算法为 `steger`，并且
`reconstruction.enable_laser_ray_correction: true`。

`image_roi_polygon` 是在线重建前的像素门控。启用时，只有多边形内部的激光
中心点才会进入射线-激光表面求交，结果中的 `filtered.outside_image_roi` 会记录
被丢弃的点数。它适合棋盘格姿态固定、需要只观察棋盘格内部的验证场景，例如：
原始提取 overlay 仍保留全线用于诊断，但三维点、测量结果和完整 PLY 只包含
ROI 内且通过重建约束的点。

```yaml
reconstruction:
  image_roi_polygon:
    - [420, 260]
    - [2010, 290]
    - [2070, 1760]
    - [390, 1730]
```

坐标必须是**原始图像**像素，顶点按顺时针或逆时针填写，边界点包含在内。
棋盘格姿态改变后应重新测量四个外角并更新配置；该固定多边形不会自动跟踪
棋盘格。若要动态跟踪，应另采一张高曝光、关闭激光的棋盘格图像做角点检测，
再把当帧四边形传给重建流程。补偿建表则应使用独立的平地扫描，不要把这里的
棋盘格 ROI 当成 `ground_u_compensation` 的输入筛选条件。

在线 GUI 的 `calibration.manifest` 仍然必须存在并校验文件哈希。几何标定刚完成、
尚未建立补偿表时，可以在临时 smoke-test 包的 manifest 中写
`files.ground_u_compensation: null`；这只表示关闭补偿，不代表该包已经通过平地
补偿验收。生产包仍应填入真实的 `ground_bias_table` 并通过独立验证。
当 `manifest` 非空时，manifest 是唯一标定来源；配置中的三个路径以及
`calibration.ground_u_compensation` 不会覆盖 manifest。也就是说，仅把
`measure_tool.yaml` 的补偿项改成 `null`，并不能关闭 manifest 里仍然指向的旧 CSV。

---

## 3. 提取与二次开发接口

### 3.1 使用 Steger

配置启动默认算法：

```yaml
extraction:
  method: steger
```

也可以直接调用后端：

```python
from laser.backends import steger_backend

points_uv = steger_backend(image, {
    "sigma": 1.5,
    "threshold": 30.0,
    "deriv_thresh": 0.5,
    "roi_margin": 120,
    "roi_max_height": 512,
    "scan_axis": "column",
})
```

实现位于共享的 `calibration/src/realtime_steger.py`，标定工具与在线测量直接
调用同一个模块；其条纹带裁剪可限制大图 Hessian 的内存和耗时。
依赖 `scipy`；安装命令仍为 `python -m pip install -r requirements.txt`。

### 3.2 脚本化调用（不开界面批处理）

核心函数全部无 Qt 依赖，可直接组合：

```python
from calibration.config_loader import load_calibration_files
from laser.backends import centroid_backend
from reconstruction.reconstructor import reconstruct_uv_to_ground, ReconstructionParams
from measurement.height_measure import measure_height_line, MeasurementParams
from utils.image_io import load_grayscale_image

calib = load_calibration_files(
    intrinsics=...,
    laser_plane=...,
    extrinsics=...,
    ground_u_compensation="ground_bias_table.csv",
)
points_uv = centroid_backend(load_grayscale_image("frame.tif"), options)
recon = reconstruct_uv_to_ground(points_uv, calib, ReconstructionParams())
result = measure_height_line(baseline_ground, height_ground, MeasurementParams())
# result.height_mean_mm / .length_mm / .angle_with_baseline_deg / .endpoints_ground
```

数据契约：像素点一律 `(N,2)` 的 `(u,v)`（OpenCV 亚像素约定）；
三维点一律 `(N,3)` mm；标定字典固定含
`K, D, plane_abcd, R, t, ground_u_compensation`。

### 3.3 关键模块速查

| 功能 | 文件 | 入口 |
| --- | --- | --- |
| 配置加载 | `app_config.py` | `load_app_config(path)` |
| 标定加载（任意路径） | `calibration/config_loader.py` | `load_calibration_files(...)` |
| 提取算法注册 | `laser/backends.py` | `AVAILABLE_METHODS` / `create_extraction_params` |
| 三维重建 | `reconstruction/reconstructor.py` | `reconstruct_uv_to_ground` |
| 高度/长度测量 | `measurement/height_measure.py` | `measure_height_line` |
| 结果落盘 | `utils/result_io.py` | `save_measurement_json` 等 |
| GUI 装配 | `gui/main_window.py` | `MainWindow(config)` |

### 3.4 与 v3 批处理脚本的关系

提取与重建算法从 `reconstruct_ground_pointcloud_v3.py` 移植，
已用真实帧验证**逐点位精确一致**（2026-07-26，
`calibration/外参标定/laser_obs/Image_20260722140055330.tif`，2435 点）。
差异仅在：本工具用手动 ROI 替代 v3 的 Zg 分位数自动阈值来区分地面/障碍物。

### 3.5 两种地面参考模式

| 模式 | 触发方式 | 高度基准 | 可用结果 |
| --- | --- | --- | --- |
| 现场基准 | 同时选择基准与障碍物 ROI | 基准点稳健中位数 | 高度、长度、地面噪声、与基准线夹角 |
| 固定零基准 | 只选择障碍物 ROI | 固定 `Zg=0 mm` | 高度、长度、高度线 RMSE；地面噪声和基准夹角显示 `—` |

如果已经添加基准 ROI、但框内没有有效激光点，工具会提示调整或删除该 ROI，
不会静默切换为固定零基准模式。

### 3.6 多 ROI 规则

- 多个基准 ROI：点集取并集，共同估计一个地面 `Zg` 和一条基准方向。
- 多个障碍物 ROI：不取并集，按红框添加顺序分别重建和拟合。
- 图中红框标记“障碍物 1、2…”，与右侧结果分组、JSON 数组及 CSV 编号一致。
- 单障碍物继续保存 `height_points.csv`；两个及以上保存
  `obstacle_1_points.csv`、`obstacle_2_points.csv` 等。

---

## 4. 常见问题

- **启动即弹"配置加载失败"**：yaml 路径或字段错误；报错信息里有具体字段名。
  工具仍可提取激光线，但"三维恢复"需要有效配置。
- **"标定加载失败"**：三个标定文件路径不对或格式/单位不符合第 2.1 节表格。
- **提取不到点 / 点很少**：确认 `scan_axis` 与激光方向匹配；再调低
  `min_local_contrast_dn` 或 `segment_min_columns`。
- **测量报"点数不足"**：ROI 框内点少于 `min_baseline_points`/`min_height_points`；
  扩大框选范围或在配置里调低门槛。
- **换了图像分辨率/相机**：无需改工具，换对应内参文件即可；
  Mono12 时记得放大 `min_local_contrast_dn`。
