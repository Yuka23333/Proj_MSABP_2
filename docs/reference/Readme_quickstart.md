# CFSA 项目 Quickstart

这份文档面向“隔一段时间重新打开项目，怎样在不误跑 CST、不破坏编号、不混用旧脚本的前提下迅速恢复工作”的场景。

它记录当前项目从二维参数化几何，到 CST/HFSS 建模、DoE、分类器、VAE、qLogEHVI 和后分析的最短上手路径。更完整的模块说明见：

- [项目主 README](README.md)
- [纯几何核心](cfsa_geometry/README.md)
- [CST 批处理与 BO](Automation_scripts/README.md)
- [HFSS 几何复现](../HFSS_workspace/README.md)

> 本文核对日期：2026-07-29。运行默认值会继续变化；文件顶部的 `F5_*` 常量、扫描 metadata 和锁定 worklist 才是当前事实。

## 1. 一页技术地图

```text
Fractal_Antenna.DEFAULT_CONFIG
  48 个数值参数：45 个优化变量 + 3 个固定变量
                 |
                 v
cfsa_geometry
  dataclass 配置 -> 6 条闭合二维曲线
                 |
                 v
postprocess
  substrate 底边对齐 y=0
  0.005 mm 量化
  连续共线点吞点
                 |
                 v
validation
  Shapely 单曲线自交检查
  SciPy cKDTree 全局最小点距检查
                 |
        +--------+--------+
        |                 |
        v                 v
QMC / DoE            CAD 适配层
Latin / Sobol        CST 或 HFSS
多进程筛选               |
CSV + metadata            v
        |            建模 / 求解 / 导出
        v                 |
固定可行 worklist <-------+
稳定 case_id              |
                          v
                   S11 / 效率 / farfield
                          |
             +------------+------------+
             |                         |
             v                         v
       LightGBM 可行性门控         BoTorch / GPyTorch
       Grouped VAE / PCA / UMAP    qLogEHVI 顺序优化
                                      |
                                      v
                           IMSE / Hypervolume /
                           Pareto onion / 范围分析
```

项目里最重要的边界是：

- `cfsa_geometry/` 只算几何，不连接 CST 或 HFSS。
- `scan_geometry_feasible_domain.py` 只做纯 Python 可行域扫描，不调用 CST。
- CST/HFSS 适配层负责把同一套二维曲线变成 CAD 实体。
- 正式 CST 批跑入口只有 `Automation_scripts/doe_run_and_export.py`。
- `scripts/doe_run_and_export.py` 是旧脚本，不属于当前生产流程。
- Field Monitor、端口、边界和 solver 配置属于模板工程；当前自动化只校验，不创建。

## 2. 先确认运行上下文

所有命令默认从项目根目录运行：

```powershell
Set-Location D:\Academic\Proj_CFSA
```

先看代码里实际生效的 F5 配置：

```powershell
rg -n "^F5_" `
  scripts\scan_geometry_feasible_domain.py `
  scripts\Automation_scripts\doe_run_and_export.py `
  scripts\Automation_scripts\bayesian_optimization_smoke.py `
  scripts\Automation_scripts\bayesian_optimization_run_200.py `
  HFSS_workspace\create_cfsa_geometry.py
```

不要根据旧日志、旧 README 或文件名猜当前样本数、起始编号、频率范围和输出目录。

### 2.1 Python 能力分组

纯几何最小依赖：

```text
numpy scipy shapely
```

绘图、CSV 和传统机器学习：

```text
matplotlib pandas scikit-learn joblib lightgbm xgboost
```

VAE、降维和 BO：

```text
torch tqdm umap-learn botorch gpytorch ninja
```

CAD 连接：

```text
CST:  cst.interface
HFSS: pywin32，或 AEDT 安装目录中的 ScriptEnv/gRPC API
```

先按任务检查，不要一次性在 CST 环境里盲目升级全部包：

```powershell
python -c "import numpy, scipy, shapely, pandas; print('geometry/data ok')"
python -c "import torch, botorch, gpytorch, lightgbm; print('BO stack ok')"

& 'C:\Users\David\.conda\envs\cstpy\python.exe' `
  -c "import cst.interface; print('CST API ok')"
```

当前已验证的 CST 解释器是：

```text
C:\Users\David\.conda\envs\cstpy\python.exe
```

截至本文核对时，系统 Python 有 BoTorch 但缺 LightGBM，`cstpy` 有 LightGBM 和 `cst.interface` 但缺 BoTorch。因此正式 BO 前必须先选定一个同时能 import `cst.interface`、`lightgbm`、`torch`、`botorch` 和 `gpytorch` 的解释器。不要等 200 点生产入口启动后才发现依赖分裂。

安装 PyTorch/CUDA、BoTorch 或 AEDT/CST Python 接口前，应查对应版本的官方安装说明；不要把普通 PyPI 上同名但来源不明的包当作 CAD 官方接口。

### 2.2 最小验证梯度

按下面的梯度逐级验证。前一级失败时不要启动后一级：

1. `py_compile`：语法和 import 边界。
2. 单元测试：纯几何、参数契约、点序和 CST VBA 文本。
3. `--dry-run`：生成计划，不连接 CAD。
4. 1 个真实 CAD case：建模、求解、导出闭环。
5. 2 个连续 case：验证删除旧模型、编号和 resume。
6. 小批次。
7. 正式 DoE 或 BO。

项目回归测试：

```powershell
python -m unittest discover -s scripts\tests -v
```

## 3. 参数化几何

### 3.1 最小 Python API

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import Fractal_Antenna as antenna
from cfsa_geometry.model import curve_data_from_geometry

config = antenna.make_geometry_config(
    substrate_thickness=1.6,
    feed={"width": 3.4},
)
geometry = antenna.build_antenna_geometry(config)
curve_data = curve_data_from_geometry(geometry)

print(curve_data["format"])
print(curve_data["curve_count"])
print([curve["name"] for curve in curve_data["curves"]])
```

`make_geometry_config()` 返回新 dataclass，不修改 `DEFAULT_CONFIG`。批量脚本应传 `config`，不要运行时修改模块全局字典。

按参数路径修改一个叶节点：

```python
from cfsa_geometry.config import config_with_parameter_value

config = config_with_parameter_value(
    antenna.DEFAULT_CONFIG,
    "inner.y_min",
    1.2,
)
```

按 CSV 列名构造配置：

```python
from Automation_scripts.fractal_model import (
    DEFAULT_PARAMETER_VALUES,
    PARAMETER_NAMES,
    config_from_parameter_values,
)

# 正式 CSV/worklist 行必须覆盖完整优化参数集；名字决定映射，不依赖列顺序。
names = list(PARAMETER_NAMES)
values = list(DEFAULT_PARAMETER_VALUES)
values[names.index("substrate_thickness")] = 1.6
values[names.index("feed.width")] = 3.4
config = config_from_parameter_values(names, values)
```

这个入口按名字映射，并会补入固定参数和兼容默认值。缺少普通优化列会直接报错，不会静默拿默认值填满一个不完整的新样本。

### 3.2 六条曲线

当前导出契约是 `fractal_antenna_curves_v1`，包含：

```text
outer_rectangle
outer_third_order_minkowski_frame
feed_rectangle
feed_pin
outer_second_order_minkowski
inner_second_order_minkowski
```

每条 `path` 都应闭合，即最后一点等于第一点。几何坐标单位是 mm，原始 JSON 不保证 substrate 底边已经是 `y=0`；后处理和 CAD 适配层会做这一步。

F5 交互预览：

```powershell
python scripts\Fractal_Antenna.py
```

静态绘图并导出 `fractal_antenna_curves.json`：

```powershell
python scripts\Fractal_Antenna.py --static
```

### 3.3 制造网格后处理

```python
from cfsa_geometry.postprocess import postprocess_curve_data

processed = postprocess_curve_data(
    curve_data,
    grid_size=0.005,
    collinear_tolerance=1e-10,
)
print(processed["postprocess"])
```

处理顺序固定为：

1. 以 `outer_rectangle` 底边为锚点，把全部点平移到 `y=0`。
2. 将 x/y 量化到 `0.005 mm` 网格。
3. 检查闭合路径上的连续三点，吞掉位于同一直线段上的中点。
4. 重新计算 `closed`、`path_count` 和 `point_count`。

不要先分别量化每条曲线再各自对齐，否则曲线之间的相对位置会发生变化。

### 3.4 几何可行性

```python
from cfsa_geometry.validation import (
    check_curve_self_intersections,
    check_min_point_distance,
)

self_report = check_curve_self_intersections(processed)
distance_report = check_min_point_distance(processed, min_distance=0.01)

feasible = (
    self_report["is_feasible_by_self_intersection"]
    and distance_report["is_feasible_by_min_distance"]
)
print(feasible)
```

判定含义：

- Shapely `LinearRing.is_simple` 和 `Polygon.is_valid` 检查每一条闭合曲线自身是否打结或自交。
- 四个主要环形结构之间不做互相求交；它们由 margin 设计隔离。
- `cKDTree` 检查全部曲线顶点的全局最小点距。
- 阈值比较发生在底边对齐、制造网格量化和吞点之后。
- `单曲线自交可行性=True` 表示当前样本通过检查，不表示参数空间中永远不可能自交。

## 4. QMC DoE 与可行域

### 4.1 参数注册表

`cfsa_geometry/parameters.py` 是优化变量的唯一注册表。当前配置有 48 个数值参数，其中 45 个参与优化，3 个固定：

```text
minkowski_frame.order
feed.center_x
feed_pin.top_inner_x
```

当前扫描分类及范围规则：

| 类别 | 当前范围 |
| --- | --- |
| `margin_delta` | 初始值的 50% 到 300% |
| `scale` | 初始值的 0% 到 150% |
| `other` | 初始值的 50% 到 300% |
| `feed` | 初始值的 80% 到 120% |
| `substrate_thickness` | 绝对范围 0.7 到 5.0 mm |

每次扫描都会把实际上下界、变量类别、结构组、固定值和种子写入 `.metadata.json`。后续分类、VAE 和 BO 应读取这份 metadata，不要重新手抄上下界。

### 4.2 最小扫描

```powershell
python scripts\scan_geometry_feasible_domain.py `
  --samples 16 `
  --method latin `
  --workers 1 `
  --seed 20260729 `
  --include-reason `
  --output results\quickstart_geometry_scan.csv
```

确认小样本正常后再使用 24 个进程：

```powershell
python scripts\scan_geometry_feasible_domain.py `
  --samples 4096 `
  --method latin `
  --workers 24 `
  --output scripts\scan_latin_4096_new.csv
```

Python API：

```python
import scan_geometry_feasible_domain as scanner

samples = scanner.generate_qmc_samples("sobol", 1024, seed=20260729)
first = scanner.evaluate_sample(0, samples[0])
print(first["geometry_feasible"], first["substrate_area_mm2"])
```

方法选择：

- Latin Hypercube：任意样本数都方便，适合一次性覆盖。
- scrambled Sobol：优先使用 `2**m` 个点，以保留低差异序列结构。
- 固定 seed 用于复现；`seed=None` 表示每次重新随机。
- 几何检查是 CPU 任务，使用 `ProcessPoolExecutor`，不连接 CST。

### 4.3 线性缩圈

```powershell
python scripts\shrink_feasible_domain_ranges.py `
  --metadata scripts\scan_latin_16384.csv.metadata.json `
  --points scripts\scan_latin_16384.csv `
  --ranges-csv
```

它计算包含全部可行点的最小轴对齐超矩形。输出中的：

```text
shrink_ratio = 新范围宽度 / 原范围宽度
```

因此它是保留率，不是“减少了多少”。接近 1 表示线性缩圈节省不了多少空间；它不能描述变量耦合或弯曲流形。

## 5. CST 连接与建模

### 5.1 模板工程前提

默认工程：

```text
D:\Academic\Proj_CFSA\cst_test_proj\test.cst
```

模板中必须预先存在：

- `Copper (annealed)`、`FR-4 (lossy)` 等脚本使用的材料名；
- 至少一个端口；
- 边界、背景、网格和 solver 设置；
- 与本次 `FrequencyRange` 完全一致的 Field Monitor。

当前正式 DoE 的 F5 频率是 `[2:0.3:14] GHz`，共 41 点。自动化不会新建 monitor。改频率时需要同时修改 CST 模板和 Python 配置。

`FrequencyRange.values()` 要求 `(stop-start)/step` 能整除，否则会在连接 CST 前报错：

```python
from Automation_scripts.cst_automation import FrequencyRange

frequencies = FrequencyRange(2.0, 0.3, 14.0).values()
assert len(frequencies) == 41
```

### 5.2 最小连接

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from Automation_scripts.cst_automation import (
    FrequencyRange,
    open_project,
    validate_project_prerequisites,
)

environment, project = open_project("cst_test_proj/test.cst")
info = validate_project_prerequisites(
    project,
    FrequencyRange(2.0, 0.3, 14.0),
)
print(info)
```

`open_project()` 会连接已有 CST Design Environment，复用已打开的同一工程，或打开指定 `.cst`。

### 5.3 最小多边形和 Brick

```python
from cst_generate_polygen import create_brick, create_extruded_polygon

# XoY 平面逆时针点序；正 thickness 沿 +Z 拉伸。
points = [
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, -1.0),
]

create_extruded_polygon(
    project,
    points,
    polygon_name="demo_polygon",
    curve_name="demo_curve",
    solid_name="demo_solid",
    material_name="Copper (annealed)",
    thickness=0.035,
    save_project=False,
    timeout=60,
)

create_brick(
    project,
    solid_name="demo_brick",
    material_name="PEC",
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    z_range=(-1.0, 0.0),
    save_project=False,
    timeout=60,
)
```

低层 `create_extruded_polygon()` 不替调用方判断物理朝向。CST `.Thickness` 遵循点序右手定则：

- XoY 平面逆时针 + 正厚度：向 `+z`；
- 顺时针 + 正厚度：向 `-z`。

完整 CFSA 导入会对每条曲线单独确保逆时针，不应通过随意翻转 thickness 修补点序问题。

### 5.4 完整模型重建

```python
from Automation_scripts.fractal_model import (
    default_config,
    rebuild_fractal_model,
)

config = default_config()
rebuild_fractal_model(
    project,
    config,
    command_timeout=60,
    save_project=True,
)
```

这个函数会：

1. 删除脚本管理的旧天线同名实体。
2. 重新计算六条曲线。
3. 创建基板、反射底板、切口工具体和六个铜层/辅助实体。
4. 执行四次布尔差集。
5. 可选保存工程。

默认层叠：

| 结构 | 材料 | z 范围 |
| --- | --- | --- |
| 顶层铜 | `Copper (annealed)` | `0` 到 `+0.035 mm` |
| 基板 | `FR-4 (lossy)` | `0` 到 `-substrate_thickness` |
| 反射底板 | 铜 | 基板背面再向 `-z` 0.035 mm |

只看计划、不打开 CST：

```powershell
python scripts\cst_import_fractal_antenna.py --dry-run
python scripts\cst_boolean_subtract.py --dry-run
python scripts\cst_delete_fractal_solids.py --preset boolean --dry-run
```

手工重建调试：

```powershell
$cstpy = 'C:\Users\David\.conda\envs\cstpy\python.exe'

& $cstpy scripts\cst_delete_fractal_solids.py --preset boolean
& $cstpy scripts\cst_import_fractal_antenna.py
& $cstpy scripts\cst_boolean_subtract.py
```

普通 VBA/CST 命令默认最多等待 60 秒；solver 同步运行且没有这个短超时。不要因为 solver 超过一分钟就强制终止 CST。

## 6. CST DoE 批处理

### 6.1 先 dry-run

显式指定 CSV、起点和数量，避免受当前 F5 起始编号影响：

```powershell
$cstpy = 'C:\Users\David\.conda\envs\cstpy\python.exe'

& $cstpy scripts\Automation_scripts\doe_run_and_export.py `
  --csv-file scripts\scan_latin_4096_optimization_v2.csv `
  --start-case-id 0 `
  -n 2 `
  --dry-run
```

最小真实闭环：

```powershell
& $cstpy scripts\Automation_scripts\doe_run_and_export.py `
  --csv-file scripts\scan_latin_4096_optimization_v2.csv `
  --start-case-id 0 `
  -n 1 `
  --farfield 2 0.3 14 `
  --out-dir results\quickstart_cst_smoke
```

一次 case 的顺序是：

```text
清空旧结果
-> 删除旧几何
-> 重新生成完整模型
-> 校验端口和 Field Monitor
-> 同步运行 solver
-> 导出结果
-> 保存参数快照和状态
```

### 6.2 固定 worklist

首次选择带 `geometry_feasible` 标签的扫描 CSV 时，会生成：

```text
<source_stem>.feasible_<id_width>d.csv
<source_stem>.feasible_<id_width>d.csv.metadata.json
```

worklist 保存：

- 稳定 `case_id`；
- 原始 sample index；
- 原始 CSV 行号；
- 可行样本序号；
- 45 个优化变量；
- 源 CSV SHA-256 和编号宽度。

已有 worklist 默认原样复用。`--rebuild-worklist` 会重新定义 case_id 到参数的映射；只要这个批次已经产生结果，就不能自行重建。

四位编号容量是 10000 个 case。超过时应在第一次锁表时使用 `--id-width 5`，而不是运行中途改宽度。

### 6.3 输出、续跑和熔断

每个 `case_XXXX/` 默认包含：

```text
S11_XXXX.csv
RadEff_XXXX.csv
ToTEff_XXXX.csv
Farfields/farfield_<frequency>GHz.txt
parameters.csv
case_metadata.json
```

批次目录还包含：

```text
run_results.csv
bad_points.csv
fatal_consecutive_failures.json   # 仅触发熔断时
```

恢复规则：

- `resume=True` 时跳过 `run_results.csv` 中已经 `ok` 的 case。
- 默认跳过 `bad_points.csv` 中的普通坏点。
- 连续 5 个 case 全部失败时停止整个程序。
- 触发熔断的 5 个点不进入坏点表，以便修复系统性故障后重试。
- 不要手工删除单个导出文件并期待 resume 自动重跑；resume 以 ledger 状态为准。

连续失败通常说明模板、连接、材料、monitor 或 CST 状态出了系统性问题，不应把后续数千点继续送进去。

## 7. HFSS/AEDT 连接与建模

HFSS 路径使用与 CST 相同的六条二维曲线，但采用 AEDT covered polyline、沿 Z 扫掠、Box 和 Boolean API。

它只创建几何和材料，不创建端口、边界、空气盒、setup、sweep 或 field monitor。

### 7.1 dry-run

```powershell
python HFSS_workspace\create_cfsa_geometry.py --dry-run
python HFSS_workspace\create_sma_connector.py --dry-run
```

### 7.2 最小连接

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("HFSS_workspace").resolve()))

from create_cfsa_geometry import connect_aedt

desktop, backend = connect_aedt(backend="auto", grpc_port=50051)
print(backend)
```

连接策略：

1. 若在 AEDT 内运行并已有 `oDesktop`，直接复用。
2. `auto` 先尝试 `ScriptEnv`。
3. 再尝试 `win32com.client` COM。
4. `backend="grpc"` 时连接指定 gRPC 端口。

Python 与 AEDT 必须使用同一 Windows 用户和相同权限级别。普通权限 Python 无法可靠连接管理员权限 AEDT，反之亦然。

### 7.3 创建 CFSA

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
sys.path.insert(0, str(Path("HFSS_workspace").resolve()))

import Fractal_Antenna as antenna
from create_cfsa_geometry import create_cfsa_in_hfss

config = antenna.make_geometry_config(substrate_thickness=1.6)
summary = create_cfsa_in_hfss(
    config=config,
    project_path=Path("HFSS_workspace/HFSS_test.aedt"),
    project_name="HFSS_test",
    design_name="HFSSDesign1",
    backend="auto",
    delete_existing=True,
    save_project=True,
)
print(summary)
```

直接运行：

```powershell
python HFSS_workspace\create_cfsa_geometry.py
```

脚本只删除自己管理的同名实体，不清空整个设计。几何会逐曲线确保逆时针，并把基板底边统一平移到 `y=0`。

### 7.4 简化 SMA

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("HFSS_workspace").resolve()))

from create_sma_connector import create_sma_in_hfss

create_sma_in_hfss(
    project_path=Path("HFSS_workspace/HFSS_test.aedt"),
    project_name="HFSS_test",
    design_name="HFSSDesign1",
)
```

当前 SMA 约束：

- 轴向为 Y；
- 内导体直径 1.27 mm；
- Teflon 外径/外导体内径 4.13 mm；
- 外导体外径 6.35 mm，长度 5 mm；
- 内导体最低点为 `z=0`；
- 9 mm 方形 xOz 法兰沿 `y-` 厚 1.27 mm；
- 两侧地柱自动贴当前 CPW 地边；
- Teflon 材料名默认 `teflon_based`。

AEDT 可能自动 N-body Unite 相接触的同材料实体，导致某个活动对象名消失但体积仍存在。此时先检查最终体积和对象列表，不要仅凭名字缺失认定几何丢失。

## 8. 几何分类、Grouped VAE 与降维

`classification_test/` 是实验分析区，不参与 CST 生产批处理。

### 8.1 LightGBM/XGBoost 可行性分类

当前 `test_env.py` 会同时训练 XGBoost 和 LightGBM：

```powershell
python scripts\classification_test\test_env.py `
  --csv scripts\classification_test\scan_latin_65536.csv `
  --out-dir scripts\classification_test\model_outputs_new `
  --n-estimators 400
```

它保存：

- `normalizer_minmax.joblib`；
- `normalization_params.json`；
- `label_encoder.joblib`；
- 两个模型及其报告、混淆矩阵和指标。

BO 使用 LightGBM 作为快速门控，但最终候选仍必须通过完整几何硬判定。分类概率不能代替 Shapely 和最小点距检查。

### 8.2 Grouped VAE

五个先验结构组：

```text
inner_ring
outer_second_ring
outer_third_frame
substrate
feed
```

最小 smoke：

```powershell
python scripts\classification_test\grouped_vae_latent.py `
  --csv scripts\classification_test\scan_latin_65536.csv `
  --metadata scripts\classification_test\scan_latin_65536.csv.metadata.json `
  --out-dir results\vae_quickstart `
  --epochs 1 `
  --batch-size 2048 `
  --reduction-sample-size 2000
```

正式运行时增加 epochs；需要 UMAP 时再加 `--run-umap`。UMAP 首次运行可能花较长时间编译 Numba。若长时间无输出，先运行：

```powershell
python scripts\classification_test\diagnose_umap_breast_cancer.py `
  --fit `
  --numba-cache-dir scripts\classification_test\umap_numba_cache
```

VAE 用来研究线性超矩形无法表达的低维结构，不直接作为几何可行性硬约束。

## 9. BoTorch qLogEHVI 优化

### 9.1 当前优化逻辑

项目中的顺序式多目标 BO 是：

```text
冻结历史 observations
-> 输入按 metadata 归一化
-> 两个 RF 目标显式标准化
-> SingleTaskGP，固定标准化噪声方差 1e-6
-> Sobol 候选池
-> LightGBM 概率门控
-> 完整几何硬门控
-> 精确计算 substrate area
-> qLogEHVI 在合法离散池中选 1 点
-> CST 求解
-> 解析新 observation
-> 下一轮重新拟合
```

默认三个优化目标均转为“越大越好”：

```text
mean_matching_efficiency
mean_radiation_efficiency
negative_substrate_area_mm2
```

两个 RF 目标由 GP 代理；面积直接由候选几何精确计算，不训练第三个 GP。qLogEHVI 衡量候选对当前非支配目标空间的期望超体积增量。

当前实现限制最多选择两个 RF 目标，因为再加面积后总目标数不超过 3。若要做 4 个以上目标，应先重新评估 EHVI 的计算成本和是否切换到 K-RVEA 等 many-objective 方法，不要只注释保护代码就直接生产运行。

### 9.2 BO 依赖预检

同一个解释器必须通过：

```powershell
python -c "import cst.interface, lightgbm, torch, botorch, gpytorch; print('BO+CST ok')"
where.exe ninja
```

还要确认：

- `initial_observations.csv` 存在且目标列完整；
- classifier 目录包含模型、normalizer 和 label encoder；
- scan metadata 的参数名和顺序与当前 45 维注册表完全一致；
- CST 模板、端口和 monitor 已准备；
- 输出目录不是历史数据目录。

### 9.3 先停在 CST 前

这个命令会拟合 GP、生成并保存候选，但不启动 CST：

```powershell
python scripts\Automation_scripts\bayesian_optimization_smoke.py `
  --budget 1 `
  --stop-before-cst `
  --device cpu
```

等 `bo_candidates.csv`、`bo_history.csv`、scaler JSON 和 GP checkpoint 都正常后，再去掉 `--stop-before-cst` 做 1 个真实点。

Python API：

```python
from pathlib import Path
from Automation_scripts.bayesian_optimization_smoke import (
    BoConfig,
    run_bayesian_optimization,
)

config = BoConfig(
    simulation_budget=1,
    output_dir=Path("results/bo_quickstart"),
    stop_before_cst=True,
    device="cpu",
)
observations = run_bayesian_optimization(config)
```

生产 200 点入口：

```powershell
python scripts\Automation_scripts\bayesian_optimization_run_200.py
```

它会先验证输入文件，并可在未检测到 CST 主进程时启动 CST Design Environment。正式启动前必须人工复核文件顶部全部 `F5_*`，尤其是初始 observation、classifier、metadata、输出目录、预算和 device。

BO 会保存候选、状态、history、observations 和 checkpoint，因此可以按 iteration 续跑。不要删除中间 ledger 后只保留 CST 文件夹；那会破坏恢复语义。

## 10. BO 与 Pareto 后分析

这些脚本都不连接 CST，只读取已经导出的 observation 和 case 结果：

```powershell
python scripts\Automation_scripts\analyze_bo_imse.py
python scripts\Automation_scripts\analyze_bo_hypervolume.py
python scripts\Automation_scripts\analyze_pareto_onion.py
python scripts\Automation_scripts\view_onion_samples.py
python scripts\Automation_scripts\analyze_onion_parameter_ranges.py
python scripts\Automation_scripts\sample_expanded_parameter_slabs.py
```

用途：

| 技术 | 回答的问题 |
| --- | --- |
| IMSE | GP 在固定可行积分域上的平均后验方差是否下降？ |
| Observed hypervolume | 已观测非支配集合相对固定 reference point 扩张了多少？ |
| Pareto onion | 去掉当前 Pareto front 后，下一层高质量点是谁？ |
| 参数范围分析 | 优秀层是否持续挤压某些变量边界？ |
| 扩展薄层 Sobol | 在建议扩张的旧边界外，能否找到新的几何可行点？ |

比较原则：

- IMSE 必须使用固定积分点和一致的目标标准化。
- Hypervolume 跨 checkpoint 比较时必须使用同一个 scaler 和 reference point。
- Pareto onion 第一层是全局非支配层；后续层是逐层剥离后的 front。
- 范围建议是证据和采样建议，不自动修改硬边界。
- LightGBM 在历史训练箱外不能当作可靠外推证据。
- 分析脚本要求的增量点数量超过当前完成量时，应降低对应 `F5_MAX_INCREMENTAL_POINTS` 或等待更多结果，而不是伪造缺失 observation。

## 11. 关键数据契约

| 文件 | 角色 | 是否可随意改 |
| --- | --- | --- |
| `fractal_antenna_curves.json` | 当前默认几何的 6 曲线顶点 | 可由几何脚本重新导出 |
| `scan_*.csv` | QMC 参数和 `geometry_feasible` | 生成后应视为数据快照 |
| `scan_*.csv.metadata.json` | 参数顺序、范围、固定值和采样设置 | 不应手改 |
| `*.feasible_4d.csv` | 稳定编号 worklist | 批次开始后禁止手改 |
| `*.feasible_4d.csv.metadata.json` | 源 hash、case 数和编号宽度 | 不应手改 |
| `case_XXXX/parameters.csv` | 单个 CAD case 参数快照 | 只读证据 |
| `run_results.csv` | CST case 状态 ledger | 由 runner 维护 |
| `bad_points.csv` | 普通坏点 ledger | 由 runner 维护 |
| `observations.csv` | BO 训练和续跑 ledger | 由 BO runner 原子更新 |
| `input_scaler.json` / `output_scaler.json` | BO 归一化和固定 reference point | 不应手改 |

任何“换 CSV 但继续沿用旧编号/旧结果”的操作，都应先比较参数列、metadata、固定值和源 hash。

## 12. 什么时候先查本地，什么时候上网

### 12.1 先查本地

以下问题先读项目代码和本地文档，因为网络不知道本工程的真实状态：

- 当前 F5 样本数、起始 case、频率和输出目录；
- 当前 45 个变量的名字、顺序和范围；
- 哪些参数已固定；
- CST 实体名、材料名、布尔顺序和导出树路径；
- 某个批次的编号、源 CSV 和 resume 状态；
- 当前结果到底完成了多少点；
- 本项目曾经验证过的坐标、点序和层叠方向。

本地 CST 官方文档：

```text
scripts/Python/main.html
scripts/Python/source/cst.interface.html
scripts/vba/vba_macro_language_overview.htm
```

搜索例子：

```powershell
rg -n "execute_vba_code|DesignEnvironment|run_solver" scripts\Python
rg -n "Solid.Subtract|Solid.Delete|ExtrudeCurve" scripts\vba
```

### 12.2 应上网查

出现这些情况时查网络，而且优先官方文档、发行说明或原始论文：

- 当前安装版本的 CST/AEDT API 与本地示例签名不一致；
- AEDT COM、ScriptEnv 或 gRPC 行为疑似随版本变化；
- BoTorch、GPyTorch、LightGBM、PyTorch/CUDA 出现弃用、版本兼容或 API 变更；
- 错误码明显属于某个软件版本的已知问题；
- 需要可引用的材料参数、标准、算法定义或论文依据；
- 准备安装/升级 PyTorch、CUDA、CST Python wheel 或 AEDT 接口。

技术搜索只使用一手来源：

- Dassault Systèmes/CST 官方文档；
- Ansys AEDT/PyAEDT 官方文档；
- PyTorch、BoTorch、GPyTorch、LightGBM、SciPy、Shapely 官方文档；
- 算法原论文。

不要为了查一个错误，把工程、参数表、许可证信息或未公开几何上传到外部网站。网络只能解释通用 API，不能替代本地工程检查。

## 13. 什么时候停止自动重试并请求用户帮助

这里的 `break` 指停止当前自动化链路、保存现场、报告已验证事实，然后请求用户完成只有人或 GUI 能完成的动作。

### 13.1 必须立即 break

- CST/HFSS 弹出许可证、崩溃恢复、保存冲突或其它模态对话框。
- 模板缺少材料、端口、边界、setup、solver 或 Field Monitor，而当前任务没有授权自动创建。
- 需要修改物理定义，但坐标轴、正负方向、厚度、材料或尺寸意图不明确。
- 要删除/覆盖已有批次、锁定 worklist、observation ledger 或大规模结果。
- Python 与 CAD 运行在不同 Windows 用户/权限级别，需要用户重启其中一个。
- CAD 工程被另一会话锁定，或必须关闭用户正在使用的 CST/AEDT。
- 即将启动长时间 solver、批量 DoE 或 200 点 BO，但输入、预算、输出目录或恢复点还没有人工确认。

### 13.2 达到阈值后 break

- DoE 已触发连续 5 点失败熔断。
- 同一个连接/权限错误在核对解释器、进程和权限后仍重复 3 次。
- 普通 CST VBA 命令超过 60 秒且无法判断是 GUI 阻塞还是正在处理。
- 候选池连续为空：LightGBM 门控或几何硬门控没有点通过。
- BO 的 metadata、classifier feature order 和当前参数注册表不一致。
- 预期的 observation、scaler、checkpoint 或 case 文件缺失，无法证明如何安全续跑。
- HFSS 自动 Unite 后无法通过对象列表、体积或截图确认物理结构仍完整。

### 13.3 请求帮助时要带什么

不要只说“CST 卡了”。至少报告：

```text
正在运行的脚本和解释器
工程路径、design 名和 backend
输入 CSV / worklist / case_id
最后一个成功阶段
完整异常类型和消息
是否存在 GUI 对话框
已经尝试过的动作
哪些文件已写入，哪些没有
希望用户执行的唯一下一步
```

例如：

```text
case 0619 在 validate_project_prerequisites 前失败；
cstpy 能 import cst.interface，但 connect_to_any_or_new 连续 3 次返回同一 COM 错误；
CST 当前以管理员权限运行，Python 是普通权限；
尚未删除几何、未启动 solver、未写 bad_points。
请用普通权限重启 CST，然后告诉我已打开 test.cst。
```

## 14. 三条最短工作流

### 14.1 修改几何后

```text
修改 Fractal_Antenna.py / cfsa_geometry
-> 单元测试
-> Fractal_Antenna.py --static
-> cst_import_fractal_antenna.py --dry-run
-> HFSS create_cfsa_geometry.py --dry-run
-> 只重建 1 次 CST 或 HFSS
-> 视觉和实体名检查
```

### 14.2 新建 DoE 批次

```text
核对参数注册表和范围
-> 16 点单进程扫描
-> 正式 Latin/Sobol 扫描
-> 检查 CSV + metadata
-> 第一次锁定 worklist 和 id_width
-> 2 点 dry-run
-> 1 点 solver/export
-> 2 点连续重建
-> 正式批跑和 resume
```

### 14.3 新建 BO 批次

```text
冻结并校验历史 observations
-> 训练/校验 LightGBM
-> 检查 BO+CST 同一解释器
-> stop-before-cst 生成 1 个候选
-> 检查 gate、area、qLogEHVI、checkpoint
-> 1 个真实 CST 点
-> 验证 observation 回写和 resume
-> 再启动生产预算
-> 用 IMSE/HV/Pareto onion 检查优化是否真的在学习
```

这三条工作流的共同原则是：先证明数据契约和一个最小闭环，再把昂贵的 CAD/solver 预算放大。
