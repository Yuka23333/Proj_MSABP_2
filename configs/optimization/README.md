# 天线采样配置

## 任意树 35D K-RVEA（独立入口）

`phase2_krvea_tree_35d.json` 配置当前任意树三目标流程，不替换下面的旧版参数化配置。
复用 316 个 Morris 独立几何，补充 128 个训练点（含参考点）和 64 个独立 holdout。
`optimization_budget=null`、`validation_approved=false` 为有意保留的验证门槛。
入口为 `scripts/optimization/prepare_krvea_tree.py` 和
`scripts/optimization/run_phase2_krvea_tree.py`；完整命令及数据契约见
[35D tree K-RVEA 说明](../../docs/KRVEA_TREE_35D_BOOTSTRAP.md)。

## 旧版采样配置

`antenna_sampling.json` 是新版 Shapely 天线的唯一默认采样配置，供 IDE/F5、
命令行、几何批量检查和 Princess/Maid 共用。配置的 23 个字段必须与
`scripts/geometry/shapely_antenna_model.py` 中的参数同名；缺失、拼错或多余字段
都会立即报错，避免 CSV 列被静默忽略。

## 当前设计空间

7 个毫米绝对值变量：

| 变量 | 默认值 | 采样范围 |
|---|---:|---:|
| `SLOT_MAIN_LENGTH` | 53 | `[15, 60]` mm |
| `SLOT_MAIN_HEIGHT` | 2 | `[1, 3]` mm |
| `PATCH_BRICK_1_SIDE_MARGIN` | 6 | `[50%, 150%]` |
| `PATCH_BRICK_1_TOP_MARGIN` | 2.6 | `[50%, 150%]` |
| `PATCH_BRICK_3_BOTTOM_MARGIN` | 2 | `[50%, 150%]` |
| `PATCH_BRICK_2_HEIGHT_MARGIN` | 15 | `[50%, 150%]` |
| `PATCH_BRICK_4_MARGIN` | 4 | `[50%, 150%]` |

其余 16 个 `*_K*` 变量均为无量纲比例，硬范围和采样范围都是 `[0, 1]`：
Upper corner 4 个、Lower corner 6 个、Branch 6 个。所有 23 个变量默认参与采样。
`BRANCH_DOWN_1_K3` 的基准值为 `0`，但仍在完整 `[0,1]` 范围内参与采样。

第一轮 DoE 的方法、种子、过滤策略和输出位置单独记录在
`doe_round1_lhs_512.json`，避免为某一轮试验修改共享参数范围配置。对应生成入口为
`scripts/optimization/prepare_doe_round1.py`。

每个变量都在 `sampling.parameters.<name>` 下独立声明。绝对范围使用
`{"mode": "absolute", "min": ..., "max": ...}`；相对 nominal 的范围使用
`{"mode": "relative", "lower": -0.5, "upper": 0.5,
"reference": "nominal"}`。当前 schema 不再使用 Global/Group 范围继承；代码中的
Group 只用于分类和解析结果展示。

## 第二阶段 K-RVEA

`phase2_krvea_roi_radiation_gain_64.json` 使用独立的 schema、plan ID 和输出
目录，服务于 `scripts/optimization/run_phase2_krvea.py`。其中
`phase2_metric` 是被冻结的传播代理指标契约；修改频点、角域、极化或增益类型时，
必须另建入口、配置和目标 schema，不能原地恢复当前 campaign。

## 1U1D 14维桥接试验

`doe_1u1d_14d_bridge_64.json` 定义从单-UP基础拓扑到1U1D拓扑的配对增量
试验。它从1024条历史记录中的993个完整单-UP结果选择16个父样本，并将每个
父样本与同一组4个DOWN枝条优化LHS设置交叉，共产生64个新求解；父样本的
`K3=0`结果直接作为16条旧基线复用，不重复求解。

准备脚本只生成工作表和血缘审计，不启动CST：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\optimization\prepare_doe_1u1d_14d_bridge.py
```

真实仿真使用固定run ID：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\simulation\princess.py start `
  --csv data\samples\doe-1u1d-14d-bridge-64.csv `
  --run-id doe-1u1d-14d-bridge-64-001 `
  --device convallariag5 `
  --device coconutg2
```

增量模型的训练/验证必须按`parent_slot`整组拆分，不能把同一父样本的四个子点
同时泄漏到训练集和holdout。计划manifest同时记录源观测、父样本、CST模板及
输出CSV的SHA-256。

## 几何策略

- 坐标以 `0.01 mm` 量化；
- 每个样本直接调用新版 Shapely 生成器产生 `Slot`、`Patch`、
  `CPW_Feed_Pin` 三条曲线；
- 生成器负责闭合环简单性和 Polygon 有效性检查，非法组合在 CSV 中记录为
  `geometry_valid=false` 和对应 `geometry_error`；
- CST 与并行几何检查器复用同一个内存生成入口，不通过共享 JSON 文件传递样本，
  因而多个 Maid 不会互相覆盖几何。
