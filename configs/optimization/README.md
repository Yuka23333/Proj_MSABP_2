# 天线采样配置

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

## 几何策略

- 坐标以 `0.01 mm` 量化；
- 每个样本直接调用新版 Shapely 生成器产生 `Slot`、`Patch`、
  `CPW_Feed_Pin` 三条曲线；
- 生成器负责闭合环简单性和 Polygon 有效性检查，非法组合在 CSV 中记录为
  `geometry_valid=false` 和对应 `geometry_error`；
- CST 与并行几何检查器复用同一个内存生成入口，不通过共享 JSON 文件传递样本，
  因而多个 Maid 不会互相覆盖几何。
