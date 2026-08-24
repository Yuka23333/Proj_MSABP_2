# Automation

`Legacy_Automation_scripts/` 是从其它项目迁移来的参考实现，目前还未做代码
标准化整合；目录外的脚本是当前 MSA-BP 项目的活动代码。

## 参数采样与量化几何检查

`antenna_sampler.py` 从
`configs/optimization/antenna_sampling.json` 读取完整配置。直接从 IDE/F5
运行和命令行运行默认使用同一个 JSON；命令行只需在必要时覆盖样本数、方法、
随机种子和输出路径。

当前采样器与 `shapely_rectangle_test.py` 的重构几何一致，严格包含 23 个自变量：
7 个毫米值和 16 个 `[0,1]` 比例值。`SLOT_MAIN_LENGTH` 使用 `[15,60] mm`，
`SLOT_MAIN_HEIGHT` 使用 `[1,3] mm`，其余 5 个毫米 margin 使用各自 nominal 的
`[50%,150%]`，所有 K 比例均使用 `[0,1]`。23 个变量默认全部参与采样。

每个变量在 JSON 中独立声明范围。绝对范围使用 `min/max`；相对范围使用相对
nominal 的有符号 `lower/upper`，例如 `-0.5/0.5` 表示
`0.5x..1.5x`。schema v2 不再使用 Global/Group 范围继承；Group 仅用于把解析结果
分类为毫米值、Upper corner、Lower corner 和 Branch。

每个样本直接在内存中生成 CST 实际接收的 `Slot`、`Patch`、
`CPW_Feed_Pin` 三条闭合曲线，并沿用生成器自身的量化、闭合环和 Polygon 有效性
检查。批量标注器和每个 Maid 都调用同一个入口，不会通过公共 JSON 文件覆盖彼此的
样本几何。

`scan_geometry_feasibility.py` 保留主槽两个特殊范围与全部 K 比例的 `[0,1]`，只将
5 个普通毫米 margin 从 nominal 的 `[95%,105%]` 向外扫描。它是旧可行域扫描流程的
兼容工具；正式 23 维采样以 JSON 的 `[50%,150%]` 配置为准。

最小检查：

```powershell
python scripts\automation\antenna_sampler.py --n-samples 2 --dry-run
```

正式写出 CSV 与解析后的配置快照：

```powershell
python scripts\automation\antenna_sampler.py `
  --config configs\optimization\antenna_sampling.json `
  --output data\samples\antenna_samples.csv
```

CSV 保留全部候选，并用 `geometry_valid`、`geometry_error` 标出检查结果；
增加 `--valid-only` 时只写出通过检查的候选。对应的
`antenna_samples.resolved.json` 会记录每个变量的最终范围和范围来源。浮点参数使用
17 位有效数字写入，保证 Maid 从 CSV 读回的值与通过量化几何检查的值完全一致。

`check_sampled_curve_intersections.py` 对量化后的三条闭合边界做两两相交诊断；只忽略
三条曲线全局底边上的共同接触，其他 crossing、overlap 和 touching 都分别记录。
默认生成 1024 个 Sobol 点，并写出逐样本 CSV 与汇总 JSON：

```powershell
python scripts\automation\check_sampled_curve_intersections.py
```

检查完成后，可直接在 IDE 中 F5 运行
`scripts/geometry/browse_illegal_samples.py`。它只选取单曲线无效或
`Slot–Patch` 真正 crossing 的样本，并通过 Previous/Next 两个按钮翻页；红色标出
问题位置，紫色仅表示固定的 `Slot–CPW_Feed_Pin` 顶边连接。

## 单次基准仿真

无参数的 CST 建模入口现在直接读取
`results/processed/antenna_polygon_vertices.json`。其中 `Patch`、`Slot`、
`CPW_Feed_Pin` 三个未重复首点的顶点数组会保持原顺序送入 CST，由 Polygon VBA
补上闭合边；导入端不再重复 Shapely、量化、自交或包含关系检查。基板矩形取
`Patch` 的全局包围范围，反射板仍与基板同尺寸，并沿用原有底边连接器避让槽。
旧 DoE 显式传入 `AntennaOutlineParameters` 时仍走旧参数化链，避免样本参数被静默忽略。

`cst_run_and_export_s11.py` 保留 `msa-bp.cst` 中的 SMA connector、边界和仿真 setup，
并重建脚本管理的 `component1:msabp_*` 几何。每个算例先清除旧结果，完成几何后只读校验
模板提供的：

- Port 1；
- 2--8 GHz、步长 0.1 GHz 的 61 个 Farfield Monitor；
- HF Time Domain solver。

自动化不会删除或重建 Port、Monitor、网格或 solver 设置；缺少任一必需模板对象时会在
启动 solver 前失败。校验通过后同步运行 solver，并导出以下三条原始 CST ASCII 曲线：

- `1D Results\S-Parameters\S1,1` → `S11.csv`；
- `1D Results\Efficiencies\Rad. Efficiency [1]` → `Rad_Eff.csv`；
- `1D Results\Efficiencies\Tot. Efficiency [1]` → `Tot_Eff.csv`。

`simulations/runs/port-monitor-recording/History_list_record.txt` 只保留为历史诊断证据；
活动自动化不读取或转写它。

```powershell
& 'C:\Users\David\.conda\envs\cstpy\python.exe' `
  scripts\automation\cst_run_and_export_s11.py
```

默认输出目录为 `results/raw/baseline/`。只检查 CST 模板先决条件时使用
`--check-only`；已由其他流程更新几何时可使用 `--skip-rebuild`。
若控制端中断但 CST solver 仍在运行，使用 `--resume-running`接回并导出；
使用 `--plot-existing` 可将已导出的 ASCII S11 绘制为
`results/figures/S11_baseline.png`。
