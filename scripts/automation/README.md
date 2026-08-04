# Automation

`Legacy_Automation_scripts/` 是从其它项目迁移来的参考实现，目前还未做代码
标准化整合；目录外的脚本是当前 MSA-BP 项目的活动代码。

## 参数采样与量化几何检查

`antenna_sampler.py` 从
`configs/optimization/antenna_sampling.json` 读取完整配置。直接从 IDE/F5
运行和命令行运行默认使用同一个 JSON；命令行只需在必要时覆盖样本数、方法、
随机种子和输出路径。

范围按每个变量独立解析，优先级为：

```text
变量专属范围 > 精确 Group 范围 > Global 范围
```

绝对范围使用 `min/max`；相对范围使用相对 nominal 的有符号
`lower/upper`，例如 `-0.2/0.3` 表示 `0.8x..1.3x`。nominal 为零的参数
必须使用绝对范围。未写 `sample` 时，普通设计参数默认参与，
`fixed_by_default` 默认不参与；关闭枝条的参数自动退出有效采样维度。

每个样本都会生成 CST 实际接收的六条源曲线，并按以下顺序检查：

1. 所有坐标以 `ROUND_HALF_UP` 量化到 0.01 mm；
2. 拒绝量化后折叠的边和重复造成的退化；
3. 对每条闭合曲线单独检查自交、自触、方向和有效性；
4. 重新检查 slot/guide 包含关系和布尔结果；
5. 默认拒绝断裂的最终金属导体。

自交检查针对送入 CST 的单个 Polygon 源曲线，不能通过
`allow_disconnected_conductor` 关闭。后者只允许完成
`Patch - slot + guide` 后出现多个金属分量。

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

## 单次基准仿真

`cst_run_and_export_s11.py` 只重建脚本管理的 `component1:msabp_*`
几何，保留 CST 工程中已创建的 SMA connector、端口、边界、solver
配置和 Farfield Monitor。它会清除旧结果、同步运行一次 solver，然后导出
`1D Results\S-Parameters\S1,1`。

```powershell
& 'C:\Users\David\.conda\envs\cstpy\python.exe' `
  scripts\automation\cst_run_and_export_s11.py
```

默认输出为 `results/raw/baseline/S11.csv`。只检查 CST 模板先决条件时使用
`--check-only`；已由其他流程更新几何时可使用 `--skip-rebuild`。
若控制端中断但 CST solver 仍在运行，使用 `--resume-running`接回并导出；
使用 `--plot-existing` 可将已导出的 ASCII S11 绘制为
`results/figures/S11_baseline.png`。
