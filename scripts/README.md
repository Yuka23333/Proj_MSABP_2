# 可执行脚本

本目录只放可直接运行的流程入口：

- `optimization/`：生成候选点或启动优化循环；
- `geometry/`：二维参数化几何、合法性检查和预览；
- [`simulation/`](simulation/README.md)：创建算例、运行 CST、导出结果；包含 Princess/Maid 分布式仿真入口与操作说明；
- `postprocessing/`：汇总指标、生成图表和报告；
- `utilities/`：数据检查、格式转换等独立小工具。

脚本应尽量保持轻量，只处理命令行参数、配置加载和流程编排。可复用实现放在 `src/msabp_opt/`，并为脚本提供 `if __name__ == "__main__":` 入口。

## DoE 三目标采样质量图

`postprocessing/plot_doe_pareto_3d.py` 可递归读取一个或多个 Princess
结果目录，按指定频带汇总基板总面积、带内最大 S11 和带内平均
Tot_Eff，并生成带参考点和采样非支配解集标记的三维散点图。参考点不参与
采样 Pareto 集的计算；当前 Tot_Eff 指标对导出的 dB 样本做算术平均。只有
`manifest.json`、缺少 `S11.csv` 或 `Tot_Eff.csv` 的未完成算例会自动跳过并计入
终端的 `skipped` 统计；其它损坏数据默认仍会明确报错。

在 IDE 中按 F5 时修改脚本顶部的 `F5_*` 常量。命令行可重复传入来源目录：

```powershell
python scripts\postprocessing\plot_doe_pareto_3d.py `
  --source results\raw\doe-round1-lhs-512 `
  --source results\raw\another-run `
  --band 3.1 4.8
```

## 完整天线几何

`geometry/antenna_outline.py` 在被导入时通过
`generate_complete_antenna_point_lists()` 返回 Patch、对称 slot 和对称 guide
三条显式闭合点列。导出点坐标统一量化到 0.01 mm，并对每条点列单独检查
量化后的折叠边、自交、自触和方向。终端直接运行时只打印这三条列表；从 IDE 按 F5/调试运行时，
会打开完整参数探索器。探索器先选 Group、再选变量；普通设计变量使用相对启动基准值的
0--200% 滑条，枝条锚点比例使用 0--1 滑条。每次合法修改都会继承此前已经接受的其他变量值；
无法绘制的组合不会覆盖上一张合法图，并记录到 `logs/antenna_outline_explorer.log`。

## CST 独立项目验证

`simulation/verify_cst_standalone.py` 会把每个 `.cst` 单独复制到一个全新的空目录，再通过 CST Python 接口打开副本并关闭，不复制同名结果目录，也不修改源文件。

必须使用能够连接 CST 的 `cstpy` 环境：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\verify_cst_standalone.py
```

如需保留 CST 打开副本后生成的临时目录用于人工检查，可增加 `--keep-workspace`。
