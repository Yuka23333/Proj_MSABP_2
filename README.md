# MSABP 天线小型化优化

本项目通过参数化建模、全波仿真和优化算法，探索在满足电磁性能约束的前提下进一步减小天线尺寸。

当前目录按“配置 → 采样/候选设计 → 仿真 → 后处理 → 报告”的数据流组织。可复用代码与一次性运行入口分开，CST 模型与批量算例分开，原始结果与最终图表分开。

## 目录结构

```text
.
├─ configs/                    # 优化器、参数边界和仿真设置
│  ├─ optimization/
│  └─ simulation/
├─ data/                       # 输入数据，不放求解器运行缓存
│  ├─ designs/                 # 基准设计和候选设计参数
│  ├─ samples/                 # DoE/优化器生成的参数样本
│  └─ reference/               # 对照实验或文献基准数据
├─ docs/                       # 设计说明、实验记录和技术文档
├─ notebooks/                  # 探索分析；成熟逻辑应迁移到 src/
├─ scripts/                    # 可直接运行的薄入口
│  ├─ geometry/
│  ├─ optimization/
│  ├─ simulation/
│  ├─ postprocessing/
│  └─ utilities/
├─ src/msabp_opt/              # 可导入、可测试的核心 Python 代码
│  ├─ geometry/
│  ├─ optimization/
│  ├─ simulation/
│  └─ postprocessing/
├─ simulations/
│  ├─ models/                  # 基准 CST 模型等受控输入
│  ├─ templates/               # 仿真模板、宏和导出模板
│  └─ runs/                    # 每次批量仿真的工作目录
├─ results/
│  ├─ raw/                     # 求解器直接导出的原始结果
│  ├─ processed/               # 清洗、汇总后的指标表
│  ├─ figures/                 # 可用于报告的图
│  └─ reports/                 # 实验总结和最终报告
├─ tests/                      # 单元测试与小规模流程测试
├─ logs/                       # 运行日志
├─ archive/                    # 已停用但需要保留的历史内容
└─ history_list.txt            # 本机临时记录，不纳入版本控制
```

## 推荐工作流

1. 在 `configs/` 中记录参数范围、约束、目标函数和求解器设置。
2. 将基准几何或候选设计参数放在 `data/designs/`，将 DoE/优化样本放在 `data/samples/`。
3. 由 `scripts/optimization/` 调用 `src/msabp_opt/optimization/` 生成候选点。
4. 由 `scripts/simulation/` 调用仿真适配层，把每次算例写入 `simulations/runs/<experiment_id>/`。
5. 将求解器直接导出的数据汇集到 `results/raw/<experiment_id>/`。
6. 后处理代码只读取原始结果，并将派生指标写入 `results/processed/<experiment_id>/`。
7. 图表和结论分别进入 `results/figures/` 与 `results/reports/`。

## 实验命名与可追溯性

建议统一使用以下实验编号：

```text
YYYYMMDD_<method>_<short-name>
```

例如：`20260729_bayes_size-min`。

每个正式实验至少应保存：

- 配置快照；
- 输入样本或随机种子；
- 代码版本或提交号；
- 仿真成功/失败清单；
- 原始指标和后处理结果；
- 一份简短结论。

`simulations/runs/`、`results/raw/`、`results/processed/` 和 `logs/` 默认视为可再生成的大文件目录，因此 `.gitignore` 只保留其占位文件。经过筛选的图表、报告、配置和代码应纳入版本管理。

## 编码约定

- `scripts/` 只负责解析参数和编排流程，不堆放 CST 操作或优化算法实现。
- CST 参数写入、求解、导出和缓存清理由 `src/msabp_opt/simulation/` 统一封装。
- 优化器、目标函数、约束和采样器放在 `src/msabp_opt/optimization/`。
- 所有路径从项目根目录或显式配置解析，避免硬编码本机绝对路径。
- 正式批处理前先用极小样本执行 dry-run 或 smoke test。
