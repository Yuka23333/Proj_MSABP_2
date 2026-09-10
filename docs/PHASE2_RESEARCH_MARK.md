# 第二阶段工程研究标记

记录日期：2026-09-07；目标更新：2026-09-10

本文件只定义第二阶段的研究方向，不追溯修改第一阶段已经完成的实验、配置、
manifest、缓存或论文数据口径。

## 目标状态

| 指标 | 第二阶段状态 | 处理方式 |
|---|---|---|
| `min(Cap Gain)` | 已废弃 | 不再用于采样、代理建模、优化、Pareto 判定或候选解排序；保留第一阶段历史值供复核 |
| `max(Tot_Eff)` | 已替换 | 第二阶段不再把 `Tot_Eff` 作为效率目标 |
| `max(Rad_Eff)` | 已取消独立目标 | 固定 ROI 的 radiation gain 已包含 `Rad_Eff`，不再重复拟合 |
| `max(固定 ROI θ 极化 radiation gain)` | 已采用 | 3.6 GHz，θ=55--85°，φ=90±50°；在线性功率域按立体角平均后转为 dBi |

## 阶段边界

第一阶段的历史四目标定义保持不变：

1. 最小化带内最差 S11；
2. 最大化带内平均 `Tot_Eff`；
3. 最小化归一化基板面积；
4. 最小化极区平均实现增益（Cap Gain）。

这些历史结果仍按原始四目标解释，不得用第二阶段定义重新命名或覆盖。

第二阶段当前批准的目标向量为：

1. 最小化带内最差 S11；
2. 最大化固定 ROI 的 θ 极化 radiation gain；
3. 最小化归一化基板面积。

这里使用 radiation gain 而非 realized gain：前者只吸收一次 `Rad_Eff`，不含失配
损耗，因此不会与独立的 S11 目标重复计算失配。

## 工程约束

- 不得通过修改旧 JSON 或恢复旧 campaign 的方式切换目标定义。
- 第二阶段必须使用新的入口脚本、配置文件、`plan_id`、输出目录和目标 schema。
- 第一阶段的 `Tot_Eff`、Cap Gain、远场文件和缓存继续保留，仅作为历史证据。
- 两轮传播样本分析已完成代理指标复核；第二阶段把 `Rad_Eff` 仅作为 radiation
  gain 的组成量，不再把它作为独立目标或重复训练代理模型。

## 当前实现

- 新入口：`scripts/optimization/run_phase2_krvea.py`；
- 新配置：`configs/optimization/phase2_krvea_roi_radiation_gain_64.json`；
- 新目标数据与 GPU relay：`phase2_krvea_data.py`、`phase2_krvea_relay.py`；
- 新计划使用独立的 `plan_id`、输出目录和三目标 schema；
- 截至 2026-09-10 只完成代码与本地非求解验证，尚未启动新一轮 CST 仿真。
