# 项目文档

用于保存参数化几何说明、优化目标与约束定义、CST 建模约定、实验决策记录和阶段性总结。

当某个流程需要隔几个月后仍能复现时，应在这里或对应目录的 README 中记录入口命令、输入、输出和已知限制。

## 研究决策记录

- `PHASE2_RESEARCH_MARK.md`：第二阶段目标变更标记；废弃 Cap Gain 和独立
  Rad_Eff 目标，以包含 Rad_Eff 的固定 ROI radiation gain 作为传播代理指标。
- `FARFIELD_COORDINATE_CONVENTION.md`：CST 远场方向、theta/phi 极化与天线坐标约定。
- `SEPTEMBER_RF_METRICS.md`：September 三信道线性功率加权指标定义。
- `BER_V2_IEEE802156_3CH.md`：IEEE 802.15.6 三信道 BER v2 实验口径。

## 外部参考资料

- `reference/cst_python_libraries.html`：CST Studio Suite 官方 Python Libraries 帮助文档的入口页。

该 HTML 来自官方 Sphinx 文档，但当前项目只包含入口页；其中指向 `_static/`、`source/` 和索引页的相对链接需要原始完整帮助文档目录才能使用。
