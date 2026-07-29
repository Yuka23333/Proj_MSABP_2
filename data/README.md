# 输入数据

- `designs/`：基准天线、已知可行点和候选设计的参数文件。
- `samples/`：DoE、Sobol、Latin Hypercube 或优化器生成的待仿真样本。
- `reference/`：测量结果、文献数据或旧设计指标等对照数据。

这里的数据应当是仿真流程的输入。求解器直接输出请放入 `results/raw/`，不要和输入样本混在一起。
