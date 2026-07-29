# 可执行脚本

本目录只放可直接运行的流程入口：

- `optimization/`：生成候选点或启动优化循环；
- `simulation/`：创建算例、运行 CST、导出结果；
- `postprocessing/`：汇总指标、生成图表和报告；
- `utilities/`：数据检查、格式转换等独立小工具。

脚本应尽量保持轻量，只处理命令行参数、配置加载和流程编排。可复用实现放在 `src/msabp_opt/`，并为脚本提供 `if __name__ == "__main__":` 入口。

## CST 独立项目验证

`simulation/verify_cst_standalone.py` 会把每个 `.cst` 单独复制到一个全新的空目录，再通过 CST Python 接口打开副本并关闭，不复制同名结果目录，也不修改源文件。

必须使用能够连接 CST 的 `cstpy` 环境：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\verify_cst_standalone.py
```

如需保留 CST 打开副本后生成的临时目录用于人工检查，可增加 `--keep-workspace`。
