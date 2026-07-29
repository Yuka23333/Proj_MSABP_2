# 仿真资产与运行目录

- `models/`：CST 工程隔离区；Git 只同步 `.cst`，同名工作/结果目录留在本机。
- `templates/`：可复制的工程模板、VBA 宏、求解器设置和导出模板。
- `runs/`：批量算例的临时工作目录，按实验编号和 case 编号分层。

建议的运行路径为：

```text
simulations/runs/<experiment_id>/case_<case_id>/
```

`runs/` 默认不进入版本管理。需要长期保存的模型应回收到 `models/`，求解器导出的指标应进入 `results/raw/`。

## CST 工程的版本管理

CST 2025 实测确认：只把 `.cst` 复制到全新的空目录，仍可正常打开完整工程；打开时 CST 会自动重建同名工作/结果目录。因此工程与自动生成目录可以共同放在 `models/`，但 Git 只同步 `.cst`。

项目根目录的 `.gitignore` 会：

- 保留 `simulations/models/*.cst` 和 `models/README.md`；
- 默认忽略 `models/` 中除此之外的所有文件和目录；
- 保留本项目管理的 `templates/` 和 `runs/` 目录结构；
- 因而自动涵盖 CST 同名目录、缓存以及 `.bak1`、`.bak2` 等备份。

可使用以下命令复验：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\verify_cst_standalone.py
```
