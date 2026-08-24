# CST 工程隔离区

本目录同时容纳两类内容：

- `*.cst`：可独立打开并恢复主要模型数据的 CST 工程容器，纳入 Git；
- 与 `.cst` 同名的目录、`bakN`、缓存和求解结果：由 CST 自动生成，只保留在本机。

## `msa-bp.cst` 的模板契约

旧版本曾出现裸 `.cst` 副本丢失 Field Monitor、几何 Pick 或依赖 Pick 的 Port 有效引用，
并在求解时报告 `No valid excitation sources defined`。当前维护的 `msa-bp.cst` 已修复：
2026-08-24 在 CST Studio Suite 2025 中将该文件单独复制到空目录后，完成了删除/重画
脚本管理几何、检查 Port 1 与 61 个 Farfield Monitor、真实求解及三条 1D 曲线导出。

Port、Field Monitor、网格和 solver 设置现在属于模板工程。自动化在每次重画后只校验这些
对象，不再通过 Python/VBA 删除、重建或静默修补；模板不完整时应在求解前直接失败。

`.gitignore` 对本目录采用“默认全部排除、只放行 `*.cst` 和本 README”的策略。因此新增工程时只需把 `.cst` 放在这里；CST 后续生成何种同级工作目录都不会被 Git 收录。

不要把需要长期保存的脚本、配置或人工导出数据放进本目录，因为除 `.cst` 和本 README 外都会被忽略：

- 仿真模板或宏放入 `../templates/`；
- 批量临时算例放入 `../runs/`；
- 求解器导出的数据放入 `../../results/raw/`。
