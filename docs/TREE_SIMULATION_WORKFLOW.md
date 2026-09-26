# 任意树自动化仿真（独立 `_tree` 路径）

## 入口与范围

- `scripts/automation/cst_build_msabp_geometry_tree.py`：无 GUI 的几何准备和 CST 建模器。
- `src/msabp_opt/simulation/distributed/case_runner_tree.py`：Maid 的树样本执行器。
- `scripts/simulation/基础仿真文件验证_tree.py`：三个设备同参数验证，F5 或命令行运行。

旧建模脚本和旧验证入口不替换。公共 `case_runner.py` 只新增 `antenna_tree`
分发分支与能力声明，Princess 只增加该模式的结果验收规则。共享调度、Bell、
心跳、重试、导出和回传机制不复制。新入口仍复用旧验证脚本的日志和比较函数。

## 模板要求与清理

模板为磁盘上的 `simulations/models/msa-bp.cst`。需有可重放的 SMA、Port 1、
2–8 GHz 监视器与 `Rogers AD 350A (lossy)` 材料定义；材料取模板值，不重新创建。
建模前只允许 `Connector:` 中的三个实体，且不允许残留辅助曲线；未知几何会报错，
不会被擅自删除。今天的材料定义为 εr=3.5、tanδ=0.003。

新建几何归于 `msabp_tree` 组件，所有结构操作写入 `MSABP_TREE::` 历史块。
后续样本先备份历史，只截断自己的连续末尾历史，再完整重建基础历史。
若树历史之后出现手工步骤、基础历史已有错误，或更新失败，则停止。
无需写死基础历史条数。不会清掉用户新增的材料定义。

今天的临时 `Tree case1:` / `Tree case2:` / `Tree case3:` 历史不属于该自动路径；
不要把带这些临时步骤的项目作为新模板，使用已经保存的干净基础模板。

## 几何契约

请求 JSON 包含 `params`、`tree`、`build_options`，对应无界面任意树模型。
默认验证选择今天的 **case3**：三块铜区，含一块悬浮岛。还支持 case1、case2、default。
三个测试 case 固定使用 `snap_fraction=0.1`、`tip_clearance=1.0`，与今天验证参数一致；
自定义请求未给选项时遵循当前模型默认值，而非强行改动模型常量。

所有源多边形整体平移，使基板底边 Y=0；按 0.01 mm 量化。独立槽、多铜区、
内孔均被保留，禁止强行合并成一条曲线。量化导致退化或非法则前置报错。
铜布尔完成后不二次量化交点，每块铜先拉伸外轮廓，再扣除全部内孔。
基板 -4.7 mm、铜厚 0.035 mm，背面反射板沿用既有尺寸和底边避让槽定义。
建模后验证实体数量、体积、材料名称、Z 范围及历史错误标记。

## 运行

先把新文件和两个路由修改一起提交、推送、在两台 Maid 拉取。
本地与远端均需 Shapely、现有仿真环境、可用 License/Bell。不要与其它本地 CST
或 Princess 任务并行占用同一设备。脚本不会自行推送代码、安装依赖或覆盖模板。

```powershell
& C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\基础仿真文件验证_tree.py --prepare-only
& C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\基础仿真文件验证_tree.py
```

第二条要求输入 `RUN`。顺序绑定 local、convallariag5、coconutg2，每台各跑同一条
样本；工作副本和结果隔离，原始模板不被建模覆盖。正式运行需要同步新增路由，
旧 Maid 不支持该模式时会被能力检查拒绝。

默认计划：`simulations/runs/base-validation-tree-001/`。
`--case case2 --run-id base-validation-tree-hole-001` 可切换测试；
`--request my_tree.json` 可输入自定义树。改参数/模板/配置/容差需新的 run ID。

结果包含 S11、Rad_Eff、Tot_Eff、FFS、最终几何 JSON 和基础历史备份/元数据。
比较完整性、参数、实际几何哈希、代码哈希和基础历史哈希，然后比较三对设备
各条曲线的 dB 差异。源代码哈希忽略 CRLF/LF 差异。默认 3.1–4.8 GHz、0.1 dB
工程容差，可用 `--band`、`--tolerance-db` 修改。FFS 暂不参与数值比较。
用 `--compare-only` 重读已有结果，缺失/仅 manifest/哈希不匹配不会通过。

## 验证边界

今天的三个 case 已在临时脚本下交互验证成功；正式 `_tree` 路径由纯几何测试、
mock CST/导出测试、Maid dry-run 和旧分发流程回归覆盖。**这不等于正式新路径
已经在真实 CST 或三台设备上求解通过**；部署后的三机联调仍待用户启动。
