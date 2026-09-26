# 三设备基础仿真验证

入口：`scripts/simulation/基础仿真文件验证.py`，支持 IDE F5。

默认使用当前 **固定 23 参数模型**的 `DEFAULT_PARAMETERS`，不是任意树模型，
也不是今天的 case 1/2/3。任意树 CST 临时建模脚本尚未接入 Maid。
本验证复用现有 Maid 建模、材料设置、求解及导出流程，不替换这些实现；
因此它验证的是这条现有流程在三台设备上的一致性，不能代替新树模型的远程验收，
也不能单独证明模板中手工材料定义被原样用于求解。

## 运行

```powershell
& C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\基础仿真文件验证.py --prepare-only
& C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\基础仿真文件验证.py
```

第二条会要求输入 `RUN`。脚本按本机、铃兰 G5、椰子 G2 的顺序，启动三个
各自只绑定一台设备的 Princess 任务；每个任务只包含同一行参数，避免任务抢占
导致某台设备没有参与。使用现有设备注册表中的 `local`、`convallariag5`、`coconutg2`。
这是顺序运行，而不是三台同时启动。不要与其它 Princess/CST 任务重叠运行。

冻结参数、模板副本、计划、日志、结果和比较图均放在：
`simulations/runs/base-validation-001/`。
原始 `simulations/models/msa-bp.cst` 不直接参与建模，三台收到同一个冻结模板。
这用于排除输入差异，不用于检查远端已有模板是否一致。

`--parameters custom.json` 接受固定模型参数名到数值的覆盖字典；未指定项使用默认值。
改参数、模板、设备配置、频段或容差需使用新的 `--run-id`。同一计划可重复运行，
各子任务由 Princess 自身恢复，不重新创建已完成结果。

```powershell
& C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\基础仿真文件验证.py --compare-only
```

## 比较契约

- 校验成功 manifest、参数完全一致、三个 1D 文件及 SHA256；缺失不算通过。
- 复用当前 CST 导出的 dB 格式 S11、Rad_Eff、Tot_Eff，不计算优化用频率加权指标。
- 默认频段 3.1–4.8 GHz；`--band 2 8` 可比较更宽频段。
- 要求全部曲线覆盖完整指定频段，拒绝外推、重复频率和非有限数值。
- 不同频率网格在双方节点的并集上作线性插值（dB 域）。
- 三对设备 × 三条曲线：最大绝对 dB 差、频率加权 RMS dB 差、最大差所在频点。
- `--tolerance-db 0.1` 为可调工程容差，不是统计置信界限；任何比较超限返回退出码 2。
- 输出 `comparison.json`、`comparison.csv`、三幅叠加曲线组成的 `comparison.png`。
- FFS 仍按现有 Maid 流程导出/回收，但此脚本不比较远场。

在远端启动前需要同步所需代码、确认 License 和 Maid Bell 可用；本脚本不自动
安装依赖、不改设备启用状态、不自动提交或拉取 Git。
