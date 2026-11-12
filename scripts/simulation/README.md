# Princess / Maid 仿真运行指南

维护日期：2026-10-03。此指南说明部署与运行边界，不代表当前远端设备已经在线或具备有效许可证。

## 架构与入口

Shore 双钥匙远控正在独立适配，见 [远端合成任务交接](../../docs/SHORE_REMOTE_SYNTHETIC_ADAPTER_2026-10-16.md)。
其入口为 `shore_synthetic_princess.py`、`shore_synthetic_bell.py`、`shore_synthetic_maid.py`，不使用下述 legacy 安装器或 HMAC 唤醒流程，不启动 CST。
下述既有科研仿真流程没有因此自动接入 Shore。

Princess 持有计划、预算、SQLite 状态和验收结果。远端 Maid 在本机控制 CST；
Maid Bell 通过本地服务/开机任务唤醒 Maid，避免 Windows OpenSSH 在断开后清理进程树。

```text
控制器/Princess -- SSH/SCP --> 代码检查、输入/私有模板部署
               -- HMAC TCP --> Maid Bell --> Maid --> 本机 CST
控制器/Princess <-- HTTP token -- hello、租约、心跳、结果
```

[princess.py](princess.py)、[maid.py](maid.py) 为入口；
核心在 [src/msabp_opt/simulation/distributed](../../src/msabp_opt/simulation/distributed/)。

| 任务 | 入口与边界 |
| --- | --- |
| 通用静态 CSV | `princess.py start`；必须确认 row schema 和模板来源 |
| 任意树三机验证 | [基础仿真文件验证_tree.py](基础仿真文件验证_tree.py)、[流程说明](../../docs/TREE_SIMULATION_WORKFLOW.md) |
| 旧固定模型三机验证 | [基础仿真文件验证.py](基础仿真文件验证.py)、[说明](../../docs/BASIC_SIMULATION_VALIDATION.md) |
| 14D 优化 | [优化指南](../optimization/README.md)；由 campaign 控制器分批调用 Princess |
| 八点 Kevin 传播 | [run_propagation_tree_diverse_8.py](run_propagation_tree_diverse_8.py)，冻结 `copy_bundle` |
| 七点 Kevin 传播 | [run_propagation_tree_7.py](run_propagation_tree_7.py)，原 `in_place` 计划仅作历史 |
| 旧固定模型13点传播 | [run_propagation_13.py](run_propagation_13.py)，独立旧模板/导出口径 |

## 环境、设备与部署

[设备注册表](../../configs/simulation/princess_devices.json) 中 local、convallariag5、coconutg2
默认均为 `enabled=false`，任务以显式设备列表选择。远端是 `telecom@<device>`，
仓库 `D:\Academic\Proj_MSABP_2`，Maid Python 为
`C:\Users\telecom\miniforge3\envs\maid\python.exe`。
本机 local 当前配置为 `C:\Users\David\.conda\envs\maid\python.exe`。

[maid.yaml](../../maid.yaml) 定义精简 worker 环境，CST Python 链接 wheel 来自本机 CST 安装，
不是普通 PyPI 接口包。优化 GPU 的 bocuda 与 Maid 环境是不同进程/职责。
交互运行本机脚本优先 `conda activate maid`，避免缺少 conda 原生 DLL 搜索路径。

部署前单独审查并同步所需代码；启动器不隐式 git push/pull、升级依赖或修改模板。
既有计划对源码/模板做封存校验；代码重构后的哈希不一致应报告，不绕过校验恢复。

多层 voxel/组织分层幻象（包括 `propagate_cylinder_voxel.cst`）仅允许 coconutg2，
不得向 convallariag5 分发，也不允许椰子失败后自动回退。普通单天线/Kevin任务不自动继承此限制。

## Maid Bell 一次性配置

在每台远端已准备好环境后，以管理员权限执行对应 DeviceId：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/simulation/install_maid_bell.ps1 -DeviceId coconutg2
```

另一台使用 `convallariag5`。安装器会修改服务/计划任务及防火墙，不能当成只读诊断。
默认 Auto 尝试 pywin32 服务；WDAC/Smart App Control 拒绝未签名宿主时回退开机 S4U 任务。
需要时显式 `-HostMode ScheduledTask`。S4U 不保存账户密码，也不依赖用户保持登录。
服务默认 LocalSystem 的用户级许可证/profile权限问题应人工检查，不能把密码写进仓库。

Bell 配置位于 `%ProgramData%\MSABP Maid Bell\bell.json`。
只接受仓库 `simulations/runs/**/maid_runtime.json`，仅启动固定 Maid，且同一设备同时一个实例。
wake 返回 PID 不是 ready；Princess 必须收到 Maid HTTP hello。

## SSH 认证：使用 Shore 管理 U 盘

2026-10-06 起，受管主机只接受 Shore 管理 U 盘（`<盘符>:\ShoreControl`）上的钥匙，
用户目录里的旧 SSH 钥匙已被撤销。运行本指南中任何涉及 SSH/SCP 的命令前，先插入该 U 盘并设置：

```powershell
$env:MSABP_SSH_BACKEND = 'shore-usb'
```

设置后，传输层把 `ssh`/`scp` 命令改写成 U 盘自带的 Plink/PSCP，每次连接直接从 U 盘读取钥匙，
并核对 U 盘 `nodes\<主机>.json` 里登记的主机公钥。钥匙不复制、不缓存、不进 agent；拔盘后不能新建连接。
盘符可以变化。同时插着多个管理盘时，用 `MSABP_SHORE_USB_ROOT` 指定 `ShoreControl` 目录。

限制：设备的 `ssh_target` 必须是 U 盘上已登记的节点和账户（目前 coconutg2、coconut、yiqiu-maple
均为 `telecom`）；convallariag5 未登记，不能用这个后端。不设置该变量时行为与以前完全相同。
实现见 [shore_usb_ssh.py](../../src/msabp_opt/simulation/distributed/shore_usb_ssh.py)。

这个后端在 2026-11-11 已在真机上确认可用（科研主控确认 SSH 路径恢复）。
换了主机或 U 盘后，先跑下面的 `doctor`，再跑 dry-run。Bell 唤醒和 HTTP 回传不经过 SSH，不受此设置影响。

## 检查和启动的区别

远端只读诊断（涉及 SSH，但不启动 CST）：

```powershell
python scripts/simulation/princess.py doctor --device convallariag5 --device coconutg2
```

首次尚未部署 runtime 时，默认位置缺失可使 doctor 非零；正式 start 会为每次 launch
部署实际 runtime 再检查。Python、cst.interface、repo、入口或 Bell缺失不是可忽略项。

分布式 dry-run 示例：

```powershell
python scripts/simulation/princess.py start --csv data/samples/antenna_samples.csv --run-id remote-dryrun-new-id --dry-run --device coconutg2
```

这仍会 SSH/SCP、唤醒 Maid、冻结/过滤 CSV、走租约和结果上传，只是不打开/求解 CST，
结果只有 manifest。示例 CSV 是固定模型输入；树任务应使用各自准备器。
真实运行必须使用不同 run ID，不能把 dry-run 的 completed 当成真实结果。

`run_remote_real_smoke.py --prepare-only` 可准备固定模型 smoke；
去掉参数并确认 RUN会做真实求解。所有脚本的无参数/F5行为需分别核对。

## 模板与导出契约

新单天线树流程使用主控冻结的 `msa-bp.cst`，求解频段2–6 GHz、0.1 GHz步长41个远场监视器。
材料、SMA、Port 1、边界、网格、solver由模板管理，缺项求解前失败；历史多余监视器可保留。
原始导出包含 S11、Rad_Eff、Tot_Eff、FFS和成功 manifest；树任务另保存几何/制造与基础历史证据。

新 S21 campaign必须将主控CST与完整同名伴生目录作为 `copy_bundle` 冻结，
每个 worker launch私有展开，远端models原件不被覆盖或隐含复用。
见 [S21模板分发](../../docs/S21_TEMPLATE_DISTRIBUTION.md)。
八点已形成下游分析；不得修改原plan重新解释该批。完整S2P导出仍待实现，当前只返回S21。
E-field可选，八点配置关闭；旧传播的本地原生场文件不一定回传到控制器。

旧13点流程依赖每台Maid自己维护的传播模板，不是新campaign的推荐模式。
复数S21读取通过 [export_s21_results_worker.py](../postprocessing/export_s21_results_worker.py)
独立进程隔离CST results DLL与interface/NumPy等加载冲突。

不能概括“Python/VBA建模不可迁移”：结构操作应写入历史并保存，
迁移还需核对伴生目录、外部依赖与目标机恢复。[模型说明](../../simulations/models/README.md)
区分文件相等、可以打开和真正求解通过。

## 续跑、失败和停机

同run ID仅接受冻结输入、结果根目录及相关计划一致的恢复。Princess给予旧Maid重连窗口，
每台设备有进程生命周期锁；锁文件存在本身不证明进程活着。
恢复需要新launch时使用新generation私有工作区，避免共用CST文件。

采样器标记的非法几何在主控排除，并保留 `excluded_rows.json`。
永久几何错误不计入“五连错”；其它错误连续5次使Maid退出并自动重启一次，
未成功前再次5次则隔离。成功会清空该连续失败状态。

默认单case最多3次attempt（不是额外重试3次）；启动恢复也有限次。
结果上传/complete使用同attempt/lease/hash幂等重试。默认单请求600秒、共同总期限1800秒；
总期限耗尽保留attempt/outbox，不伪报成simulation failure，由租约恢复机制处理。
优化器对仍未解决case应停止，不能伪造电磁标签。

查看静态run状态：

```powershell
python scripts/simulation/princess.py status --run-id <RUN_ID>
```

紧急停止入口 [dismiss_all_maids.py](dismiss_all_maids.py) 会实际终止Bell明确监督的Maid及其CST进程树，
持久化停止标记、释放租约并退还本次attempt；不是诊断命令。
CSV、已完成结果和数据库保留。原run显式start才恢复调度。
旧Bell不支持stop时保留错误，不回退到未经认证的强杀。

## 文件位置与安全

`simulations/runs/<run-id>/` 保存source/worklist、SQLite、runtime及incoming；
worker位于 `workers/<device-id>/`，恢复代位于 `launches/<generation>/`。
attempt、outbox、项目和伴生目录是独立对象：收到completed_ack后仅清理成功attempt/outbox，
不自动删除项目和CST侧车。旧传播原生场的 `output/local_only/` 按manifest手动归档。

通常验收结果在 `results/raw/<run-id>/`，campaign可显式指定其他results-root。
日志在 `logs/`，主控终端也应保存。运行目录虽Git忽略，不能随意清空。

Maid HTTP token和Bell HMAC TCP不额外套TLS，仅限可信Tailscale tailnet；
不要把8765/8766转发公网。runtime含API token，不提交、不贴入共享日志。
设备注册表不保存SSH私钥/长期密码。修改bind_host时同时核对advertise_url及连通性。
