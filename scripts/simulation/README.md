# Princess / Maid 分布式 CST 仿真

这套流程用于把同一份采样 CSV 分发给多台 Windows/CST 主机并行计算。入口是
[`princess.py`](princess.py)；[`maid.py`](maid.py) 通常不需要手工启动。

## 实际拓扑

```text
Princess（本机）
  ├─ SCP：短连接分发 worklist、CST 项目副本和 Maid runtime JSON
  └─ SSH：短连接尝试 PowerShell Start-Process，取得 PID 后断开
                                      │
                                      ▼
                            Maid（远端本地进程）
                              ├─ 本机 cst.interface → 本机无头 CST
                              └─ 主动 HTTP hello 确认存活，再领任务、心跳、上传结果
```

SSH 只负责部署和一次“叫醒”尝试。默认的轻量路径是在 SSH 远程 PowerShell 中执行隐藏的
`Start-Process`，取得远端 PID 后结束 SSH；Maid 若成功脱离会作为远端本地进程继续运行，
随后由它调用该主机上的 `cst.interface` 启动和控制无头 CST。Princess 必须在启动期限内
收到 Maid 主动发来的 HTTP hello，才会确认本次叫醒成功，不能仅凭 `Start-Process` 返回
PID 判断。反过来，如果 `Start-Process` 已经成功、但 SSH 回执超时或丢失，只要 Maid 在
本轮叫醒的时间截点之后完成 hello，Princess 就以 hello 为事实将它视为存活；没有可靠
回执时不会强求 PID 匹配。这里的“本机”始终是相对于 Maid 所在主机而言；Princess 不
通过 SSH 持续控制 CST，也不通过 SSH 直接启动或操纵 CST。

这条路径不要求 TeamViewer 在线，也不要求 CST 出现在交互桌面。设备的默认
`launch_mode` 是 `ssh_process`，但不能无条件假设所有 Windows OpenSSH 配置都会让
`Start-Process` 的子进程在 SSH 断开后继续存活；最终以 Maid hello 实测为准。如果某台
主机会在远程会话结束时清理子进程，则把该设备改为 `scheduled_task`，用计划任务实现持久
脱离 SSH。这个 fallback 解决的是进程存活问题，不是为了提供可见桌面，也不是 CST 无头
运行的前置条件。

## 当前设备与环境

设备注册表位于
[`configs/simulation/princess_devices.json`](../../configs/simulation/princess_devices.json)。
当前两台远端采用相同目录结构：

| 设备 ID | SSH 目标 | 仓库 | Maid Python |
| --- | --- | --- | --- |
| `convallariag5` | `telecom@convallariag5` | `D:\Academic\Proj_MSABP_2` | `C:\Users\telecom\miniforge3\envs\maid\python.exe` |
| `coconutg2` | `telecom@coconutg2` | `D:\Academic\Proj_MSABP_2` | `C:\Users\telecom\miniforge3\envs\maid\python.exe` |

环境定义在仓库根目录的 [`maid.yaml`](../../maid.yaml)。`maid` 是刻意保持精简的远端
运行环境，不应替换成本机历史包袱较重的 `cstpy` 环境。

注册表中的关键字段如下：

- `bind_host`、`advertise_url` 和 `port`：Princess 监听地址以及 Maid 回连地址；
- `enabled`：未传 `--device` 时是否默认参与；
- `launch_mode`：常规远端使用 `ssh_process`；
- `ssh_target`、`repo_root`、`python_path`：SSH 地址和远端本地路径；
- `runtime_config_path`：注册表层面的默认/doctor 检查路径；`start` 实际会为每次唤醒
  生成独立路径，并在启动该 Maid 时覆盖这个值。

当前三个设备都设置为 `enabled: false`，用于避免误启动。运行命令中重复使用
`--device` 可以显式选择设备，并只对本次运行生效，例如：

```powershell
--device convallariag5 --device coconutg2
```

需要加入新主机时，可以直接按同一 schema 编辑 JSON，也可以使用：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py add-device `
  --id new-host `
  --ssh-target telecom@new-host `
  --repo-root D:\Academic\Proj_MSABP_2 `
  --python-path C:\Users\telecom\miniforge3\envs\maid\python.exe
```

## 首次 push / pull 后的检查与启动

以下命令均从 Princess 主机的仓库根目录
`D:\Academic\Proj_MSABP_2` 执行。先确认本地提交已经 push，再让两台 Maid 主机拉取
同一提交：

```powershell
ssh telecom@convallariag5 "git -C D:\Academic\Proj_MSABP_2 pull --ff-only"
ssh telecom@coconutg2 "git -C D:\Academic\Proj_MSABP_2 pull --ff-only"
```

### 1. 只读 doctor

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py doctor `
  --device convallariag5 `
  --device coconutg2
```

`doctor` 通过 SSH 只读检查远端仓库、Maid Python、Python 依赖、仓库中的 CST 模板、
入口脚本和注册表默认位置的 runtime JSON，不会启动 CST。第一次只完成 `git pull`、
尚未运行 Princess 时，报告中出现 `missing runtime_config` 是预期现象，因此该次命令会
返回非零退出码；`start` 会先用 SCP 部署本次 launch 专属的 runtime JSON，再对实际启动
配置执行完整 doctor。其他缺项，尤其是 `python`、`cst.interface`、`repo_root` 或
`maid_entrypoint` 缺失，则必须先修复。

### 2. 先跑分布式 dry-run

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py start `
  --csv data\samples\antenna_samples.csv `
  --run-id remote-dryrun-001 `
  --dry-run `
  --device convallariag5 `
  --device coconutg2
```

`--dry-run` 仍会完整测试 CSV 冻结与过滤、SCP、SSH 唤醒与 hello 确认、Maid 主动回连、
任务租约、几何预检、结果打包上传和 Princess 落盘，但不会连接或启动 CST，也不会产生
S11/FFS。
dry-run 会把任务标记为完成；真实仿真必须使用新的 `--run-id`。

### 3. 运行真实仿真

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py start `
  --csv data\samples\antenna_samples.csv `
  --run-id doe-001 `
  --device convallariag5 `
  --device coconutg2
```

也可以追加 `--device local`，让本机 `cstpy` 环境作为第三个 Maid。`start` 会持续运行
Princess HTTP 服务并监督租约，直到所有有效任务进入终态。按 `Ctrl+C` 不会删除状态；
使用同一 `--run-id` 和完全相同的 CSV 再次启动即可从持久化状态继续。若 CSV 内容发生
变化，应使用新的 run ID。

恢复同一 run 时，Princess 会先给原 Maid 一个重连窗口，再决定是否重新唤醒；可用
`--resume-grace-seconds` 显式设置，未设置时按 heartbeat/poll 周期自动计算。每台 Windows
设备还有一个进程生命周期级单实例锁
`simulations/runs/maid.<device-id>.lock`，因此旧 Maid 仍存活时，新 Maid 会拒绝启动，
不会有两个进程同时控制该设备。锁文件可以长期存在；是否被进程持有才代表 Maid 是否
正在运行，不能仅凭文件存在判断状态。

### 4. 查看状态

在另一个 PowerShell 窗口中，或运行结束后执行：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py status `
  --run-id doe-001
```

输出包含 `completed`、`running`、`pending`、`failed`、总数，以及每台 Maid 的状态。

## CSV 过滤与失败策略

Princess 首先冻结原始 CSV，再在本机生成实际下发的 `worklist.csv`。采样器已经标记为
`geometry_valid=false` 的行会在 Princess 侧直接排除，原因写入
`excluded_rows.json`；这些行不会发给 Maid，也不会计入 Maid 的连续错误次数。

Maid 运行中发现的永久几何预检错误同样不计入“五连错”。其余 CST/基础设施错误会累积：

1. 同一 Maid 连续 5 次报错时，Princess 保留这 5 次失败记录，Maid 退出；
2. Princess 自动通过 SSH 将该 Maid 重新叫醒一次；
3. 若重启后在尚未成功完成任何算例前再次连续 5 次报错，该 Maid 被隔离
   （`quarantined`），不再自动领取任务；
4. 任何一次成功完成都会清空连续错误以及“未成功前已自动重启”的计数。

任务使用租约和心跳；Maid 或网络意外中断后，过期任务会由 Princess 回收。默认
`--max-attempts 3`：可重试错误和租约过期都会占用一次 attempt，未到上限时任务重新排队，
达到上限时任务进入 `failed`，不会无限循环。Maid 的启动/hello 恢复也有独立的默认上限
`--max-recovery-launch-attempts 3`；耗尽后该设备在当前 Princess 进程中停止自动唤醒。

与恢复有关的其他默认参数为：`--startup-timeout-seconds 30`（等待新 Maid hello）、
`--stale-idle-seconds 30`（空闲 Maid 多久未刷新后按失联恢复）和
`--artifact-timeout-seconds 600`（单次大结果上传/完成请求的 HTTP 超时）。
`--artifact-commit-deadline-seconds 1800` 是一次结果的 upload 与 complete 共同使用的
总重试期限，默认给大 FFS 留出三倍单请求时间。Maid 对 hello、领任务及结果提交中的瞬时
网络错误使用退避重试；heartbeat 单次失败会记日志，并在下一周期继续发送。其中 hello
受启动期限约束，上传和完成提交始终使用同一 attempt/lease/hash 幂等重试，避免把
“Princess 已收到但应答超时”误报成一次 CST 失败。若总提交期限仍耗尽，Maid 会退出该
提交路径并保留 attempt 与 outbox，不发送 simulation failure；Princess 随后按租约过期
和最大 attempt 规则恢复任务，因此持续断网不会让该 Maid 永久续租卡住。

## 文件位置

Princess 主机上的持久化状态：

- `simulations/runs/<run-id>/source.csv`：冻结的原始输入；
- `simulations/runs/<run-id>/worklist.csv`：去除无效几何后的实际任务表；
- `simulations/runs/<run-id>/excluded_rows.json`：被排除行及原因；
- `simulations/runs/<run-id>/princess.sqlite3`：任务、租约、尝试、错误和 Maid 状态；
- `simulations/runs/<run-id>/princess_runtime.json`：本次运行元数据和 API token；
- `simulations/runs/<run-id>/incoming/`：上传暂存与完成回执；
- `results/raw/<run-id>/case_####/`：验收后的 `S11.csv`、
  `Farfield Source [1].ffs` 和 `manifest.json`。dry-run 结果只有 manifest。

每台远端 Maid 的工作区根目录为
`simulations/runs/<run-id>/workers/<device-id>/`：

- `worklist.csv`：该 run 的实际任务表，在同一设备的各次 launch 间共享；
- 第一次启动新 run/新设备时，launch 根目录就是上述工作区根目录；
- 恢复已有 run 且需要重新唤醒、五连错重启或离线恢复时，使用新的
  `launches/<launch-generation>/` 作为 launch 根目录，避免旧 Maid/CST 与新进程共用项目和
  输出目录；
- `<launch-root>/model/msa-bp.cst`：该次 launch 的独立 CST 项目副本；
- `<launch-root>/model/msa-bp/`：CST 在项目旁自动创建的结果侧车目录；
- `<launch-root>/maid_runtime.json`：只供该次 launch 读取的运行配置；
- `<launch-root>/output/attempts/<attempt-id>/`：单次 attempt 的临时结果；
- `<launch-root>/output/outbox/<attempt-id>.zip`：等待上传/确认的结果包；
- `logs/maid.<device-id>.<launch-id>.stdout.log` 和 `.stderr.log`：仓库根目录下的远端 Maid
  日志。

Princess 返回 `completed_ack` 后，Maid 会删除对应的成功 attempt 目录及 outbox ZIP；
失败、进程中断或清理失败留下的内容会保留，便于诊断。项目副本及 CST 侧车目录不会因此
自动删除。注册表中的 `simulations/runs/active_maid_runtime.json` 只是默认/独立 doctor
路径，不是 `start` 所创建 Maid 的实际 per-launch runtime 路径。

Princess 自身的实时日志目前输出到启动它的终端。上述运行目录、原始结果和日志均已由
`.gitignore` 排除，不应提交到 Git。

## 网络与安全边界

Maid 与 Princess 之间使用带随机 token 的普通 HTTP，而不是 TLS。它只适合运行在双方
均可信的 Tailscale tailnet 内；不要把端口 `8765` 转发到公网，也不要把
`bind_host` 改成面向不可信局域网或公网的监听地址。当前配置使用 Princess 的
Tailscale 地址 `100.99.182.30`；如果该地址变化，需要同时更新 `bind_host` 和
`advertise_url`，并确认两台 Maid 可以访问它。

`princess_runtime.json` 和分发到 Maid 的 runtime JSON 含本次 API token，应视为运行时
秘密；SSH 身份验证仍由系统 OpenSSH 密钥配置负责，设备注册表本身不保存 SSH 私钥。
