# Princess / Maid 分布式 CST 仿真

这套流程用于把同一份采样 CSV 分发给多台 Windows/CST 主机并行计算。入口是
[`princess.py`](princess.py)；[`maid.py`](maid.py) 通常不需要手工启动。

## 实际拓扑

```text
Princess（本机）
  ├─ SCP：短连接分发 worklist、CST 项目副本和 Maid runtime JSON
  ├─ SSH：只做部署/doctor，不承载常驻进程
  └─ JSON/TCP + HMAC：敲响 8766 端口的 Maid Bell
                                      │
                                      ▼
               Maid Bell（Windows 服务或开机 S4U 计划任务）
                                      │ 本地创建进程
                                      ▼
                            Maid（远端本地进程）
                              ├─ 本机 cst.interface → 本机无头 CST
                              └─ 主动 HTTP hello、领任务、心跳、上传结果
```

`convallariag5` 和 `coconutg2` 已实测：Windows OpenSSH 会在 SSH 断开时清理该会话创建
的完整进程树，所以 `Start-Process` 返回 PID 也不能让 Maid 常驻。默认 `launch_mode`
因此改为 `bell`：Princess 仍用 SSH/SCP 安全部署文件，但叫醒动作交给已经在目标机本地
运行的 Maid Bell。这里的计划任务仅负责常驻托管 Bell，与设备注册表中直接叫醒 Maid 的
旧 `launch_mode=scheduled_task` 不是一回事；Princess 侧仍统一使用 `launch_mode=bell`。
`ssh_process` 只保留为调试模式。

Bell 的机器配置不保存长期密码。Princess 每个 run 生成的随机 API token 会随
`maid_runtime.json` 经 SCP 到达远端；TCP wake 请求用同一 token 做 HMAC 签名。Bell 只
接受指向该仓库 `simulations/runs/**/maid_runtime.json` 的请求，只能启动固定的
[`maid.py`](maid.py)，并且同一时间只允许一个 Maid。Bell 返回 PID 只是启动回执；Princess
仍必须在期限内收到 Maid 主动发来的 HTTP hello 才确认其真正存活。

这条路径不要求 TeamViewer 持续在线，也不要求 CST 出现在交互桌面。这里的“本机”始终
是相对于 Maid 所在主机而言；Princess 不通过 SSH 持续控制 CST。

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
- `launch_mode`：常规远端使用 `bell`；
- `ssh_target`、`repo_root`、`python_path`：SSH 地址和远端本地路径；
- `bell_host`、`bell_port`：Tailscale/MagicDNS 下的 Bell 地址，默认端口为 `8766`；
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
  --bell-host new-host `
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

### 0. 每台远端一次性安装 Maid Bell

先在普通终端把精简环境更新到包含 `pywin32`：

```powershell
cd D:\Academic\Proj_MSABP_2
C:\Users\telecom\miniforge3\Scripts\conda.exe env update `
  --name maid `
  --file maid.yaml
```

然后通过 TeamViewer 在目标机打开“以管理员身份运行”的 PowerShell，各执行一次与主机名
匹配的命令：

```powershell
# convallariag5 上
powershell -ExecutionPolicy Bypass -File scripts\simulation\install_maid_bell.ps1 `
  -DeviceId convallariag5

# coconutg2 上
powershell -ExecutionPolicy Bypass -File scripts\simulation\install_maid_bell.ps1 `
  -DeviceId coconutg2
```

安装器会生成 `%ProgramData%\MSABP Maid Bell\bell.json`、创建只允许 Tailscale IPv4
(`100.64.0.0/10`) 访问 TCP 8766 的防火墙规则、启动并本机 ping。默认
`-HostMode Auto`：优先注册延迟自动启动的 Windows 服务；如果 Smart App Control/WDAC
拒绝未签名的 pywin32 `pythonservice.exe`，安装器会删除失败的服务并自动降级为开机启动
的 S4U 计划任务。两种托管方式运行完全相同的 Bell TCP 协议，脚本也都可以重复执行。

需要跳过服务尝试时可显式指定：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\simulation\install_maid_bell.ps1 `
  -DeviceId coconutg2 `
  -HostMode ScheduledTask
```

S4U fallback 默认使用本机 `telecom` 账户，不保存密码，也不需要用户保持登录；它没有访问
需要 Windows 用户凭据的 SMB 共享的能力，但 Maid 与 Princess 的 Tailscale TCP 通信不
依赖这种凭据。

服务模式下 SCM 默认用 `LocalSystem` 启动。先用下面的分布式 dry-run 验证 Bell/Maid
基础设施；
如果它通过而真实 CST 因用户级 license/profile 权限失败，再在 `services.msc` 中把
`MSABPMaidBell` 的“登录”账户改为 `\.\telecom` 并重启。账户密码只应交给 Windows
Service Control Manager，不要写进仓库或命令行。

### 1. 只读 doctor

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py doctor `
  --device convallariag5 `
  --device coconutg2
```

`doctor` 通过 SSH 只读检查远端仓库、Maid Python、Python 依赖、仓库中的 CST 模板、
入口脚本、Maid Bell 和注册表默认位置的 runtime JSON，不会启动 CST。第一次只完成
`git pull`、
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

`--dry-run` 仍会完整测试 CSV 冻结与过滤、SCP、Bell 唤醒与 hello 确认、Maid 主动回连、
任务租约、几何预检、结果打包上传和 Princess 落盘，但不会连接或启动 CST，也不会产生
S11/FFS。
dry-run 会把任务标记为完成；真实仿真必须使用新的 `--run-id`。

### 3. 运行真实仿真

第一次做真实双机 smoke 时，推荐直接在 IDE 中打开
`scripts/simulation/run_remote_real_smoke.py` 并按 F5。脚本顶部集中放置 run ID、设备和样本数；
默认生成 4 个 Sobol 样本，保存 CSV 后再逐行重建几何复验，然后要求输入一次 `RUN` 才会
通过 `convallariag5` 与 `coconutg2` 的 Maid Bell 启动真实 CST。这个入口不会传入
`--dry-run`。

固定的 run ID 使意外中断后的再次 F5 能恢复原 Princess 状态。若采样配置或输入内容有意
改变，请先给 `RUN_ID` 换一个新值。实时终端输出同时追加到
`logs/princess.<run-id>.real-smoke.log`，便于完整回传报错。若只想检查输入而不启动 CST：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\simulation\run_remote_real_smoke.py --prepare-only
```

命令行无人值守运行时可追加 `--yes` 跳过确认；IDE/F5 默认保留人工确认。

真实 Maid 把 `msa-bp.cst` 视为仿真 setup 的唯一来源。每个算例先执行 `DeleteResults`，
重建参数化几何，再只读校验 Port 1、2--8 GHz/0.1 GHz 的 61 个 Farfield Monitor 和
HF Time Domain solver，最后启动求解。Maid 不再删除或重建 Port、Monitor、网格及 solver
设置；模板不完整时该算例会在求解前明确失败。

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

Princess 覆盖远端已存在的 worklist/runtime 时使用同目录、哈希校验后的强制移动；不再
调用 Windows PowerShell 5.1 在两台 Maid 上均会报“路径格式不合法”的
`[System.IO.File]::Replace(..., $null)`。

### 3.1 双天线传播 13-case 任务

传播任务使用独立入口 [`run_propagation_13.py`](run_propagation_13.py)，任务表由
[`prepare_propagation_13.py`](prepare_propagation_13.py) 生成：12 个纯几何 k-medoids
代表解加候选 #1，共 13 个 case；#35 不重复加入，因为已保留肉眼几乎相同的 #34。

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\simulation\prepare_propagation_13.py

C:\Users\David\.conda\envs\cstpy\python.exe `
  scripts\simulation\run_propagation_13.py
```

两台 Maid 的 `simulations\models\msa-bp-propagation.cst` 是各自主机上的权威模板，必须
已经手工配置好 SMA/接口、Port 1/2、Muscle 幻象、三个 E-field Monitor 和 solver。
Princess 不会从本机上传 `.cst` 覆盖它们，而是要求每台 Maid 在本机原子复制模板，创建
该 launch 的私有项目副本。每个 case 只清除旧结果并重建 `component1` 与
`component1_1` 中的天线金属、基板和反射板，再把第二副天线关于 Y=0 镜像并沿 Y 平移
300 mm；基础设施对象完全不改。

Princess 每个 case 只接收 `S21.csv` 和 `manifest.json`。由 Port 1 激励得到的 CST 原生
E-field `.m3d/.rex` 文件保存在运行该 case 的 Maid 本地：

```text
<launch-root>/output/local_only/case_<sample-id>/e_field_native/
```

成功上传后 Maid 只清理临时 attempt 与 outbox，不删除上述 `local_only` 目录。发生 Maid
重启时，新一代 launch 有自己的 `local_only`；最终 E-field 归档需要按 manifest 记录的
绝对目录到对应主机取回。

### 4. 查看状态

在另一个 PowerShell 窗口中，或运行结束后执行：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\princess.py status `
  --run-id doe-001
```

输出包含 `completed`、`running`、`pending`、`failed`、总数，以及每台 Maid 的状态。

### 5. 紧急一键下班

Princess 或 CST 卡住时，可在 IDE 中直接 F5 运行，或从仓库根目录执行：

```powershell
C:\Users\David\.conda\envs\cstpy\python.exe scripts\simulation\dismiss_all_maids.py
```

脚本只处理各 Maid Bell 此刻明确报告为运行中的 runtime，不扫描或修改其他历史 run。它先
在对应 Princess SQLite 中设置持久化停机标记，将当前租约原子释放回 `pending` 并退还
本次 attempt 额度，再通过带当前 run token 签名的 Bell `stop` 命令结束 Maid 以及它
拥有的 CST/DBStorage 进程树。CSV、已完成结果和数据库不会删除。再次显式使用同一 run ID
执行 `princess.py start` 时，Princess 才会清除停机标记并恢复调度。

Bell 服务必须已经 pull 到包含 `stop` 协议的版本并重启一次；否则旧服务会返回
`unsupported Bell command: 'stop'`，脚本会保留错误报告而不会改用未经认证的远程强杀。

## CSV 过滤与失败策略

Princess 首先冻结原始 CSV，再在本机生成实际下发的 `worklist.csv`。采样器已经标记为
`geometry_valid=false` 的行会在 Princess 侧直接排除，原因写入
`excluded_rows.json`；这些行不会发给 Maid，也不会计入 Maid 的连续错误次数。

Maid 运行中发现的永久几何预检错误同样不计入“五连错”。其余 CST/基础设施错误会累积：

1. 同一 Maid 连续 5 次报错时，Princess 保留这 5 次失败记录，Maid 退出；
2. Princess 自动通过 Maid Bell 将该 Maid 重新叫醒一次；
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
- `results/raw/<run-id>/case_####/`：验收后的 `S11.csv`、`Rad_Eff.csv`、
  `Tot_Eff.csv`、`Farfield Source [1].ffs` 和 `manifest.json`。dry-run 结果只有 manifest。

`princess.py start --results-root <path>` 可把多个互不相同的静态 batch run
汇入同一个结果根目录，供 qLogEHVI 等迭代控制器使用。该路径会写入
`princess_runtime.json` 并在恢复时严格核对，不能用同一 run id 静默改投其他目录。

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
- `simulations/runs/maid-bell.<device-id>.state.json`：Bell 当前监督的 Maid PID 和 runtime；
- `logs/maid-bell.<device-id>.log`：Windows 服务日志；Bell 的无密钥机器配置位于
  `%ProgramData%\MSABP Maid Bell\bell.json`，不在 Git 仓库中。
- `logs/maid-bell.<device-id>.task.log`：S4U 计划任务托管 Bell 时的启动/异常日志。

Princess 返回 `completed_ack` 后，Maid 会删除对应的成功 attempt 目录及 outbox ZIP；
失败、进程中断或清理失败留下的内容会保留，便于诊断。项目副本及 CST 侧车目录不会因此
自动删除。注册表中的 `simulations/runs/active_maid_runtime.json` 只是默认/独立 doctor
路径，不是 `start` 所创建 Maid 的实际 per-launch runtime 路径。

Princess 自身的实时日志目前输出到启动它的终端。上述运行目录、原始结果和日志均已由
`.gitignore` 排除，不应提交到 Git。

## 网络与安全边界

Maid 与 Princess 之间使用带随机 token 的普通 HTTP；Bell wake 使用带 HMAC 的 JSON/TCP，
两者都不额外套 TLS。它们只适合运行在双方均可信的 Tailscale tailnet 内；不要把端口
`8765` 或 `8766` 转发到公网，也不要把
`bind_host` 改成面向不可信局域网或公网的监听地址。当前配置使用 Princess 的
Tailscale 地址 `100.99.182.30`；如果该地址变化，需要同时更新 `bind_host` 和
`advertise_url`，并确认两台 Maid 可以访问它。

`princess_runtime.json` 和分发到 Maid 的 runtime JSON 含本次 API token，应视为运行时
秘密；Bell 的 HMAC 也使用这个临时 token。SSH 身份验证仍由系统 OpenSSH 密钥配置负责，
设备注册表本身不保存 SSH 私钥或 Bell 密钥。
