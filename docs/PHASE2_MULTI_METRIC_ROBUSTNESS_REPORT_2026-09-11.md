# Proj_MSABP_2 第二阶段研究报告：多指标传播代理的鲁棒性

**研究快照日期：** 2026-09-11  
**项目：** 参数化多槽背板天线（MSA-BP）的全波仿真、代理指标与多目标优化  
**当前结论状态：** 数值仿真结果已复核；尚未进行实物测量或有限尺寸人体模型验证

## 1. 执行摘要

本项目建立了一套从参数化几何、CST全波仿真、远场代理提取、分布式任务执行到多目标优化的完整流程。第二阶段不再使用已经废弃的polar-cap gain，而将贴体传播方向上的θ极化radiation gain作为传播代理，与最差带内S11和基板面积共同构成三目标K-RVEA优化。

主要结果如下：

1. 第二阶段K-RVEA完成64个新全波样本，64/64成功，无惩罚样本。
2. 从新一轮非支配解中选取6个具有代表性的候选进行独立双天线传播仿真；选择时未使用其S21或BER结果。
3. `krvea_0023`在以下四种链路指标中均排名第一：
   - September三信道加权S21；
   - September三信道BER-v2；
   - 旧版3.1–4.8 GHz均匀平均S21；
   - 旧版单个超宽脉冲BER。
4. 相对于当前权威的Roblin–Wei参考天线，`krvea_0023`在旧版指标下取得：
   - 均匀S21提升1.463 dB；
   - 达到BER = 1e-4所需的发射端参考Eb/N0降低1.967 dB。
5. 使用前37个传播样本重新拟合旧版指标专用ROI，并将新6点严格锁定为holdout：
   - 旧版BER排序6/6完全正确，Spearman ρ = 1.0000；
   - 均匀S21排序ρ = 0.8286；
   - 第一、第二和最后一名均预测正确。

因此，当前证据支持的核心主张不是“某一种加权指标具有唯一正确性”，而是：**该框架具有指标可替换性，并在多种链路质量定义之间表现出较强的跨指标鲁棒性。**

## 2. 研究问题与设计空间

研究目标是在保证超宽带匹配和可制造几何的同时，进一步减小天线尺寸并改善贴体传播。天线采用参数化槽和枝条结构，完整参数系统包含23个可优化变量；当前第二阶段固定部分结构，只激活以下11个变量：

- 绝对尺寸：
  - `SLOT_MAIN_LENGTH`
  - `PATCH_BRICK_1_SIDE_MARGIN`
  - `PATCH_BRICK_1_TOP_MARGIN`
  - `PATCH_BRICK_2_HEIGHT_MARGIN`
- 上部角结构：
  - `UPPER_CORNER_NOTCH_1_K1`
  - `UPPER_CORNER_NOTCH_1_K2`
  - `UPPER_CORNER_EAR_1_K1`
  - `UPPER_CORNER_EAR_1_K2`
- 上部枝条：
  - `BRANCH_UP_1_K`
  - `BRANCH_UP_1_K2`
  - `BRANCH_UP_1_K3`

所有几何坐标在建模前量化到0.01 mm。基板面积由设计变量解析计算，不训练额外GP，因此该目标的代理不确定度恒为零。

## 3. 仿真与分布式计算框架

### 3.1 单天线仿真

单天线CST任务负责生成：

- S11；
- radiation efficiency与total efficiency；
- Farfield Source（FFS）。

FFS用于离线计算空间辐射代理，因此大量代理分析不需要重新运行CST。

### 3.2 双天线贴体传播仿真

传播模型包含两个相同天线，间隔300 mm，位于开放边界的无限平面Muscle幻象上方。双端口仿真导出复数S21，并在本地保留E-field结果。当前结论仅适用于这一固定距离、姿态和无限平面幻象，不应直接外推到有限圆柱人体或真实人体。

### 3.3 Princess/Maid

Princess负责CSV进度、租约、失败恢复和设备调度；Maid在各设备本地连接CST并运行求解。当前使用的远程设备为：

- `convallariag5`
- `coconutg2`

CST工程能打开并不等价于可以跨机器求解。端口、pick、monitor、材料和历史树均可能产生可移植性问题；远程部署前仍需做本机求解冒烟检查。

## 4. 第二阶段三目标优化

第二阶段K-RVEA采用11维输入和三个最小化目标：

1. **最小化最差带内S11线性幅度**
   - 频带：3.1–4.8 GHz；
   - reduction：频带内最大`|S11|`。
2. **最大化固定ROI内的θ极化radiation gain**
   - 优化器内部最小化其负dBi值；
   - 频率：3.6 GHz；
   - θ：55°–85°；
   - φ：90° ± 50°；
   - 空间平均在线性功率域完成，并使用立体角权重；
   - Rad_Eff只吸收一次；不计入mismatch，避免与S11重复计数。
3. **最小化归一化基板面积**
   - 参考面积：2720.2 mm²；
   - 标称设计面积归一化为1。

历史训练集包含960个样本。新一轮预算为64点，按q = 4执行，共16个batch；所有64点成功完成。

## 5. 两套传播指标

### 5.1 September三信道指标

采用IEEE 802.15.6低频段的三个499.2 MHz信道：

- ch0：3494.4 MHz；
- ch1：3993.6 MHz，mandatory channel；
- ch2：4492.8 MHz。

每个信道内部使用归一化二次多项式频率权重。三信道通过

\[
J=(1-\lambda)X_1+\lambda\min(X_0,X_1,X_2),\qquad \lambda=0.5
\]

组合，其中`X1`为mandatory channel效用。所有平均首先在线性尺度完成，再转换为dB。

### 5.2 旧版宽带指标

旧版指标用于检验结论是否依赖September权重：

- **均匀S21：** 在3.1–4.8 GHz内对线性`|S21|`进行均匀算术平均，再计算`20 log10(mean)`；
- **旧版BER：** 使用5 Mbps、3.1–4.8 GHz单个超宽BPSK脉冲，经复数S21信道、iFFT和匹配滤波后，计算BER = 1e-4对应的发射端参考Eb/N0。

本批旧版BER的最大`|ISI/main|`约为7.7×10⁻⁹，因此排序主要反映接收脉冲能量，而不是高速符号混叠。

## 6. 新64点与传播holdout选择

64个新样本中：

- 64/64成功；
- 37个属于64点内部的三目标非支配集；
- 11个满足最差带内return loss不低于7 dB；
- 其中9个同时满足return-loss门槛并属于非支配集。

传播验证没有使用S21标签挑选“看起来最好的”样本。首先固定三个目标端点，再在归一化三目标空间使用确定性maximin补充三个覆盖点：

| Case | 选择角色 | Return loss | 固定ROI gain | 归一化面积 |
|---|---|---:|---:|---:|
| `krvea_0023` | ROI gain端点 | 9.749 dB | -5.350 dBi | 1.0637 |
| `krvea_0034` | S11端点 | 9.937 dB | -9.482 dBi | 1.0072 |
| `krvea_0046` | 面积端点 | 8.842 dB | -10.557 dBi | 0.9663 |
| `krvea_0050` | 非支配集覆盖 | 8.078 dB | -7.324 dBi | 1.0060 |
| `krvea_0015` | 非支配集覆盖 | 9.354 dB | -8.556 dBi | 1.0227 |
| `krvea_0061` | 非支配集覆盖 | 8.637 dB | -9.648 dBi | 0.9913 |

## 7. September指标结果

| Case | September S21 | BER-v2所需Eb/N0 | 归一化面积 |
|---|---:|---:|---:|
| `krvea_0023` | **-34.292 dB** | **50.392 dB** | 1.0637 |
| `krvea_0015` | -37.355 dB | 53.582 dB | 1.0227 |
| `krvea_0050` | -37.447 dB | 50.811 dB | 1.0060 |
| `krvea_0034` | -39.647 dB | 56.383 dB | 1.0072 |
| `krvea_0061` | -40.334 dB | 57.317 dB | 0.9913 |
| `krvea_0046` | -43.355 dB | 60.141 dB | 0.9663 |

固定September ROI在新6点上的Spearman相关性为：

- S21：ρ = 0.9429；
- `-Eb/N0`：ρ = 1.0000。

与前37点合并后，43点相关性为：

- S21：ρ = 0.9346；
- `-Eb/N0`：ρ = 0.9325。

`krvea_0023`在43个实测传播信道中同时刷新September S21和BER-v2最佳结果。相对于此前最佳点，其S21提升0.853 dB，所需Eb/N0降低2.202 dB。

## 8. 旧版指标结果与Roblin–Wei对照

当前权威Roblin–Wei参考及6个holdout的旧版指标如下。S21越接近0 dB越好，所需Eb/N0越低越好。

| 设计 | 均匀S21 | 相对参考 | 所需Eb/N0 | 相对参考 |
|---|---:|---:|---:|---:|
| **`krvea_0023`** | **-37.984 dB** | **+1.463 dB** | **42.970 dB** | **降低1.967 dB** |
| `krvea_0061` | -39.253 dB | +0.195 dB | 44.875 dB | 降低0.062 dB |
| Roblin–Wei | -39.447 dB | — | 44.937 dB | — |
| `krvea_0050` | -40.076 dB | -0.629 dB | 46.011 dB | 劣化1.075 dB |
| `krvea_0015` | -40.461 dB | -1.014 dB | 45.589 dB | 劣化0.653 dB |
| `krvea_0034` | -40.560 dB | -1.113 dB | 45.955 dB | 劣化1.018 dB |
| `krvea_0046` | -42.138 dB | -2.691 dB | 48.106 dB | 劣化3.169 dB |

`krvea_0061`与参考基本持平；`krvea_0023`的提升则同时出现在S21和BER中，并且幅度足以排除简单的名次抖动解释。

## 9. 旧版指标专用ROI：37点拟合、6点锁定验证

为检验当前传播代理是否只对September指标有效，使用前37个传播样本重新搜索旧版指标专用ROI。新6点在整个ROI搜索完成前保持不可见。

### 9.1 拟合结果

- field family：radiation gain；
- component：θ；
- θ：45°–100°；
- φ：90° ± 15°；
- 频率：3.8–4.4 GHz；
- 共搜索145299个候选区域。

训练集相关性：

- 均匀S21：ρ = 0.9471；
- `-旧版BER Eb/N0`：ρ = 0.9414；
- robust joint：0.9414。

### 9.2 锁定holdout结果

| ROI预测排名 | Case | 均匀S21实际排名 | 旧版BER实际排名 |
|---:|---|---:|---:|
| 1 | `krvea_0023` | 1 | 1 |
| 2 | `krvea_0061` | 2 | 2 |
| 3 | `krvea_0015` | 4 | 3 |
| 4 | `krvea_0034` | 5 | 4 |
| 5 | `krvea_0050` | 3 | 5 |
| 6 | `krvea_0046` | 6 | 6 |

Holdout相关性：

- 均匀S21：ρ = 0.8286；
- `-旧版BER Eb/N0`：ρ = 1.0000；
- BER排名6/6完全正确。

旧版ROI与September ROI的窗口不同，但二者都位于φ = 90°附近的端射区域，都由θ极化radiation gain主导，并且最终都选择`krvea_0023`。这表明具体平均规则会改变最优积分窗口，但没有改变当前数据支持的主要传播机制与冠军设计。

## 10. 当前科学解释

当前结果支持以下分层结论：

1. **强证据：** `krvea_0023`在两套S21定义和两套BER定义下均排名第一。
2. **强证据：** 端射区域的θ极化radiation gain能够跨指标预测传播表现。
3. **中等证据：** ROI的具体角度和频率边界依赖应用指标；September三信道与全带宽单脉冲不应强制共享完全相同的ROI。
4. **框架性结论：** 更换链路指标只需要重新标定代理区域，不需要改变几何参数化、K-RVEA、CST任务或Princess/Maid基础设施。

适合论文的保守表述为：

> Although the proposed channel-weighted metric is not claimed to be a universally optimal definition of link quality, the framework is deliberately metric-flexible. Distinct broadband link criteria produce different proxy regions while preserving the same endfire-polarized radiation mechanism and identifying the same best-performing design. A legacy-metric proxy fitted on 37 samples further reproduces the BER ranking perfectly on a six-sample holdout set. Alternative application-specific link metrics can therefore be incorporated through proxy recalibration without changing the geometry, optimization, or distributed simulation framework.

不建议将当前结论写成“对任意指标均普适”。更准确的说法是“在本研究评估的多种链路指标之间具有鲁棒性和可替换性”。

## 11. 当前研究决策

基于现有结果，暂不执行旧版ROI驱动的额外64点K-RVEA：

- 当前三目标优化已经找到同时支配Roblin–Wei参考的`krvea_0023`；
- 旧版ROI在独立holdout上成功恢复BER排序；
- 再跑64点主要用于研究不同指标下的完整Pareto前沿漂移，而不是证明当前方法有效所必需。

当前建议：

- 将`krvea_0023`冻结为主要候选；
- 将`krvea_0050`保留为September指标下的近标称面积折中解；
- 下一阶段优先分析两者的几何与场机制，或进行更接近真实人体的有限模型/实物验证。

## 12. 局限性

- 新holdout只有6点，而且是为覆盖三目标前沿而选择，不是独立同分布随机样本。
- 所有传播结果来自固定间距、固定姿态、无限平面Muscle幻象。
- 尚未覆盖人体曲率、组织分层、姿态变化、制造误差和实物测量。
- ROI搜索仍属于数据驱动的区域选择；其物理解释需要复数场、极化、相位和功率流证据进一步支持。
- CST工程跨主机可移植性有限；“能够打开”不保证端口、monitor、材料和求解状态均有效。
- `results/raw`和`results/processed`通常不进入Git，跨主机复现时必须单独传输并检查哈希。

## 13. 关键复现入口

### 第二阶段优化

- `scripts/optimization/run_phase2_krvea.py`
- `configs/optimization/phase2_krvea_roi_radiation_gain_64.json`
- `src/msabp_opt/optimization/phase2_krvea_data.py`
- `src/msabp_opt/optimization/phase2_krvea_relay.py`

### 第三批传播候选

- `scripts/simulation/prepare_propagation_active_learning_6_batch03.py`
- `scripts/simulation/run_propagation_active_learning_6_batch03.py`
- `data/samples/propagation_active_learning_6_batch03.csv`

### 新旧链路指标

- `scripts/postprocessing/september_rf_metrics.py`
- `scripts/postprocessing/ber_06_run_ieee802156_3ch.py`
- `scripts/postprocessing/ber_03_average_s21.py`
- `scripts/postprocessing/ber_04_run_experiment.py`

### 旧版ROI独立验证

- `scripts/postprocessing/fit_legacy_roi_validate_batch03.py`
- `results/processed/onbody_proxies/active_learning_batch03/legacy_roi_fit37_holdout6/legacy_roi_holdout_summary.json`
- `results/processed/onbody_proxies/active_learning_batch03/legacy_roi_fit37_holdout6/legacy_roi_holdout_validation_6.csv`
- `results/processed/onbody_proxies/active_learning_batch03/legacy_roi_fit37_holdout6/roi_fit_training37/result.json`

## 14. 参考数据身份

当前权威参考为`roblin_wei_2012_reference`，不得与`david_2_body_unverified_trial`混淆。

- 当前Roblin–Wei Touchstone SHA-256：  
  `8df86849fef40872aa8c48adb1f6e37d791d2d1a93b7769fd50348a4c121a027`
- 转换后参考`S21.csv` SHA-256：  
  `2eb5f9752a78fba857b0d6bb26df8d3f74af2862d36146b774b93adc740ab06e`
- 前37点FFS tensor SHA-256：  
  `b2c50765f968fe611a36746dee1a4e8a8d558a916e516a6221caea49da232cca`
- 新6点传播worklist SHA-256：  
  `d2611a37d6d212f30ef7617670bee58eded935cf1706a7dac83bed83981ff2a4`

上述参考S21哈希已同时与Touchstone转换manifest、均匀S21记录和旧版BER manifest核对一致。

