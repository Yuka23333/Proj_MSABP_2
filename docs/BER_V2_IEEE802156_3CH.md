# BER 2.0：IEEE 802.15.6 三信道指标

BER 2.0 固定使用 IEEE 802.15.6 低频段的三个相邻信道：

| 信道 | 中心频率 | 带宽 | 频率范围 | 角色 |
|---|---:|---:|---:|---|
| ch0 | 3494.4 MHz | 499.2 MHz | 3244.8–3744.0 MHz | 扩展信道 |
| ch1 | 3993.6 MHz | 499.2 MHz | 3744.0–4243.2 MHz | 必须信道 |
| ch2 | 4492.8 MHz | 499.2 MHz | 4243.2–4742.4 MHz | 扩展信道 |

在相同的发射端参考 `Eb/N0` 下，定义每个信道的高值优效用为

```text
Xi = 1 - BERi
```

三信道指标为

```text
J = (1 - lambda) * X1 + lambda * min(X0, X1, X2)
```

输出同时给出与其严格等价的低值优 BER：

```text
BER_v2 = (1 - lambda) * BER1 + lambda * max(BER0, BER1, BER2)
```

因此 `lambda=0` 只考察必须信道 ch1，`lambda=1` 只考察三个信道中的最差者。
脚本不提供隐式默认值；正式运行时必须显式记录 `lambda`。

## 运行

```powershell
python scripts\postprocessing\ber_06_run_ieee802156_3ch.py --robustness-lambda <LAMBDA>
```

可用 `--input-dir`、`--binary-input` 和 `--output-dir` 覆盖默认数据位置。
结果目录包含逐信道 BER、聚合 BER、目标 BER 所需 `Eb/N0` 汇总、原始重复计数和
完整 manifest。

这里严格对齐的是三个中心频率和 499.2 MHz 信道带宽。为保证和 BER 1.x 做受控
对照，当前仍沿用项目既有的 antipodal pulse、理想匹配滤波器、公共比特序列和公共
噪声假设；这不应表述为完整实现了 IEEE 802.15.6 PHY。

`ieee802156_three_channel.py` 还提供统一的二次多项式频率窗：

```text
w(f) = max(0, 1 - (2 * (f - fc) / BW)^2)
```

窗口在中心为 1、两端为 0，积分后归一化。后续 S21、Rad_Eff 和平面辐射代理均应
调用该公共实现，不再分别定义带内平均规则。
