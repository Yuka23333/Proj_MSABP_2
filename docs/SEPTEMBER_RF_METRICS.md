# September 宽带指标

September 是第二阶段独立使用的宽带归约口径。第一阶段脚本、结果与指标定义
保持不变。

当前只实现两项已经批准的修改：

1. S21 在每个 IEEE 802.15.6 信道内对线性功率传输 |S21|^2 做二次多项式
   频率加权，聚合完成后才转换为 dB。
2. Rad_Eff 先把 CST 导出的 dB 样本逐点转换成线性效率，再直接对效率做相同的
   频率加权；不会对 dB 做算术平均，也不会用原始功率之比替代效率平均。

两个指标都保留 ch0、必须信道 ch1、ch2 和最差信道，并使用：

    M_September = (1-lambda)*M_ch1 + lambda*min(M_ch0,M_ch1,M_ch2)

两个指标均为高值优。脚本要求显式传入 lambda。

## S21

    python scripts\postprocessing\september_rf_metrics.py s21 --robustness-lambda <LAMBDA>

## Rad_Eff

    python scripts\postprocessing\september_rf_metrics.py rad-eff --input-dir <RUN_DIRECTORY> --robustness-lambda <LAMBDA>

两种模式均可用 --output 指定 CSV，并用 --overwrite 显式覆盖已有输出。
每个结果表旁边都会写出独立 manifest。
