# 远场坐标与极化约定

本项目固定采用 CST 导出远场的球坐标约定：

- theta=0 deg 指向 +Z，即垂直基板和人体的 broadside/off-body 方向；
- theta=90 deg 是 XoY 平面，也就是平行于基板和人体的传播平面；
- phi 从 +X 朝 +Y 旋转；
- theta=90 deg、phi=90 deg 是本项目天线的正前方 endfire，即 +Y；
- theta=90 deg、phi=270 deg 是后方 endfire，即 -Y；
- theta=90 deg、phi=0/180 deg 分别是 +X/-X 方向。

在标准球坐标基矢下，theta=90 deg 时：

    e_r     = (cos(phi), sin(phi), 0)
    e_theta = (0, 0, -1)
    e_phi   = (-sin(phi), cos(phi), 0)

因此，在 theta=90 deg 的整个 XoY 平面上：

- E_theta 始终垂直于基板和人体，是垂直极化；
- E_phi 始终平行于基板和人体，是水平极化；
- 上述极化映射与 phi 无关。

符号方向在处理复场相位时必须保留；只计算功率时使用 |E_theta|^2 和
|E_phi|^2，正负号不影响结果。

## ffs_onbody_proxies.py 中的对应关系

- P_hor 和 P_hor80 中的 hor 表示 horizon belt，不表示水平极化；
- G_theta_hor 是 theta=90 deg 上对全部 phi 平均的垂直极化指标，不是
  phi=90 deg 的 endfire 指标；
- G_theta_hor_min 是全部 phi 中最弱的垂直极化指标；
- chi_TM 是 theta=90 deg 上对全部 phi 积分后的垂直极化功率占比；
- G_endfire_phi90_vertical、G_endfire_phi90_horizontal 和
  G_endfire_phi90_total 才是正前方 endfire 的三个分量；
- chi_vertical_endfire_phi90 是正前方 endfire 的垂直极化功率占比。

当 theta 小于 90 deg 时，E_theta 不再严格平行于 Z 轴。因此 P_hor/P_hor80
覆盖的有限 theta 带内，E_theta 应称为球坐标 theta 分量；只有在 theta=90 deg
边界上才能严格称为垂直极化。
