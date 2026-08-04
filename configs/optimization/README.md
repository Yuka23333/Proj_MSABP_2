# 天线采样配置

`antenna_sampling.json` 是参数采样器的默认配置，也是 IDE/F5 与命令行共享的
唯一配置入口。当前 `±2%` Global 范围只是用于验证框架能够运行的保守示例，
不是最终优化边界；后续应根据 CST 范围试验逐组替换。

## 继承规则

对每个变量分别使用：

```text
sampling.parameters.<name>.range
    > sampling.groups.<exact-group>.range
    > sampling.global.range
```

缺少的 `sample` 状态使用代码注册表默认值。未知参数、未知 Group、拼错字段、
空范围以及零 nominal 的相对范围都会立即报错。

## 枝条

枝条开关位于 `branches`。关闭时，其位置、长度和宽度仍保留 nominal，但不会
占用采样维度；开启时长度与宽度必须为正。`anchor_t` 沿父主干从起点到终点
取值，硬范围为 `[0, 1]`。

SMA 焊接禁入区是固定硬约束：`X=[-4.76, 4.76] mm`、
`Y=[0, 4.5] mm`。任何启用的 slot 枝条 Polygon 都不得覆盖该区域。

## 几何策略

- `coordinate_quantum_mm` 默认且建议保持 `0.01`；
- `reject_self_intersection` 必须为 `true`；
- `allow_disconnected_conductor` 默认 `false`，只应在明确允许金属浮岛的试验中开启。
