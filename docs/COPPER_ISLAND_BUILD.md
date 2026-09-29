# 独立铜片／内部孤岛：导出与 CST 建模

## 范围与现状

本次增加的是**所有铜片分块的保留和建模能力**，不是二级枝条生成器，也不修改采样合法域。
现有左右镜像保持 `(x, y) → (-x, y)`，数学上为关于 Y 轴对称。

设计意图仍是：槽不切穿金属外边界；真正的内部孤岛由二级枝条相交围出。
测试样例故意把 `BRANCH_UP_1_K3` 与 `BRANCH_DOWN_1_K3` 同时设为 `1`，其余保持
`shapely_antenna_model.DEFAULT_PARAMETERS`，以切穿外边界来强制触发分块。
**这只是建模压力测试，不能作为合法设计或二级枝条已经实现的证据。**

0.01 mm 源坐标量化后，压力测试共有 5 块铜：

| 序号 | 平面连接关系 | 面积 / mm² |
| --- | --- | ---: |
| 0 | 馈电主体 | 553.632 |
| 1 | 左侧浮置铜片 | 186.638 |
| 2 | 左侧 SMA ground 焊盘接触铜片 | 214.870 |
| 3 | 右侧 SMA ground 焊盘接触铜片 | 214.870 |
| 4 | 右侧浮置铜片 | 186.638 |

总面积 `1356.648 mm²`；按 `0.035 mm` 铜厚，总体积 `47.48268 mm³`。
这里的“浮置”仅基于与馈线及平面焊盘区域的连接判别，非完整三维电连接验证。

## 复现图与完整曲线

```powershell
python scripts/geometry/prepare_copper_island_case.py
```

- 完整参数、原始三条源曲线及所有铜片曲线：`data/geometry/copper_island_stress.json`。
- 默认几何与分块对照图：`results/figures/copper_island_stress.png`（生成物不推送）。
- 不覆盖现用 `results/processed/antenna_polygon_vertices.json`，不启动 CST 或求解。

## JSON 扩展

保留原有 `meta` 和 `vertices.{Patch,Slot,CPW_Feed_Pin}`。
可选 `conductor_components` 为最终 `(Patch − Slot) ∪ CPW_Feed_Pin` 的**全部**连通铜面：

```json
{
  "conductor_components": [
    {"exterior": [[0, 0], [1, 0], [1, 1]], "holes": []}
  ]
}
```

以上仅表示结构，非可运行的天线数据。每条曲线有序、不重复末尾闭合点，单位 mm。
第 0 块含完整馈线，其余按几何位置稳定排序。保留每块的所有孔洞，不能只存 exterior。
`role` 与 `area_mm2` 是附加说明；CST 按坐标重算面积，不信任附加面积值。

如果源槽本身因闭环带孔，额外用 `source_holes: {"Slot": [hole_curve, ...]}` 保留孔洞；
有 `source_holes` 的导出必须附带 `conductor_components`。否则只存槽外圈会把内部孤岛填掉。
源曲线先量化，再做布尔运算；保留新交点的精度，不再二次量化，避免引入裂缝。

调用 `polygon_export_payload(..., include_conductor_components=True)` 启用扩展；
有多块铜或源孔洞时才加入扩展字段。采样参数直接进入 CST builder 时已启用此模式。
原有三曲线文件及单连通默认几何保留原建模路径。扩展数据会检查分块是否有效、缺失、
重叠、与源曲线布尔结果不一致，以及第 0 块是否真的包含馈线。

## CST 行为

- 主体沿用 `component1:msabp_patch_solid`。
- 其余每块在 `component1_msabp_islands` 下单独拉伸为 `msabp_island_001` 等。
  该名字表示非馈电主体分块，不代表其中每块都电气浮置；接地分块也保留。
- 每块先拉伸外圈，再逐个减去孔洞工具，避免填掉内部孤岛或重复画铜。
- 全部分块曲线保留在 `msabp_conductor_components` 中；原三条源轮廓仍保留供查看。
- 基板、反射板及连接器避让槽沿用原代码；不删除／重建 SMA、端口或监视器。
- 每次重建清理专属分块 component／curve，包括从多块恢复成单块时，避免旧孤岛残留。
- 新分块路径使用 `model3d.add_to_history` 记录结构修改，不回退为无历史的即时几何修改。
- 校验每块闭合曲线、材质、体积及全部铜片的总体积，不能只校验主块。

准备好的隔离测试项目可使用：

```powershell
conda run -n cstpy python scripts/automation/cst_build_msabp_geometry.py `
  --project "<隔离测试目录>\copper-islands.cst" `
  --polygon-json data/geometry/copper_island_stress.json
```

加 `--dry-run` 可只打印全部 VBA。**不要把这组压力参数送入正式采样任务。**
本次未扩展传播双天线的镜像／搬移对象列表；该流程使用分块铜之前仍需单独适配。

## 验证记录（2026-09-24）

- 对照图生成并目视检查；全部分块、左右对称性、面积通过测试。
- 另有合成的闭环槽回归：不切穿外边界，保留主铜片孔洞和其内面积 `132 mm²` 的孤岛。
  这是导出／建模接口测试，不是声称二级枝条生成已经完成。
- 相关 Python 回归测试通过；CST 指令序列经 mock 检查，包括历史写入和逐块校验。
- **实机建模／保存尚未验证**：隔离 `.cst` 检查停在连接 Design Environment，尚未进入
  建模命令；仅终止本次启动的 Python/CST 进程，没有启动求解。原模型与未修改隔离副本
  SHA-256 一致：`bbc90824d50364619fe25484b60c492c6cb67487916215b1113d604514f1ee7e`。
