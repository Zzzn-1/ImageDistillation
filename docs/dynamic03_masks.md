# Dynamic03：人物、汽车类和骑行两轮车 mask

脚本：`scripts/Dynamic03_generate_masks.py`。默认输入 `dataset/Dynamic03_raw/`，按文件名中的数字顺序读取连续帧。
删除规则、时序延续、携带物关联和断点续跑通过 31 项回归测试；对索引 1873～1878 的 6 帧执行了 CPU YOLO/SAM2 推理检查，尚未全量重跑增强规则。

## 删除规则

| 目标 | 默认规则 |
| --- | --- |
| person | 所有检测到的人都删除，包括行人、骑行者、静止的人 |
| bicycle / motorcycle | 与同一个人多次满足骑行位置关系及相机补偿后的共同运动，才自动删除；不再被整条轨迹的运动比例否决 |
| car / bus / truck | 所有检测到的目标都删除，包括停放、移动、没有运动证据或尚未形成稳定轨迹的目标 |
| 人物携带的 backpack / handbag / suitcase / book | render 补检，要求与人物位置和 mask 邻接关系相符，再加入删除区域 |
| 停放或骑行证据不足的 bicycle / motorcycle | 基础判断为保留；render 可将已确认删除车辆的决定短时延续到匹配的新 ID，两轮车仍需复核 |

car、bus、truck 不再使用运动比例筛选，停放的汽车、公交车和卡车也删除。
“删除”指输出白色修复 mask，不直接改动原图；真正去除像素需要后续修复步骤。

YOLO11 的 COCO 模型没有“骑行者”或“停放车辆”这两种类别。SAM2 负责根据提示分割轮廓，也不会决定哪些对象应该删除。
因此不能简单地把六种类别的 mask 全部合并，也不能仅凭人与车的框相交就认定骑行。

## 方法与边界

1. YOLO11-seg 逐帧检测六类目标；人物、两轮车、汽车类分别使用 BoT-SORT，避免人物与车辆互相接管 ID。bicycle 和 motorcycle 共用一组，允许两者之间的类别抖动。保留所有原始检测，包括尚未形成稳定轨迹的检测，避免跟踪确认阶段丢掉 person。
2. 默认比较相隔 5 帧的图像。排除目标框后，用背景 KLT 特征匹配和 RANSAC 单应矩阵估计相机运动。
3. 在目标的 YOLO mask 内跟踪特征，减去相机预测位移和附近背景的残余位移。要求足够特征、背景拟合质量、空间覆盖及目标内部位移一致性。人物和两轮车光流不足时，允许使用相机补偿后的框中心位移作为备用证据；要求检测置信度至少 0.35、框不贴图像边缘、预测与实测尺寸比例在 0.65～1.55 内，且采用更高位移阈值。汽车类不使用该备用方法。
4. 从首次有效证据开始取票，后续票间隔至少 5 帧，不再只选固定的 `index % 5 == 0` 帧。汽车类无条件删除，运动票不参与其删除决定。两轮车要求与同一个 person 至少 3 张共同运动票，每张票都必须满足两者自身运动及位移一致性；人车票单独计数，不受整轨迹运动比例限制。
5. 汇总整条轨迹的决定。人物和汽车类在所有检出帧删除。经确认骑行的两轮车轨迹在其所有检出帧删除，包含同一轨迹中暂时停车的帧；如果同一两轮车后来无人停放，应对停放帧使用 keep 逐帧覆盖。
6. SAM2 使用对象框、YOLO 内部正点和其他保留／删除对象的负点细化。低分、空分割或明显偏离 YOLO 的结果回退到 YOLO mask。
7. 对保留的两轮车也做分割，将 SAM2 与 YOLO 轮廓并集作为保护区；合并和膨胀后再次排除保护区。已识别的人物及其携带物像素优先删除。

## 快速骑行目标和人物携带物

render 默认启用相邻帧骑行补判，兼容原有 v3 分析缓存：当 5 帧间隔光流跟不上快速目标、人物贴边，或背景单应矩阵的覆盖检查失败时，使用相邻帧的可靠 KLT 特征估计人与车相对背景的运动。背景必须有至少 30 个有效点、覆盖至少 3 个网格，且平移残差的 80% 分位不超过 2.5 像素。人与车都要显著运动、方向和幅度相近、符合已有骑乘几何关系，同一人车配对连续两次通过才补为 remove。人工 keep 优先，补判只作用于当前帧，不将整条旧轨迹改成 remove。

对于已确认删除的两轮车，如果另一个更小的两轮车 mask 至少 95% 被它包含，会把该检测当作重复片段一并删除，避免其 keep 保护区抵消车轮等区域。人工 keep 可以阻止此规则。`--no-rider-recovery` 同时关闭这两项增强。

用户报告的 `1733458836_529432058.png` 中，原分析的 413、419、421 都是 keep，413 没有骑行票；419 的 mask 有 98.46% 落在 413 内。新预览通过相邻帧运动补判删除 413/421，处理重复片段 419，同时保留停放摩托车 380/390。

携带物默认复用 `models/yolo11x-seg.pt`，执行整图和较大人物区域的裁剪检测。背包、手提包、箱包、书本只有在人物附近且与人物 mask 接近时才删除；不把整个人物框涂白。`--carried-conf 0.15` 控制检测阈值，`--carried-yolo` 可替换 COCO 分割权重，`--no-carried-items` 关闭补检。遮挡、检测错误或分割不完整仍可能漏删／误删。

**COCO 没有 newspaper 类别，不能承诺默认检测器自动找全报纸。** 两种显式支持方式：

1. `--carried-world models/yolov8s-worldv2.pt`：使用本地 YOLO-World 权重按 `newspaper`、`sheet of paper` 等文字检测，再交给 SAM2 分割。当前工作区没有该权重，环境也没有 CLIP；需先准备权重并在 maskgen 环境安装 `pip install git+https://github.com/ultralytics/CLIP.git`。首次文字编码还可能下载 CLIP 权重。此自动报纸检测路径尚未做真实图片验证。
2. `--carried-prompts configs/你的携带物提示.json`：直接指定漏检物体的框，使用当前 SAM2 分割，不需要新增模型。例如：

```json
{
  "实际图片文件名.png": [
    {"box": [100, 120, 150, 180], "label": "newspaper"}
  ]
}
```

框是原图像素坐标 `[左, 上, 右, 下]`，请替换示例数值。提示框也必须分割成功并与人物关联才会加入 mask，不会将框直接填白。修改提示后使用新输出目录。

可直接复用完整分析生成增强结果：

```bash
python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis \
  --output outputs/dynamic03_masks_refined \
  --temporal-hold 5 --dilate 2 --overlay-every 1
```

检查结果：`outputs/dynamic03_refinement_check/overlays/1733458836_529432058.png` 是本次实际生成的示例。
`manifest.jsonl` 会记录 `rider_recovery`、`duplicate_of`、携带物的 `label` 和 `carried_by`，方便核对为何被删除。

## render 中断后继续运行

使用原命令加 `--resume`，保持分析、覆盖规则、提示内容和其他参数一致：

```bash
python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis \
  --output outputs/dynamic03_masks_refined \
  --temporal-hold 5 --dilate 2 --overlay-every 1 --resume
```

程序检查 manifest 中连续完成的帧及输出文件，从第一张未完整保存的帧继续。为恢复时序状态，会先回放已完成帧的光流和决定，但不会重跑这些帧的 YOLO/SAM2，也不会重写已完成图片。最后一行记录被中断截断时自动恢复；参数或轨迹决定变化则拒绝混用。

如果继续的是增强前的 `outputs/dynamic03_masks`，需加 `--no-rider-recovery --no-carried-items --resume` 才与旧参数相符。这只继续旧规则；要修正先前漏删的自行车和背包，应按上面的增强命令输出到新目录。

**这是可复核的启发式基线，不能保证“停放两轮车零误删”。** 保护区依赖检测和分割本身正确。强视差、相机与目标同速运动、低纹理、漏检、同组内的轨迹 ID 切换，以及行人与停放车辆靠得很近，都可能导致误判。框位移备用证据也可能受到检测抖动影响。背景单应模型加局部补偿不等于完整的三维运动估计。

两轮车全程停着但有人骑坐时，自动规则倾向保留车辆；人仍删除。应复核后把该车辆轨迹设为 `remove`。推车和骑车也不能仅靠这套几何规则可靠区分。严格区分这些动作可进一步引入姿态／动作识别或人工标注。

尚未获得跟踪 ID 的目标使用一次性的负数 ID：person、car、bus、truck 仍删除，两轮车基础判断为保留。render 的时序关联也适用于负数 ID，可用逐帧覆盖修正。负数 ID 本身不代表跨帧的同一个对象。

本版使用 **SAM2 图像预测器逐帧分割，没有 SAM2 视频传播**；某帧 YOLO 完全漏检时，不会凭轨迹自动生成这一帧的 mask。下面的时序功能延续删除决定，最终分割仍使用当前帧检测和 SAM2。

## render 中跨 ID 延续删除

默认启用 `--temporal-hold 5`。同一辆两轮车连续两帧被基础规则或人工覆盖设为 remove，且相邻帧关联成立后，可以将删除决定延续到后面的匹配检测，包括 bicycle/motorcycle 类别变化和跟踪 ID 变化。

- 使用双向 Farneback 光流将上一帧 YOLO mask 对齐到当前帧，排除前后光流不一致的像素。匹配分辨率的最长边最多 640 像素。
- 与当前帧两轮车 mask 的 IoU 至少为 `--temporal-iou 0.5`，双方必须互为最佳匹配，并比各自次佳候选至少高 0.1。检测重叠、身份不明确时不延续，不仅凭相同 ID 或相邻位置认定同一辆车。
- 没有新的基础 remove 或有效的人车共同运动证据时，最多延续 5 帧。延续所得的 remove 不会给自己续期，避免一轮错误不断扩散。新的人车共同运动证据可以给已确认的匹配链续期。
- 最多跨过两帧完全缺失的检测来重新关联，但缺失帧不生成额外 mask；持续遮挡或匹配失败仍可能漏删。
- 人工逐帧／轨迹 keep 优先，并清除匹配目标的删除历史。没有确认来源的停放两轮车保持基础 keep；光流误配仍可能误删，应检查新增删除区域。
- `--temporal-hold 0` 关闭时序延续；增大到 10 可延续更久，也会增加错误关联或骑行结束后继续删除的风险。

**已有 decision_version 为 3 的分析结果可以直接复用，无需重新 analyze。** 例如全量分析目录是 `outputs/dynamic03_analysis` 时：

```bash
python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis \
  --output outputs/dynamic03_masks_temporal \
  --temporal-hold 5 --temporal-iou 0.5 \
  --dilate 2 --overlay-every 1 --device cuda:0
```

仅适用于 1000～1299 预览分析的 `dynamic03_track55_remove.json` 不要直接用于全量分析；轨迹 ID 只在对应的那次分析中有效。复用预览时，可使用 `--analysis outputs/dynamic03_analysis_v3_1000_300 --overrides configs/dynamic03_track55_remove.json`，并指定新输出目录。

`manifest.jsonl` 中新增删除的检测有 `base_action: keep` 和 `temporal` 字段，记录来源 ID、来源帧、上一 ID、匹配 IoU、距离上次独立证据的帧数，以及是否由新骑行证据续期。`summary.json` 的 `temporal_removals` 是从 keep 改为 remove 的检测次数。输出 `tracks.json` 仍保存分析阶段的基础判断，最终逐帧决定以 manifest 为准。

## 已有预览升级到新版规则

本版使用 `decision_version: 3`，旧分析结果的汽车类决定可能是 keep，render 会拒绝直接使用。
请用新目录重新 analyze 和 render：

```bash
python scripts/Dynamic03_generate_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_all_cars \
  --start 1000 --limit 300 --overlay-every 1 --device cuda:0

python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis_all_cars \
  --output outputs/dynamic03_masks_all_cars \
  --dilate 2 --overlay-every 1 --device cuda:0
```

重新分析后需要复核轨迹 ID，不要直接复用旧的 overrides。
`tests/test_dynamic_mask_decisions.py` 覆盖静止汽车类删除、停放两轮车保留、骑行关联、类别变化及覆盖规则。

## 第一步：分析一段连续帧

下面的命令供你手动运行，本次未执行。使用现有 `maskgen` 环境，需要 `ultralytics`、OpenCV、NumPy、PyTorch，以及已安装的本地官方 SAM2 包。默认复用已有 `models/yolo11x-seg.pt` 和 `models/sam2.1_hiera_base_plus.pt`，不会主动下载模型。

```bash
conda activate maskgen
python scripts/Dynamic03_generate_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_preview \
  --start 0 --limit 300 \
  --device cuda:0 --overlay-every 5
```

选择一段同时有骑行和停放车辆的连续片段更有代表性，可修改 `--start`。不要先抽成间隔很大的稀疏帧集，光流和关联都依赖连续性。阈值是初始值，尚未在 Dynamic03 上验证。

输出：

- `overlays/`：检测框和轨迹 ID，第一阶段黄色框只表示检测结果，不表示最终删除决定。
- `tracks.json`：每条轨迹的类别、首尾帧、运动票、骑行关联票、默认 `action`、原因和 `review`。`rider_votes_by_person` 列出关联人物及票数；`motion_sources` 是逐帧证据来源计数（不是间隔采样票数）。`actions_by_class` 明确各类别的决定；异常混合类别轨迹可能显示 `action: mixed`，此时按类别决定，person、car、bus、truck 始终删除。
- `frames.jsonl`：每帧检测、相机估计状态、目标运动证据、原图路径。
- `seeds/`：压缩的 YOLO 二值实例 mask，第二阶段复用，无需重跑 YOLO；全量运行需预留磁盘空间。
- `config.json`、`summary.json`：参数和完成信息。只有分析完整成功才写入 `summary.json`。
- `overrides.example.json`：空的人工覆盖模板。

`review: true` 是提醒检查，不会阻止第二阶段执行。默认保留不等于已经证明静止。特别检查 `summary.json` 中背景拟合失败的数量；失败时不会用未补偿的位移冒充车辆运动。

## 第二步：修正轨迹决定

另存一份 JSON，例如 `configs/dynamic03_overrides.json`。以下 ID 和帧名仅用于示例，必须替换成当前分析中的实际值；不存在的 ID 或帧名会报错。

```json
{
  "tracks": {
    "42": "keep",
    "57": "remove"
  },
  "frames": {
    "1733458773_910068274.png": {
      "57": "keep"
    }
  }
}
```

`keep` 保护停放的两轮车，`remove` 用于人工确认有人骑乘但自动规则漏判的两轮车。两轮车优先级为：逐帧覆盖 > 轨迹覆盖 > 时序延续 > 自动决定。person、car、bus、truck 强制删除，不能用 keep 覆盖；混合类别轨迹的 keep 只影响其中的两轮车检测。不要将无人停放的两轮车设为 remove。轨迹 ID 只对本次分析有效；更换选帧范围、模型或重新分析后，不要直接复用旧 ID。

## 第三步：生成 SAM2 mask

不需要人工覆盖时省略 `--overrides` 这一行：

```bash
python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis_preview \
  --overrides configs/dynamic03_overrides.json \
  --output outputs/dynamic03_masks_preview \
  --device cuda:0 --dilate 2 --overlay-every 5
```

- `masks/原图主体.png`：白色 255 为待删除／修复区域，黑色 0 为保留区域。
- `raw_masks/`：保护处理后、尚未膨胀的删除区域。
- `colmap_masks/原图完整文件名.png`：与修复 mask 反色，黑色排除，白色保留。
- `overlays/`：红色覆盖最终删除像素；红框表示删除对象，黄框表示保留对象，附轨迹 ID。
- `manifest.jsonl`：每帧实际决定、SAM2／YOLO 回退情况和 mask 占比。
- `config.json`：本次渲染参数和实际覆盖规则；`tracks.json` 保存原自动判断，最终逐帧决定看 manifest。

先检查停放车辆是否保持完整、骑行者和车轮是否覆盖、人物与停放车辆相接处是否串分。修改 overrides 后，用新输出目录重新 render 即可，无需重新分析。不能靠增大膨胀半径修正对象选择错误。

## 全量处理

去掉 `--limit` 会分析所有剩余帧；全量分析的轨迹 ID 需要重新检查。

```bash
python scripts/Dynamic03_generate_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_full \
  --device cuda:0 --overlay-every 30

python scripts/Dynamic03_generate_masks.py render \
  --analysis outputs/dynamic03_analysis_full \
  --output outputs/dynamic03_masks_full \
  --device cuda:0 --overlay-every 30
```

如果全量结果需要人工覆盖，在第二条命令加上针对全量轨迹编写的 `--overrides`。新运行的输出目录必须为空或不存在；render 可用相同参数加 `--resume` 续跑，analyze 暂不支持续跑。原数据集不会被写入。

调参要点：`--motion-px` 单位为原图像素，是跨 `--motion-lag` 帧的残余位移，并非每秒速度。提高 `--motion-px` 或 `--min-votes` 会减少两轮车误删，也会增加骑行车辆漏删；`--moving-ratio` 仅为兼容旧命令保留，不再影响删除决定；增大 `--motion-lag` 有利于慢速运动判定，但会增大匹配难度。默认 300 帧预览应先用于检查这些取舍。

代码复用了本机 Ultralytics 的 BoT-SORT `update` 接口及检测索引映射；升级 Ultralytics 后需确认这个内部接口仍兼容。官方接口参考：[Ultralytics 跟踪说明](https://docs.ultralytics.com/modes/track/)、[SAM2 图像与视频预测](https://github.com/facebookresearch/sam2)。
