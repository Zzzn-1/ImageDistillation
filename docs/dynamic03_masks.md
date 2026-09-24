# Dynamic03：人物、骑行车辆和移动车辆 mask

脚本：`scripts/generate_dynamic_masks.py`。默认输入 `dataset/Dynamic03_raw/`，按文件名中的数字顺序读取连续帧。
本次只编写了脚本和说明，没有执行脚本、模型推理或测试，也没有生成任何 mask。

## 删除规则

| 目标 | 默认规则 |
| --- | --- |
| person | 所有检测到的人都删除，包括行人、骑行者、静止的人 |
| bicycle / motorcycle | 与同一个人多次满足骑行位置关系及相机补偿后的共同运动，才自动删除；不再被整条轨迹的运动比例否决 |
| car / bus / truck | 相机运动补偿后，有足够自身运动证据才自动删除 |
| 停放车辆、运动证据不足的车辆 | 保留，车辆轨迹全部标记为待复核 |

这里把“移动的”解释为同时修饰 car、bus、truck；停放的汽车、公交车和卡车也保留。
“删除”指输出白色修复 mask，不直接改动原图；真正去除像素需要后续修复步骤。

YOLO11 的 COCO 模型没有“骑行者”或“停放车辆”这两种类别。SAM2 负责根据提示分割轮廓，也不会决定哪些对象应该删除。
因此不能简单地把六种类别的 mask 全部合并，也不能仅凭人与车的框相交就认定骑行。

## 方法与边界

1. YOLO11-seg 逐帧检测六类目标；人物、两轮车、汽车类分别使用 BoT-SORT，避免人物与车辆互相接管 ID。bicycle 和 motorcycle 共用一组，允许两者之间的类别抖动。保留所有原始检测，包括尚未形成稳定轨迹的检测，避免跟踪确认阶段丢掉 person。
2. 默认比较相隔 5 帧的图像。排除目标框后，用背景 KLT 特征匹配和 RANSAC 单应矩阵估计相机运动。
3. 在目标的 YOLO mask 内跟踪特征，减去相机预测位移和附近背景的残余位移。要求足够特征、背景拟合质量、空间覆盖及目标内部位移一致性。人物和两轮车光流不足时，允许使用相机补偿后的框中心位移作为备用证据；要求检测置信度至少 0.35、框不贴图像边缘、预测与实测尺寸比例在 0.65～1.55 内，且采用更高位移阈值。汽车类不使用该备用方法。
4. 从首次有效证据开始取票，后续票间隔至少 5 帧，不再只选固定的 `index % 5 == 0` 帧。汽车类仍要求至少 3 张运动票、运动比例至少 60%。两轮车要求与同一个 person 至少 3 张共同运动票，每张票都必须满足两者自身运动及位移一致性；人车票单独计数，不受汽车类的整轨迹运动比例限制。
5. 汇总整条轨迹的决定。经确认的运动轨迹在其所有检出帧都删除，包含同一轨迹中暂时停车的帧；需要不同时间段不同规则时用逐帧覆盖。
6. SAM2 使用对象框、YOLO 内部正点和其他保留／删除对象的负点细化。低分、空分割或明显偏离 YOLO 的结果回退到 YOLO mask。
7. 对保留车辆也做分割，将 SAM2 与 YOLO 轮廓并集作为保护区；合并和膨胀后再次排除保护区。YOLO 已识别的人物像素优先删除。

**这是可复核的启发式基线，不能保证“停放车辆零误删”。** 保护区依赖检测和分割本身正确。强视差、相机与目标同速运动、低纹理、漏检、同组内的轨迹 ID 切换，以及行人与停放车辆靠得很近，都可能导致误判。框位移备用证据也可能受到检测抖动影响。背景单应模型加局部补偿不等于完整的三维运动估计。

两轮车全程停着但有人骑坐时，自动规则倾向保留车辆；人仍删除。应复核后把该车辆轨迹设为 `remove`。推车和骑车也不能仅靠这套几何规则可靠区分。严格区分这些动作可进一步引入姿态／动作识别或人工标注。

尚未获得跟踪 ID 的目标使用一次性的负数 ID：person 仍删除，车辆默认保留，可用逐帧覆盖修正。负数 ID 不代表跨帧的同一个对象。

本版使用 **SAM2 图像预测器逐帧分割，没有 SAM2 视频传播**；某帧 YOLO 完全漏检时，不会凭轨迹自动生成这一帧的 mask。先确保选车规则正确，再考虑用短片段视频传播补漏，避免把错误决定传播到整段。

## 已有预览升级到新版规则

针对旧预览中 229/237 和 403/452 这类漏判，脚本现已修改自动逻辑，而不是在代码里写死这些 ID。
旧记录中 403 混合了 person/motorcycle，452 只有 1 张运动票、0 张骑行票；逐帧记录中有的骑行匹配被旧版固定抽样漏掉。新版增加按类别组跟踪、稀疏光流备用证据和独立人车投票，具体效果尚未运行验证。

**必须重新 analyze，直接 render 旧分析不会重新计算运动或更改旧决定。** 当前旧预览的参数是 `--start 1000 --limit 300`，可使用新目录手动运行：

```bash
python scripts/generate_dynamic_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_preview_v2 \
  --start 1000 --limit 300 --overlay-every 1 --device cuda:0

python scripts/generate_dynamic_masks.py render \
  --analysis outputs/dynamic03_analysis_preview_v2 \
  --output outputs/dynamic03_masks_preview_v3 \
  --dilate 2 --overlay-every 1 --device cuda:0
```

分组跟踪后轨迹 ID 会改变，不要给这次新分析传入旧的 `dynamic03_preview_overrides.json`。
新版 `config.json` 和 `tracks.json` 会记录 `decision_version: 2`。
已编写不依赖模型的回归用例 `tests/test_dynamic_mask_decisions.py`，覆盖类别变化、停放车辆、相机补偿和稀疏投票等规则；遵照要求未执行测试或推理。

## 第一步：分析一段连续帧

下面的命令供你手动运行，本次未执行。使用现有 `maskgen` 环境，需要 `ultralytics`、OpenCV、NumPy、PyTorch，以及已安装的本地官方 SAM2 包。默认复用已有 `models/yolo11x-seg.pt` 和 `models/sam2.1_hiera_base_plus.pt`，不会主动下载模型。

```bash
conda activate maskgen
python scripts/generate_dynamic_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_preview \
  --start 0 --limit 300 \
  --device cuda:0 --overlay-every 5
```

选择一段同时有骑行和停放车辆的连续片段更有代表性，可修改 `--start`。不要先抽成间隔很大的稀疏帧集，光流和关联都依赖连续性。阈值是初始值，尚未在 Dynamic03 上验证。

输出：

- `overlays/`：检测框和轨迹 ID，第一阶段黄色框只表示检测结果，不表示最终删除决定。
- `tracks.json`：每条轨迹的类别、首尾帧、运动票、骑行关联票、默认 `action`、原因和 `review`。`rider_votes_by_person` 列出关联人物及票数；`motion_sources` 是逐帧证据来源计数（不是间隔采样票数）。`actions_by_class` 明确各类别的决定；异常混合类别轨迹可能显示 `action: mixed`，此时按类别决定，person 始终删除。
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

`keep` 保护停放车辆，`remove` 删除已确认的骑行／移动车辆。优先级为：逐帧覆盖 > 轨迹覆盖 > 自动决定。不允许把人物设为 `keep`。轨迹 ID 只对本次分析有效；更换选帧范围、模型或重新分析后，不要直接复用旧 ID。

## 第三步：生成 SAM2 mask

不需要人工覆盖时省略 `--overrides` 这一行：

```bash
python scripts/generate_dynamic_masks.py render \
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
python scripts/generate_dynamic_masks.py analyze \
  --input dataset/Dynamic03_raw \
  --output outputs/dynamic03_analysis_full \
  --device cuda:0 --overlay-every 30

python scripts/generate_dynamic_masks.py render \
  --analysis outputs/dynamic03_analysis_full \
  --output outputs/dynamic03_masks_full \
  --device cuda:0 --overlay-every 30
```

如果全量结果需要人工覆盖，在第二条命令加上针对全量轨迹编写的 `--overrides`。所有输出目录必须为空或不存在；原数据集不会被写入。当前不支持断点续跑。

调参要点：`--motion-px` 单位为原图像素，是跨 `--motion-lag` 帧的残余位移，并非每秒速度。提高 `--motion-px`、`--moving-ratio` 或 `--min-votes` 会减少误删，也会增加漏删；增大 `--motion-lag` 有利于慢速运动判定，但会增大匹配难度。默认 300 帧预览应先用于检查这些取舍。

代码复用了本机 Ultralytics 的 BoT-SORT `update` 接口及检测索引映射；升级 Ultralytics 后需确认这个内部接口仍兼容。官方接口参考：[Ultralytics 跟踪说明](https://docs.ultralytics.com/modes/track/)、[SAM2 图像与视频预测](https://github.com/facebookresearch/sam2)。
