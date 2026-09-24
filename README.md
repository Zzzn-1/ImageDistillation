# ImageDistillation

连续图像中的人物分割 → ProPainter 背景修复 → 3DGS 数据准备。

Dynamic03 的人物、骑行车辆和移动车辆 mask 使用独立脚本 `scripts/generate_dynamic_masks.py`，
先分析轨迹和运动，再结合可修改的保留／删除规则生成 SAM2 mask。详见 [Dynamic03 使用说明](docs/dynamic03_masks.md)。
该脚本本次仅编写，未执行或验证推理效果；停放车辆保护需要结合轨迹复核。

## 1. 生成人物 mask

已提供逐帧基线：YOLO11-seg 检测 person，SAM2 使用人物框细化分割。
当前脚本没有视频传播，YOLO 漏检时也可能漏分；应检查预览后再决定是否增加时序跟踪。
仅当 SAM2 返回空 mask 时回退到对应 YOLO mask，不自动修复投影、反射。

在项目根目录运行：

```bash
conda activate maskgen
python scripts/generate_masks.py --output outputs/masks_preview --start 100 --limit 200 --overlay-every 10
```

检查 `outputs/masks_preview/overlays/`，确认手脚、画面边缘和运动模糊区域是否完整覆盖。
全量生成时使用独立输出目录：

```bash
python scripts/generate_masks.py --output outputs/masks_full --overlay-every 30
```

默认使用 `models/yolo11m-seg.pt` 和 `models/sam2.1_hiera_base_plus.pt`，原始尺寸不变。
可用 `--yolo models/yolo11x-seg.pt` 对比大模型，`--device cuda:1` 选择第二张 GPU。
`--conf 0.25` 是检测阈值，`--dilate 4` 是原图尺度的膨胀半径。
输出目录必须为空，防止混入不同参数或不同选帧范围的旧结果。

输出内容：

- `person_masks/`：未膨胀的人物分割，白色为人。
- `masks/`：已膨胀的修复 mask，白色为待补全区域。
- `colmap_masks/`：黑色排除、白色保留，文件名为原图完整文件名加 `.png`。
- `overlays/`：红色覆盖修复区域，绿色框为 YOLO 检测框。
- `manifest.jsonl`：原图路径、原始排序索引、框、分数、mask 占比。
- `config.json` 和 `summary.json`：参数和运行汇总。summary 只在完整运行成功后写入。

无人检测帧会生成全黑修复 mask；这并不保证实际无人，需要抽查。

### 本地验证（2026-09-23）

在 GPU 上完成了索引 100–299 的 200 帧测试，两组输出均检查了尺寸、二值性和 COLMAP 反色一致性：

- `outputs/masks_preview/`：YOLO11m，conf=0.25。
- `outputs/masks_preview_x/`：YOLO11x，conf=0.15。

`outputs/masks_preview_x/comparison.jpg` 左侧为 m、右侧为 x。
x 配合较低阈值补到了第 190 帧的人物，但第 180 帧画面边缘刚进入的局部人体仍漏检。
两组模型和阈值同时变化，此对比不能单独归因于模型大小。
当前结果是逐帧基线，后续应接入时序传播或补充提示后再用于可靠的整段去人。

## 2. 导出逐帧 mask 叠加图

使用原图和每帧 mask 直接生成 PNG 叠加图，不需要重新运行 YOLO/SAM2，也不依赖抽样保存的 `overlays/`。
需要当前环境中的 OpenCV 和 NumPy。

在项目根目录执行：

```bash
conda activate maskgen
python scripts/masks_to_overlay.py
```

默认读取 `Input/Images/` 和 `Input/masks/`，并输出到 `Input/mask_overlays/`。按文件名中的时间戳数字顺序处理所有帧，每帧保存一张红色 mask 叠加图，保留原图的文件名主体，统一输出为 PNG。例如 `1732437358_641172409.png` 对应 `Input/mask_overlays/1732437358_641172409.png`。白色 mask 区域显示为半透明红色，黑色区域保持原图内容；空 mask 对应的叠加图与原图相同。

- `--start 1000 --limit 200`：只导出时间戳排序后的连续 200 帧，序号与分割脚本一致。
- `--alpha 0.45`：红色覆盖透明度。

mask 按原图文件名匹配（扩展名统一为 `.png`），不会静默跳过缺失 mask。
尺寸不一致、非二值 mask 或输出目录已存在时会报错。修改 mask 后，请使用新的输出目录重新导出。
此脚本不改变原图和 mask，仅供人工检查。

### Dynamic03 的 mask 叠加图

如果已生成索引 1000～1299 的预览 mask，在项目根目录执行：

```bash
conda activate maskgen
python scripts/masks_to_overlay.py \
  --input dataset/Dynamic03_raw \
  --masks outputs/dynamic03_masks_preview_v3/masks \
  --output outputs/dynamic03_overlay_v3 \
  --start 1000 --limit 300 \
  --alpha 0.45
```

- `--input`：原图目录。
- `--masks`：实际生成的修复 mask 子目录，例如 `.../masks/`，不是 `analysis` 或反色的 `colmap_masks` 目录。
- `--output`：尚不存在的新目录；即使已有目录为空也会报错。
- `--start 1000 --limit 300`：按原图文件名数字排序，从索引 1000（第 1001 张）开始导出 300 帧。范围应与已有 mask 对应；如果所有原图都有 mask，可以省略这两个参数以导出全量。
- `--alpha 0.45`：红色覆盖透明度，取值大于 0 且不超过 1，越大越明显。

请把示例中的 mask 路径替换成实际输出路径。结果保存到 `outputs/dynamic03_overlay_v3/`，每帧一张 PNG，红色表示待删除区域。该命令只读取已有 mask，不运行 YOLO 或 SAM2。

### 将指定帧的 mask 导入 CVAT 手动修改

在项目根目录运行，默认包含 `1732437232_917264223.png` 至 `1732437234_116649628.png` 的全部帧（含首尾）：

```bash
conda activate maskgen
python scripts/prepare_cvat_masks.py
```

脚本读取 `Input/Images/` 与 `Input/masks/`，输出到 `outputs/cvat_1732437232_917264223_to_1732437234_116649628/`。其中 `images.zip` 用于创建 CVAT 任务，`annotations.zip` 用于任务创建后通过 **Upload annotations → Segmentation Mask 1.1** 导入已有 mask。ZIP 中按时间戳顺序列出帧；原始图片和 mask 不会被修改。CVAT 格式见[官方说明](https://docs.cvat.ai/docs/dataset_management/formats/format-smask/)。

修改帧范围时指定首尾图片文件名，输出目录会随范围自动命名：

```bash
python scripts/prepare_cvat_masks.py \
  --frame-range 1732437261_440158367.png 1732437261_640047550.png
```

也可指定 `--input`、`--masks` 和 `--output-dir`。输出目录必须是新目录；缺失 mask、尺寸不匹配或非二值 mask 会报错。导入的 `person` mask 以现有二值 mask 为基础，同一帧的白色区域没有逐人的实例 ID；要单独区分相邻或相连的人，需要在 CVAT 中手动处理。修改完成后导出 **Segmentation Mask 1.1**，从 `SegmentationClass/` 按导出文件中的 `labelmap.txt` 找到 `person`，转换回原项目需要的黑色 0、白色 255，再检查尺寸和文件名。

将 CVAT 导出的 ZIP 转回逐帧黑白 PNG（从项目根目录执行）：

```bash
python scripts/cvat_export_to_masks.py \
  --input outputs/你的CVAT导出.zip \
  --output outputs/cvat_corrected_masks
```

转换脚本按导出 ZIP 的 `labelmap.txt` 查找 `person`，支持灰度类别编号或 RGB 类别颜色；按 `ImageSets/Segmentation/` 的帧列表输出同名 PNG。没有标注的单帧输出全黑 mask；如果整个 ZIP 都没有 `SegmentationClass/*.png`，则所有列出的帧都输出全黑 mask。导出前请在 CVAT 确认这些帧确实没有需要保留的人物标注。输出目录须是新目录，原始 `Input/masks/` 不会被覆盖。

### 批量清空一段时间戳的 mask

指定首尾 mask 文件名（含首尾），只在新目录输出这段区间的全黑 mask：

```bash
conda activate maskgen
python scripts/blackout_mask_range.py \
  --frame-range 1732437232_917264223.png 1732437234_116649628.png
```

默认从 `Input/masks/` 读取，输出到 `outputs/masks_black_1732437232_917264223_to_1732437234_116649628/`。可用 `--masks` 更换来源、`--output` 指定新的输出目录。只输出选中时间戳的文件，其他 mask 不会复制；原始 mask 不会被覆盖。

## 3. 自动修补少量漏检帧

如果检查叠加图时发现只有少数帧漏掉人物，而它们的前后帧已经检测到了同一个人，可以用相邻帧 mask 通过光流传播来补。这个步骤不会重跑 YOLO/SAM2，默认输出到新目录，方便对比。

例如修补两帧漏检：

```bash
conda activate maskgen
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --output outputs/masks_full_x_patched \
  --frames 1732437261_440158367.png 1732437261_640047550.png \
  --radius 6 --dilate 0 --preview
```

输出目录里的 `masks/`、`person_masks/` 和 `colmap_masks/` 会同步更新；`patch_previews/` 中每张图从左到右是修补前、修补后、新增区域。
脚本保留已有 mask，使用邻帧未膨胀的 `person_masks/` 传播候选区域，逐个排除与目标帧已有 mask 重叠或太近的连通区域，再加入其余候选区域。默认 `--dilate 0`；即使设为正数，也只膨胀新增区域，`person_masks/` 始终保存未额外膨胀的补丁。

`--existing-margin 4` 是已有 mask 周围的候选排除范围，不会写入输出；另外始终保留一圈像素的间隔，防止新增区域连接到已有 mask。如果光流漂移造成附近误补，可以增大该值。此方法没有人物实例 ID，不能保证每个新增区域都是真实漏检人物；两人紧贴、重叠，或同一个人只漏掉手脚时，可能被保守跳过。请检查预览中黄色的新增区域。如果补得过宽，优先把 `--radius 6` 降到 `3`。

如果只想输出指定的修补帧，而不是复制完整 mask 目录，添加 `--only-selected`：

```bash
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --output outputs/mask_patch_only \
  --frames 1732437261_440158367.png 1732437261_640047550.png \
  --radius 6 --dilate 0 --preview --only-selected
```

这种模式不会改动 `--mask-root` 里的已有 mask，只会在 `--output` 目录写出被选中的补丁帧。`--output` 不能和 `--mask-root` 相同，且必须是空目录或新目录。

### 已有人物的臀部、手脚等局部缺失

默认模式会跳过与已有 mask 相连的区域，因此不能修补已检出人物的局部轮廓。可添加 `--fill-partial`，例如（将帧名替换为实际缺失的图片）：

```bash
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --output outputs/mask_partial_v1 \
  --frames 1732437261_440158367.png 1732437261_640047550.png \
  --radius 3 --dilate 0 --fill-partial --preview --only-selected
```

局部补全先从与已有 mask 重叠的邻帧传播区域累计像素投票，再补入与已有 mask 相连、且获得至少 `--partial-votes 2` 帧及 `--partial-ratio 0.6` 比例支持的像素。比例的分母是通过最小白色像素数检查的邻帧数量。此模式允许改变已有轮廓，但不对局部补丁做形态学膨胀；原有白色像素始终保留，原始文件不覆盖。`--existing-margin` 仅限制独立人物补丁，不限制局部补全。

邻帧也漏掉同一部位时无法靠此方法恢复；光流的共同偏移仍可能误补，务必检查黄色新增区域。`partial_added_pixels` 显示局部补全的像素数。`--mask-root` 应指向含相邻帧的完整 mask 目录，不能直接使用只含几张补丁的目录。

如果 YOLO 本身分割到了臀部，但 SAM2 漏掉了，可用 `--yolo-repair` 在指定区域补入 YOLO 的人物分割。下面针对 640×480 图片中右后方黑衣人的例子，处理首尾之间全部 26 帧：

```bash
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --output outputs/mask_hip_repair_v3 \
  --frame-range 1732437352_369941473.png 1732437353_202486515.png \
  --radius 6 --dilate 0 --preview --only-selected \
  --yolo-repair models/yolo11x-seg.pt \
  --repair-roi 430 0 620 195 --repair-conf 0.1 --repair-crop --device cpu
```

ROI 格式是 `左 上 右 下`，单位为原图像素。仅采用检测框中心在 ROI 内的人物，并将其新增分割裁剪到 ROI 内，不膨胀原有 mask；ROI 不是整块涂白。这只限制 YOLO 补全，原来的邻帧独立人物补丁逻辑仍会运行。人物移出该区域时应调整 ROI。此模式会重跑 YOLO，需要 `ultralytics` 和本地分割权重，可指定 `--device cuda:0` 使用 GPU。`--frame-range` 包含首尾帧，与 `--frames` 二选一。

`--repair-crop` 额外将 ROI 裁剪后送入 YOLO，再映射回原图，与整图检测的局部补丁合并；未启用时只对整图推理再筛选 ROI。上述例子的部分帧需要裁剪推理才能检出黑衣人，也有帧需要整图检测补充。裁剪区域内其他人物也可能被补全，因此范围应尽量只包括待修补的人。

### 镜头移动时补检黑外套人物，并保留已有补丁

下面处理 `1732437349_639115334.png` 至 `1732437352_536442518.png`（含首尾，共 88 帧）：

```bash
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --base-patch outputs/mask_patch_only outputs/mask_hip_repair_v3 \
  --output outputs/mask_blackcoat_repair_v1 \
  --frame-range 1732437349_639115334.png 1732437352_536442518.png \
  --radius 6 --dilate 0 --preview --only-selected \
  --yolo-repair models/yolo11x-seg.pt \
  --repair-roi-file configs/blackcoat_repair_rois.json \
  --repair-conf 0.1 --repair-crop --repair-contained --device cpu
```

`--base-patch` 将选中帧已有补丁与原始 mask 取并集，保留其白色区域；邻帧传播仍读取完整的 `--mask-root`，不会覆盖来源文件。目前须与 `--only-selected` 一起使用，输出只包含本次选中帧。

ROI 配置根据抽样图像设置关键帧搜索框，并按帧序插值，用于跟随这段镜头中的目标位置；它不是人物 mask，也不是自动人物跟踪。`--repair-contained` 要求检测框位于搜索区域内并距边界至少 2 像素，过滤被裁剪边界截断的前景人物。黑衣人本身碰到区域边界也会被略过，因此换数据时应调整配置。人物完全被遮挡时无需补入不可见的人体；仅露出很小部分时检测仍可能遗漏，请检查预览。

其中 `1732437352_270035505.png` 和 `1732437352_336639881.png` 在 v1 中仍漏检，可用专门的裁剪配置再次修补：

```bash
python scripts/patch_mask_gaps.py \
  --mask-root outputs/masks_full_x \
  --base-patch outputs/mask_patch_only outputs/mask_hip_repair_v3 outputs/mask_blackcoat_repair_v1 \
  --output outputs/mask_blackcoat_fix2_v2 \
  --frames 1732437352_270035505.png 1732437352_336639881.png \
  --radius 6 --dilate 0 --preview --only-selected \
  --yolo-repair models/yolo11x-seg.pt \
  --repair-roi-file configs/blackcoat_fix2_rois.json \
  --repair-conf 0.1 --repair-crop --repair-contained --device cpu
```

该命令仅输出两帧。已生成的 `outputs/mask_blackcoat_repair_v2/` 是将这两帧补丁合入 v1 后的完整 88 帧结果，其余 86 帧沿用 v1；来源记录在 `merge_config.json`，慢速预览是 `review_slow.mp4`。复跑上述命令须改用新的空输出目录。

修补后重新导出叠加图时，注意 `--masks` 指向新目录：

```bash
python scripts/masks_to_overlay.py --masks outputs/masks_full_x_patched/masks --output outputs/masks_full_x_patched/review_overlays
```

## 4. 运行 ProPainter

### Dynamic01：使用 CVAT 修正后的 mask

已提供 `scripts/inpaint_dynamic01.py`，默认读取：

- 原图：`/home/lzn/code/3DGS/ImageDistillation/Input/Images/`
- mask：`/home/lzn/code/3DGS/ImageDistillation/Input/masks_cvat/`
- 输出：`outputs/dynamic01_propainter_cvat/images/原图文件名主体.png`

本次只编写并静态检查脚本，没有执行输入校验、测试或模型推理。脚本复用 `run_propainter.py` 的分段、文件名恢复和断点续跑功能；现有 `--input Input` 用法仍默认读取 `Input/masks/`，而新入口明确使用 `masks_cvat/`。

在项目根目录手动执行，全量修复：

```bash
conda activate propainter
cd /home/lzn/code/3DGS/ImageDistillation
python scripts/inpaint_dynamic01.py --gpu 0
```

也可以先选一段连续帧检查效果，例如索引 1000～1159：

```bash
python scripts/inpaint_dynamic01.py \
  --start 1000 --limit 160 \
  --output outputs/dynamic01_propainter_cvat_preview \
  --gpu 0
```

**前后帧补全方式：** 本地 ProPainter 先估计双向光流、补全被遮挡区域的光流并传播可见背景，再结合前后邻帧和片段内的参考帧生成修复结果。默认 `--neighbor-length 10` 对应每个推理中心前后各最多 5 帧的局部邻域；`--ref-stride 10` 从当前片段每隔 10 帧选取额外参考帧。不是逐张图片独立修复。

为控制显存，默认每段 80 帧、相邻段重叠 20 帧，在重叠区中间交接输出。新入口会检查保留帧在分段交接处有足够的前后邻帧；所选序列的首尾只能使用实际存在的帧。使用 `--start/--limit` 时，参考范围也仅限所选序列，不会额外读取范围外图像。远处参考帧仅来自当前片段，默认并不读取整套数据集作为参考。

mask 必须与原图同尺寸、同文件名主体，保存为二值 PNG：白色 255 表示删除／修复，黑色 0 表示保留。默认推理不扩边；可通过 `--inference-mask-dilate` 扩大 ProPainter 内部使用的 mask，但最终始终只将原始 mask 的白色区域替换为修复结果，黑色区域像素保持原图内容。原图和 `masks_cvat` 不会被修改。所选图像至少两帧、尺寸一致，宽高需要是 8 的倍数；缺失 mask 或格式不符会报错。

若背景在片段的前后帧都被遮挡，模型仍需推测内容，不能保证恢复真实背景。请检查物体边缘、残影、细纹理和片段交接处的闪烁；增大 `--chunk-size` 可扩大参考范围，但会增加显存占用。

常用选项：

- `--output 新目录`：自定义输出，已有目录只允许在参数和输入一致时配合 `--resume` 使用。
- `--resume`：中断后使用原命令加此参数，复用已完成片段。例如全量运行可执行 `python scripts/inpaint_dynamic01.py --gpu 0 --resume`。
- `--dry-run`：只校验输入、权重文件和分段计划，不运行模型、不创建输出；本次未执行。
- `--gpu 1`：选择另一张 GPU；默认使用 FP16，可加 `--fp32`。
- `--chunk-size 40 --overlap 10 --neighbor-length 6`：显存不足时用新的输出目录尝试较短片段。
- `--images 路径 --masks 路径`：更换原图和 mask 目录，新入口应使用这两个参数覆盖默认来源。
- `--inference-mask-dilate 4`：只在 ProPainter 内部对光流 mask 和修复 mask 进行 4 次二值膨胀，减少边界附近的前景污染；最终输出的替换范围仍为原始 CVAT mask。默认 0，改变此值后需要新输出目录。
- `--propagation-length 180`：覆盖官方图像传播阶段的内部 100 帧上限，数值不能超过 `--chunk-size`。不指定时保留官方逻辑。加大后可能增加显存需求。
- `--debug-frames 原图文件名.png ...`：保存指定帧的实际推理 mask、传播结果、剩余未知区域、最终模型结果、光流数组和对照图；所列文件必须位于本次选帧范围内。

需使用已安装 ProPainter 依赖的环境，本地源码位于 `third_party/ProPainter/`，权重为其 `weights/` 下的 `raft-things.pth`、`recurrent_flow_completion.pth`、`ProPainter.pth`。脚本会先检查本地权重文件，再启动推理。

输出除 `images/` 外，还保留 `run_config.json`（实际输入路径、参数、帧映射及分段计划）、`.done/` 完成标记，以及全部完成后的 `summary.json`。改变 CVAT mask、选帧或修复参数后需要使用新输出目录。

### mask 已覆盖，但修复后仍出现腿部残影

已查看 `dynamic01_propainter_cvat_preview` 中的 `1732437264_284211397.png`、对应原图和 CVAT mask。视觉上主要腿部已被白色 mask 覆盖，输出却在该区域生成裤腿状残影。这提示应检查模型修复、光流传播和邻帧信息，不能仅凭输出残影断定当前帧漏标。

运行记录显示这帧是原序列索引 1046，来自第一段 1000～1079；尽管预览共选了 160 帧，后面的 1080～1159 没有参与这帧的修复。检查片段末帧时，人仍挡在镜头前。原参数还将 ProPainter 的光流及修复 mask 膨胀都设为了 0。边界污染和参考范围不足是待验证的可能因素，并非已确认的唯一原因。

可先保持原来的分段参数，只测试推理 mask 扩边，以便与已有结果比较：

```bash
python scripts/inpaint_dynamic01.py \
  --start 1000 --limit 160 \
  --chunk-size 80 --overlap 20 \
  --neighbor-length 10 --ref-stride 10 \
  --inference-mask-dilate 4 \
  --output outputs/dynamic01_propainter_cvat_preview_dilate4 \
  --gpu 0
```

如仍有大片残影，再将这 160 帧作为同一段处理，让目标帧能够使用后续参考帧：

```bash
python scripts/inpaint_dynamic01.py \
  --start 1000 --limit 160 \
  --chunk-size 160 --overlap 40 \
  --neighbor-length 10 --ref-stride 10 \
  --inference-mask-dilate 4 \
  --output outputs/dynamic01_propainter_cvat_preview_context160 \
  --gpu 0
```

第二条命令的显存需求高于原来的 80 帧片段；这里恰好只有一个 160 帧片段，`--overlap 40` 不会产生额外片段。两次尝试都保留原始 CVAT mask 与原图，不能对旧输出使用 `--resume`。如果目标背景在参考帧中不可见、视角差异太大或传播失败，单纯扩边和增加帧数也可能无法解决，需要进一步检查参考帧及其 mask。原始 mask 外的像素始终保持原图，因此推理扩边不会修复原始 mask 外的漏标。

这些对比命令尚未执行，不能据此保证残影已解决。

### 已扩大到 180 帧，仍有连续裤腿残影

最新检查的 `dynamic01_propainter_cvat_preview/run_config.json` 已是 `start=1000, limit=180, chunk_size=180, mask_dilation=4`。`1732437264_084539890.png` 仍有大片裤腿状纹理，此时不能再仅归因于未选到后面的帧。

本地官方 `inference_propainter.py` 另有一层限制：`subvideo_length_img_prop = min(100, args.subvideo_length)`。它把图像传播分成最多 100 帧的小段，并在各小段前后各附加最多 10 帧。例如索引 1040 在这次运行的局部索引为 40，原图像传播只能使用局部 0～109，也就是原序列 1000～1109。1160 帧虽然参与后续 transformer 参考帧补全，却不参与目标帧所在小段的图像传播。这是已确认的实现限制，但还不能断定是残影的唯一原因。

新增 `scripts/propainter_runtime.py` 适配器，通过检查并调整本地官方入口的指定语法节点，在运行时开放传播长度和插入诊断导出。它不修改 `third_party` 源码；官方入口结构不兼容时会在模型加载前报错。不开启新参数时仍走原入口。开启时在运行配置中记录适配器及官方入口的 SHA-256，代码变化后不混用旧的断点结果。

手动运行下列命令可让图像传播也覆盖整段，同时保留三张问题帧的诊断信息：

```bash
python scripts/inpaint_dynamic01.py \
  --start 1000 --limit 180 \
  --chunk-size 180 --overlap 40 \
  --neighbor-length 10 --ref-stride 10 \
  --inference-mask-dilate 4 --propagation-length 180 \
  --debug-frames 1732437264_084539890.png 1732437264_184378386.png 1732437264_284211397.png \
  --output outputs/dynamic01_propainter_propagation180_debug \
  --gpu 0
```

查看 `debug/clip_00000/1732437264_084539890/comparison.png`，两行依次为：原图、实际推理 mask 叠加、图像传播结果；剩余未知区域、传播已接受的区域、transformer 输出。白色 `remaining_mask.png` 表示传播后仍未补全；`propagated.png` 中这些位置的中性灰色并不代表已恢复的背景。`model_output.png` 是最终按原始 CVAT mask 合成之前的模型输出，实际交付结果仍在 `images/`。

- 如果裤腿纹理已出现在 `propagated.png`，并且该处在 `propagated_valid.png` 中为白色，优先检查光流补全、对齐及邻帧 mask 是否引入了错误背景。
- 如果该处仍在 `remaining_mask.png` 中为白色，或传播结果正常、最终模型输出才出现纹理，则重点检查剩余大洞及模型参考帧补全；不能把未知区域的灰色当成传播污染。
- `flows.npz` 包含所选帧前后相邻帧对的原始／补全光流，方向字段表示该帧对的正向／反向，数组为 `2×H×W`，单位为像素。`metadata.json` 记录源帧索引、传播局部范围（左闭右开）和 mask 内仍未补全的比例。

如需只诊断原先的 100 帧传播逻辑，省略 `--propagation-length 180` 并选择另一个新输出目录，保留其余参数。导出不会改变推理张量或 CVAT mask。当前只编写并静态检查了适配器及回归用例，没有执行测试或推理，也没有验证残影是否消失。

### 原有 Input/masks 工作流与历史验证

输入使用 `Input/Images/` 与 `Input/masks/`，相同时间戳文件名一一对应，mask 白色为需要修复的区域。已验证当前 5251 对输入尺寸为 640×480，mask 均为二值图。三个推理权重应放在 `third_party/ProPainter/weights/`：`raft-things.pth`、`recurrent_flow_completion.pth`、`ProPainter.pth`。

使用 `scripts/run_propainter.py` 分段调用本地官方脚本。官方脚本会先把整段图像及 mask 放入显存，单独设置 `--subvideo_length 80` 不能让 5251 帧输入变成流式读取。

先试跑 80 帧（索引 3640–3719，包含近期修补的位置）：

```bash
conda activate propainter
cd /home/lzn/code/3DGS/ImageDistillation
python scripts/run_propainter.py \
  --input Input --output outputs/propainter_preview \
  --start 3640 --limit 80 --gpu 0
```

结果为 `outputs/propainter_preview/images/原时间戳.png`。本脚本保留原图尺寸及时间戳文件名，仅用修复结果替换白色 mask 区域，黑色区域像素保持输入原图内容。默认 FP16、每段 80 帧、重叠 20 帧，官方参数为 `--neighbor_length 10 --ref_stride 10 --mask_dilation 0`，不再次膨胀 mask。请检查残留人物、修复纹理及帧间闪烁。

确认试跑效果后处理全量：

```bash
python scripts/run_propainter.py \
  --input Input --output outputs/propainter_full --gpu 0
```

当前输入会分成 88 段处理，最终在 `outputs/propainter_full/images/` 保存 5251 张 PNG。重叠区域在中间交接，每帧只取一个片段的结果；分段之间仍可能存在闪烁，需要检查交接位置。临时输入及分段结果在该段完成后清理，保留最终 PNG、`run_config.json`、分段完成标记及成功结束后的 `summary.json`。

中断后使用相同参数加 `--resume`：

```bash
python scripts/run_propainter.py \
  --input Input --output outputs/propainter_full --gpu 0 --resume
```

脚本跳过已完成且输出文件仍存在的片段，重跑未完成片段。输入文件名、尺寸/格式检查、文件大小和修改时间参与校验；输入或处理参数改变时需使用新输出目录。不同 GPU 可继续同一任务。

- `--dry-run`：校验输入及分段计划，不推理、不创建输出目录。
- `--gpu 1`：改用另一张 GPU。
- 如果显存不足，可用新输出目录加 `--chunk-size 40 --overlap 10 --neighbor-length 6`；较短片段的背景参考范围也会缩小。

本地已完成全部输入校验及分段、文件名恢复、背景像素保留和断点续跑的模拟测试；实际 GPU 推理效果需运行后检查。相机位姿估计仍优先使用原图及 COLMAP 排除 mask；修复区域应保留标记，后续视质量考虑降低 3DGS 监督权重。
