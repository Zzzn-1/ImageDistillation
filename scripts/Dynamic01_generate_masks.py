#!/usr/bin/env python3
"""Per-frame YOLO11 + SAM2 baseline. White pixels mark removal regions."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Dynamic01_raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, default=ROOT / "models/yolo11m-seg.pt")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=ROOT / "models/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dilate", type=int, default=4, help="Dilation radius in pixels")
    parser.add_argument("--start", type=int, default=0, help="Zero-based sorted frame index")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining frames")
    parser.add_argument("--overlay-every", type=int, default=10, help="0 disables previews")
    args = parser.parse_args()
    if min(args.start, args.limit, args.dilate, args.overlay_every) < 0:
        parser.error("start/limit/dilate/overlay-every must be nonnegative")
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("conf must be in (0, 1], imgsz must be positive")
    return args


def main():
    args = parse_args()
    for path in (args.input, args.yolo, args.sam_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    files = sorted((p for p in args.input.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file()),
                   key=natural_key)
    files = files[args.start:args.start + args.limit if args.limit else None]
    if not files:
        raise ValueError("No input frames selected")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate image stems would overwrite output masks")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {args.output}")

    # Keep library configuration and caches inside the workspace.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check GPU access or use --device cpu")
    yolo = YOLO(str(args.yolo))
    predictor = SAM2ImagePredictor(build_sam2(
        args.sam_config, str(args.sam_checkpoint), device=str(device)))
    for name in ("masks", "person_masks", "colmap_masks", "overlays"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch=torch.__version__, method="per-frame YOLO boxes + SAM2",
                  temporal_propagation=False, selected_frames=len(files))
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (args.dilate * 2 + 1,) * 2)

    def save(path, array):
        if not cv2.imwrite(str(path), array):
            raise OSError(f"Failed to write {path}")

    counts = {"frames": 0, "with_detections": 0, "yolo_fallbacks": 0}
    started = time.monotonic()
    with (args.output / "manifest.jsonl").open("w") as manifest, torch.inference_mode():
        for index, path in enumerate(files):
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Unreadable image: {path}")
            h, w = bgr.shape[:2]
            result = yolo.predict(bgr, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                  retina_masks=True, device=str(device), verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            union = np.zeros((h, w), dtype=bool)
            people = []
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" else nullcontext())
            with autocast:
                if len(boxes):
                    predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                for person_idx, (box, confidence) in enumerate(zip(boxes, confidences)):
                    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
                    mask = masks[0].astype(bool)
                    fallback = not mask.any()
                    if fallback:
                        if result.masks is None:
                            raise RuntimeError(f"Both segmenters failed: {path}")
                        mask = cv2.resize(result.masks.data[person_idx].cpu().numpy(),
                                          (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                        counts["yolo_fallbacks"] += 1
                    union |= mask
                    people.append({"box_xyxy": box.tolist(), "yolo_conf": float(confidence),
                                   "sam_score": float(scores[0]), "yolo_fallback": fallback})
            raw = union.astype(np.uint8) * 255
            removal = cv2.dilate(raw, kernel) if args.dilate else raw
            name = path.stem + ".png"
            save(args.output / "person_masks" / name, raw)
            save(args.output / "masks" / name, removal)
            save(args.output / "colmap_masks" / (path.name + ".png"), 255 - removal)
            if args.overlay_every and index % args.overlay_every == 0:
                overlay = bgr.copy()
                selected = removal > 0
                overlay[selected] = (bgr[selected] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                save(args.output / "overlays" / name, overlay)
            record = {"index": args.start + index, "source": str(path.resolve()),
                      "mask": "masks/" + name, "width": w, "height": h,
                      "people": people, "masked_fraction": float((removal > 0).mean())}
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            counts["frames"] += 1
            counts["with_detections"] += int(bool(len(boxes)))
            if index % 10 == 0 or index == len(files) - 1:
                print(f"[{index+1}/{len(files)}] {path.name}: {len(boxes)} people, "
                      f"mask {record['masked_fraction']:.1%}", flush=True)
    counts["seconds"] = round(time.monotonic() - started, 2)
    (args.output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
