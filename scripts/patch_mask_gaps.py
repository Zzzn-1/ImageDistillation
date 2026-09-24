#!/usr/bin/env python3
"""Add missing person regions from nearby frames while protecting existing masks."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Dynamic01_raw")
    parser.add_argument("--mask-root", type=Path, default=ROOT / "outputs/masks_full_x")
    parser.add_argument("--base-patch", type=Path, nargs="+", default=[],
                        help="Keep existing selected-frame patches from these directories; "
                             "neighbor masks still come from --mask-root")
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--frames", nargs="+",
                        help="Frame filenames or stems to patch, for example 173...png")
    selection.add_argument("--frame-range", nargs=2, metavar=("FIRST", "LAST"),
                           help="Patch all frames between these filenames, inclusive")
    parser.add_argument("--radius", type=int, default=6,
                        help="Use source masks within this many neighboring frames")
    parser.add_argument("--dilate", type=int, default=0,
                        help="Dilate newly added regions only (default: 0)")
    parser.add_argument("--existing-margin", type=int, default=4,
                        help="Reject candidates within this many pixels of existing masks; "
                             "this protection margin is not written to the output")
    parser.add_argument("--min-source-pixels", type=int, default=100,
                        help="Skip source masks with fewer white pixels")
    parser.add_argument("--fill-partial", action="store_true",
                        help="Also fill incomplete existing people using multi-frame agreement")
    parser.add_argument("--partial-votes", type=int, default=2,
                        help="Minimum supporting source frames for partial filling (default: 2)")
    parser.add_argument("--partial-ratio", type=float, default=0.6,
                        help="Required fraction of usable source frames for partial filling (default: 0.6)")
    parser.add_argument("--yolo-repair", type=Path,
                        help="Optional YOLO segmentation checkpoint for direct partial repair")
    roi_selection = parser.add_mutually_exclusive_group()
    roi_selection.add_argument("--repair-roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Limit YOLO repair to this region and detections centered inside it")
    roi_selection.add_argument("--repair-roi-file", type=Path,
                               help="JSON mapping keyframe filenames to ROI boxes; interpolate by frame index")
    parser.add_argument("--repair-conf", type=float, default=0.1,
                        help="YOLO confidence for local repair (default: 0.1)")
    parser.add_argument("--repair-crop", action="store_true",
                        help="Also run YOLO on the ROI crop and combine with full-image repair")
    parser.add_argument("--repair-contained", action="store_true",
                        help="Require repair detection boxes to fit inside the ROI with a "
                             "2-pixel margin, excluding people cut by ROI edges")
    parser.add_argument("--device", default="cpu", help="Device for optional YOLO repair")
    parser.add_argument("--preview", action="store_true",
                        help="Write before/after preview images for patched frames")
    parser.add_argument("--only-selected", action="store_true",
                        help="Only write patched files for --frames instead of copying all masks")
    args = parser.parse_args()
    if args.radius <= 0:
        parser.error("radius must be positive")
    if args.dilate < 0 or args.min_source_pixels < 0 or args.existing_margin < 0:
        parser.error("dilate, min-source-pixels and existing-margin must be nonnegative")
    if args.partial_votes < 2 or not 0 < args.partial_ratio <= 1:
        parser.error("partial-votes must be at least 2 and partial-ratio must be in (0, 1]")
    if bool(args.yolo_repair) != bool(args.repair_roi or args.repair_roi_file):
        parser.error("--yolo-repair requires --repair-roi or --repair-roi-file, and vice versa")
    if args.repair_crop and not args.yolo_repair:
        parser.error("--repair-crop requires --yolo-repair and --repair-roi")
    if args.repair_contained and not args.yolo_repair:
        parser.error("--repair-contained requires --yolo-repair and --repair-roi")
    if args.base_patch and not args.only_selected:
        parser.error("--base-patch currently requires --only-selected")
    if not 0 < args.repair_conf <= 1:
        parser.error("repair-conf must be in (0, 1]")
    if args.repair_roi:
        x1, y1, x2, y2 = args.repair_roi
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            parser.error("repair-roi must satisfy 0 <= X1 < X2 and 0 <= Y1 < Y2")
    return args


def scan_images(directory):
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = sorted((p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=natural_key)
    if not files:
        raise ValueError(f"No input images found in {directory}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate image stems would make mask matching ambiguous")
    return files


def normalize_target(name):
    path = Path(name)
    return path.stem if path.suffix else name


def copy_tree_contents(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.suffix.lower() == ".mp4" or path.name == "patch_previews":
            continue
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def read_gray(path, cv2):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable grayscale image: {path}")
    return image


def read_bgr(path, cv2):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def source_mask_path(mask_root, subdir, stem, image_name=None):
    if subdir == "colmap_masks":
        if image_name is None:
            raise ValueError("image_name is required for colmap_masks")
        return mask_root / subdir / f"{image_name}.png"
    return mask_root / subdir / f"{stem}.png"


def warp_mask(source_frame, target_frame, source_mask, cv2, np):
    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        target_gray, source_gray, None,
        pyr_scale=0.5, levels=4, winsize=35, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0)
    h, w = source_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # remap needs source coordinates for each target pixel (backward flow).
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(source_mask, map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def select_missing_regions(warped, protected, kernel, cv2, np):
    """Reject whole components belonging to existing people, not just overlap pixels.

    Masks have no instance IDs: touching people can form one component and be
    skipped together. This is deliberately conservative about existing masks.
    """
    count, labels = cv2.connectedComponents((warped == 255).astype(np.uint8),
                                            connectivity=8)
    blocked = set(np.unique(labels[protected != 0]).tolist())
    raw_added = np.zeros_like(warped)
    added = np.zeros_like(warped)
    for label in range(1, count):
        if label in blocked:
            continue
        component = (labels == label).astype(np.uint8) * 255
        expanded = cv2.dilate(component, kernel) if kernel is not None else component
        if np.any((expanded != 0) & (protected != 0)):
            continue
        raw_added |= component
        added |= expanded
    return raw_added, added


def overlapping_regions(warped, existing, cv2, np):
    """Candidates for an incomplete person must overlap an existing person."""
    _, labels = cv2.connectedComponents((warped == 255).astype(np.uint8),
                                        connectivity=8)
    matched = np.unique(labels[existing != 0])
    matched = matched[matched != 0]
    return np.isin(labels, matched)


def select_partial_regions(votes, source_count, existing, min_votes, ratio, cv2, np):
    """Add supported pixels connected to an existing mask, without dilation."""
    required = max(min_votes, math.ceil(source_count * ratio))
    supported = (votes >= required) & (existing == 0)
    _, labels = cv2.connectedComponents(supported.astype(np.uint8), connectivity=8)
    boundary = cv2.dilate((existing != 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    touching = np.unique(labels[boundary != 0])
    touching = touching[touching != 0]
    return np.isin(labels, touching).astype(np.uint8) * 255


def select_roi_masks(masks, boxes, roi, shape, np, contained=False):
    """Select instances centered in an ROI and clip additions to that ROI."""
    x1, y1, x2, y2 = roi
    selected = np.zeros(shape, dtype=np.uint8)
    for mask, (bx1, by1, bx2, by2) in zip(masks, boxes):
        if mask.shape != shape:
            raise ValueError("YOLO mask dimensions do not match target image")
        if contained and not (x1 + 2 <= bx1 and y1 + 2 <= by1
                              and bx2 <= x2 - 2 and by2 <= y2 - 2):
            continue
        if x1 <= (bx1 + bx2) / 2 < x2 and y1 <= (by1 + by2) / 2 < y2:
            selected[y1:y2, x1:x2] |= (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
    return selected


def load_patch_base(mask_root, patch_roots, stem, cv2, np):
    """Union prior patches without ever using them as temporal source masks."""
    before = read_gray(source_mask_path(mask_root, "masks", stem), cv2)
    raw_before = read_gray(source_mask_path(mask_root, "person_masks", stem), cv2)
    for root in patch_roots:
        removal_path = source_mask_path(root, "masks", stem)
        person_path = source_mask_path(root, "person_masks", stem)
        if not removal_path.exists() and not person_path.exists():
            continue
        for path, base in ((removal_path, before), (person_path, raw_before)):
            patch = read_gray(path, cv2)
            if patch.shape != base.shape or not np.isin(patch, [0, 255]).all():
                raise ValueError(f"Invalid patch dimensions or non-binary mask: {path}")
            base |= patch
    return before, raw_before


def load_repair_rois(path, files, target_stems, np):
    """Interpolate user-specified search boxes, never interpolate mask pixels."""
    keyframes = json.loads(path.read_text())
    if not isinstance(keyframes, dict) or not keyframes:
        raise ValueError("ROI file must map keyframe filenames to [x1, y1, x2, y2]")
    by_stem = {p.stem: i for i, p in enumerate(files)}
    anchors = []
    for name, box in keyframes.items():
        stem = normalize_target(name)
        if stem not in by_stem or not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Invalid ROI keyframe: {name}")
        if not all(type(v) is int for v in box) or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
            raise ValueError(f"Invalid ROI coordinates: {name}")
        anchors.append((by_stem[stem], box))
    anchors.sort()
    indices = [index for index, _ in anchors]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate ROI keyframe stems")
    rois = {}
    for stem in target_stems:
        index = by_stem[stem]
        if not indices[0] <= index <= indices[-1]:
            raise ValueError(f"ROI keyframes do not cover selected frame: {stem}")
        rois[stem] = [int(round(np.interp(index, indices, [box[c] for _, box in anchors])))
                      for c in range(4)]
    return rois


def render_preview(frame, before, after, cv2, np):
    before_overlay = frame.copy()
    after_overlay = frame.copy()
    added_overlay = frame.copy()
    before_sel = before == 255
    after_sel = after == 255
    added_sel = after_sel & ~before_sel
    before_overlay[before_sel] = (
        before_overlay[before_sel] * 0.55 + np.array([0, 0, 255]) * 0.45
    ).astype(np.uint8)
    after_overlay[after_sel] = (
        after_overlay[after_sel] * 0.55 + np.array([0, 0, 255]) * 0.45
    ).astype(np.uint8)
    added_overlay[added_sel] = (
        added_overlay[added_sel] * 0.45 + np.array([0, 255, 255]) * 0.55
    ).astype(np.uint8)
    for label, image in (("before", before_overlay),
                         ("after", after_overlay),
                         ("added", added_overlay)):
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([before_overlay, after_overlay, added_overlay])


def main():
    args = parse_args()
    import cv2
    import numpy as np

    if args.output.resolve() == args.mask_root.resolve():
        raise ValueError("--output must be different from --mask-root to avoid overwriting existing masks")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty or new output directory: {args.output}")
    if not args.mask_root.is_dir():
        raise NotADirectoryError(args.mask_root)
    for subdir in ("masks", "person_masks", "colmap_masks"):
        if not (args.mask_root / subdir).is_dir():
            raise NotADirectoryError(args.mask_root / subdir)
    for root in args.base_patch:
        if args.output.resolve() == root.resolve():
            raise ValueError("--output must differ from every --base-patch directory")
        for subdir in ("masks", "person_masks"):
            if not (root / subdir).is_dir():
                raise NotADirectoryError(root / subdir)

    files = scan_images(args.input)
    by_stem = {p.stem: i for i, p in enumerate(files)}
    if args.frame_range:
        endpoints = [normalize_target(name) for name in args.frame_range]
        if any(stem not in by_stem for stem in endpoints):
            raise FileNotFoundError(f"Frame range endpoint not found: {args.frame_range}")
        first, last = [by_stem[stem] for stem in endpoints]
        if first > last:
            raise ValueError("Frame range must be in chronological order")
        target_stems = [p.stem for p in files[first:last + 1]]
    else:
        target_stems = [normalize_target(name) for name in args.frames]
    missing_targets = [stem for stem in target_stems if stem not in by_stem]
    if missing_targets:
        raise FileNotFoundError(f"Target frame not found: {missing_targets[0]}")
    repair_rois = (load_repair_rois(args.repair_roi_file, files, target_stems, np)
                   if args.repair_roi_file else {stem: args.repair_roi for stem in target_stems})

    repair_model = None
    if args.yolo_repair:
        if not args.yolo_repair.is_file():
            raise FileNotFoundError(args.yolo_repair)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
        import torch
        from ultralytics import YOLO
        if args.device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        repair_model = YOLO(str(args.yolo_repair))

    if args.only_selected:
        args.output.mkdir(parents=True, exist_ok=True)
        for subdir in ("masks", "person_masks", "colmap_masks"):
            (args.output / subdir).mkdir(parents=True, exist_ok=True)
    else:
        copy_tree_contents(args.mask_root, args.output)
    preview_dir = args.output / "patch_previews"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    kernel = None
    if args.dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (args.dilate * 2 + 1,) * 2)
    records = []
    for stem in target_stems:
        target_index = by_stem[stem]
        target_path = files[target_index]
        target_frame = read_bgr(target_path, cv2)
        before, raw_before = load_patch_base(args.mask_root, args.base_patch, stem, cv2, np)
        if before.shape != target_frame.shape[:2] or raw_before.shape != before.shape:
            raise ValueError(f"Mask dimensions do not match target frame: {stem}")

        protected = np.maximum(before, raw_before)
        # Reserve a small gap even with margin=0 so additions cannot attach to
        # an existing component. This mask is used only for candidate rejection.
        protected = cv2.dilate(protected, np.ones((3, 3), dtype=np.uint8))
        if args.existing_margin:
            protection_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.existing_margin * 2 + 1,) * 2)
            protected = cv2.dilate(protected, protection_kernel)
        raw_propagated = np.zeros_like(before)
        propagated = np.zeros_like(before)
        existing = np.maximum(before, raw_before)
        partial_votes = np.zeros(before.shape, dtype=np.uint32) if args.fill_partial else None
        source_count = 0
        for source_index in range(max(0, target_index - args.radius),
                                  min(len(files), target_index + args.radius + 1)):
            if source_index == target_index:
                continue
            source_path = files[source_index]
            source_mask = read_gray(source_mask_path(args.mask_root, "person_masks", source_path.stem), cv2)
            if int((source_mask == 255).sum()) < args.min_source_pixels:
                continue
            source_frame = read_bgr(source_path, cv2)
            if source_mask.shape != before.shape or source_frame.shape != target_frame.shape:
                raise ValueError(f"Source dimensions do not match target frame: {source_path.name}")
            warped = warp_mask(source_frame, target_frame, source_mask, cv2, np)
            if args.fill_partial:
                partial_votes += overlapping_regions(warped, existing, cv2, np)
            # Filter each neighbor separately: unioning first can merge a
            # missing person's region with an already detected person's region.
            raw_new, new = select_missing_regions(warped, protected, kernel, cv2, np)
            raw_propagated |= raw_new
            propagated |= new
            source_count += 1

        partial = np.zeros_like(before)
        if args.fill_partial:
            partial = select_partial_regions(partial_votes, source_count, existing,
                                             args.partial_votes, args.partial_ratio, cv2, np)
            # Do not dilate partial repairs: each added pixel needs frame support.
            raw_propagated |= partial
            propagated |= partial
        repair = np.zeros_like(before)
        if repair_model is not None:
            target_roi = repair_rois[stem]
            if target_roi[2] > before.shape[1] or target_roi[3] > before.shape[0]:
                raise ValueError(f"repair-roi exceeds target dimensions: {stem}")
            x1, y1, x2, y2 = target_roi
            # Full-image and cropped detection can miss different frames, so
            # crop mode supplements rather than replaces full-image evidence.
            for use_crop in ([False, True] if args.repair_crop else [False]):
                repair_input = target_frame[y1:y2, x1:x2] if use_crop else target_frame
                result = repair_model.predict(repair_input, classes=[0], conf=args.repair_conf,
                                              imgsz=640, retina_masks=True, device=args.device,
                                              verbose=False)[0]
                if result.masks is None:
                    continue
                repair_roi = [0, 0, x2 - x1, y2 - y1] if use_crop else target_roi
                local = select_roi_masks(result.masks.data.cpu().numpy(),
                                         result.boxes.xyxy.cpu().numpy(),
                                         repair_roi, repair_input.shape[:2], np,
                                         contained=args.repair_contained)
                if use_crop:
                    repair[y1:y2, x1:x2] |= local
                else:
                    repair |= local
            # Direct segmentation evidence, never a dilation of the old mask.
            raw_propagated |= repair
            propagated |= repair
        after = np.maximum(before, propagated).astype(np.uint8)
        added = int(((after == 255) & (before != 255)).sum())

        if not cv2.imwrite(str(args.output / "masks" / f"{stem}.png"), after):
            raise OSError(f"Failed to write patched mask for {stem}")
        raw_after = np.maximum(raw_before, raw_propagated).astype(np.uint8)
        if not cv2.imwrite(str(args.output / "person_masks" / f"{stem}.png"), raw_after):
            raise OSError(f"Failed to write patched person mask for {stem}")
        colmap_name = target_path.name + ".png"
        if not cv2.imwrite(str(args.output / "colmap_masks" / colmap_name), 255 - after):
            raise OSError(f"Failed to write patched COLMAP mask for {stem}")
        if args.preview:
            preview = render_preview(target_frame, before, after, cv2, np)
            preview_path = preview_dir / f"{stem}.jpg"
            if not cv2.imwrite(str(preview_path), preview):
                raise OSError(f"Failed to write preview: {preview_path}")

        record = {
            "frame": target_path.name,
            "index": target_index,
            "source_frames_used": source_count,
            "before_pixels": int((before == 255).sum()),
            "after_pixels": int((after == 255).sum()),
            "added_pixels": added,
            "partial_added_pixels": int((partial == 255).sum()),
            "yolo_repair_added_pixels": int(((repair == 255) & (before != 255)).sum()),
            "repair_roi": repair_rois[stem],
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    patch_config = {
        "input": str(args.input),
        "mask_root": str(args.mask_root),
        "base_patch": [str(root) for root in args.base_patch],
        "output": str(args.output),
        "frames": target_stems,
        "radius": args.radius,
        "dilate": args.dilate,
        "existing_margin": args.existing_margin,
        "min_source_pixels": args.min_source_pixels,
        "fill_partial": args.fill_partial,
        "partial_votes": args.partial_votes,
        "partial_ratio": args.partial_ratio,
        "yolo_repair": str(args.yolo_repair) if args.yolo_repair else None,
        "repair_roi": args.repair_roi,
        "repair_roi_file": str(args.repair_roi_file) if args.repair_roi_file else None,
        "repair_conf": args.repair_conf,
        "repair_crop": args.repair_crop,
        "repair_contained": args.repair_contained,
        "device": args.device,
        "only_selected": args.only_selected,
        "method": "backward optical-flow warp of neighboring person masks; add only "
                  "components separated from existing masks; dilate new regions only; "
                  "optional partial filling by multi-frame agreement or ROI-limited "
                  "YOLO segmentation without dilation",
        "records": records,
    }
    (args.output / "patch_config.json").write_text(
        json.dumps(patch_config, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved patched masks to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
