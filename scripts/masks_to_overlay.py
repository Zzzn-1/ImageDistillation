#!/usr/bin/env python3
"""Save one red mask overlay PNG per image, in timestamp order."""

import argparse
import math
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Input/Images")
    parser.add_argument("--masks", type=Path, default=ROOT / "Input/masks")
    parser.add_argument("--output", type=Path, default=ROOT / "Input/mask_overlays")
    parser.add_argument("--alpha", type=float, default=0.45, help="Red overlay opacity")
    parser.add_argument("--start", type=int, default=0, help="Zero-based index in timestamp order")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining frames")
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("alpha must be in (0, 1]")
    if args.start < 0 or args.limit < 0:
        parser.error("start and limit must be nonnegative")
    return args


def main():
    args = parse_args()
    import cv2
    import numpy as np

    for directory in (args.input, args.masks):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")

    files = sorted((p for p in args.input.iterdir() if p.is_file()
                    and p.suffix.lower() in {".png", ".jpg", ".jpeg"}), key=natural_key)
    files = files[args.start:args.start + args.limit if args.limit else None]
    if not files:
        raise ValueError("No images selected")
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate image stems: output and mask filenames would be ambiguous")
    mask_paths = [args.masks / f"{p.stem}.png" for p in files]
    missing = [p for p in mask_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks; first: {missing[0]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mask_overlays_", dir=args.output.parent) as temp_dir:
        temp_output = Path(temp_dir)
        for index, (path, mask_path) in enumerate(zip(files, mask_paths)):
            frame = cv2.imread(str(path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                raise ValueError(f"Unreadable image or mask: {path}, {mask_path}")
            if frame.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions do not match: {path}, {mask_path}")
            if np.any((mask != 0) & (mask != 255)):
                raise ValueError(f"Mask must contain only 0 and 255: {mask_path}")

            selected = mask == 255
            overlay = frame.copy()
            overlay[selected] = np.rint(
                frame[selected] * (1 - args.alpha) + np.array([0, 0, 255]) * args.alpha
            ).astype(np.uint8)
            target = temp_output / f"{path.stem}.png"
            if not cv2.imwrite(str(target), overlay):
                raise OSError(f"Failed to write overlay: {target}")
            if index % 250 == 0 or index == len(files) - 1:
                print(f"[{index+1}/{len(files)}] {path.name}", flush=True)
        temp_output.rename(args.output)

    print(f"Saved {len(files)} overlays to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
