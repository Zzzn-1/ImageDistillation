#!/usr/bin/env python3
"""Write only the masks in an inclusive timestamp range as black PNGs."""

import argparse
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, default=ROOT / "Input/masks",
                        help="Source directory of timestamp-named PNG masks")
    parser.add_argument("--frame-range", nargs=2, required=True, metavar=("FIRST", "LAST"),
                        help="First and last mask filenames or stems, inclusive")
    parser.add_argument("--output", type=Path,
                        help="New directory containing only selected masks; "
                             "defaults to outputs/masks_black_<first>_to_<last>")
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2
    import numpy as np

    if not args.masks.is_dir():
        raise NotADirectoryError(args.masks)
    files = sorted((path for path in args.masks.iterdir()
                    if path.is_file() and path.suffix.lower() == ".png"), key=natural_key)
    if not files:
        raise ValueError(f"No PNG masks found in {args.masks}")
    by_stem = {path.stem: index for index, path in enumerate(files)}
    if len(by_stem) != len(files):
        raise ValueError("Duplicate mask stems")
    first, last = (Path(name).stem for name in args.frame_range)
    if first not in by_stem or last not in by_stem:
        raise FileNotFoundError(f"Range endpoints must exist in {args.masks}: {first}, {last}")
    start, stop = by_stem[first], by_stem[last]
    if start > stop:
        raise ValueError("Frame range must be in timestamp order")

    output = args.output or ROOT / "outputs" / f"masks_black_{first}_to_{last}"
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".black_masks_", dir=output.parent) as temp_dir:
        temporary = Path(temp_dir)
        for path in files[start:stop + 1]:
            target = temporary / path.name
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
                raise ValueError(f"Expected 8-bit grayscale PNG mask: {path}")
            if not cv2.imwrite(str(target), np.zeros_like(mask)):
                raise OSError(f"Failed to write black mask: {target}")
        temporary.rename(output)

    print(f"Saved {stop - start + 1} black masks to {output.resolve()} "
          f"({files[start].name} to {files[stop].name})")


if __name__ == "__main__":
    main()
