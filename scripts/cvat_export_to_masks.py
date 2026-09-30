#!/usr/bin/env python3
"""Convert CVAT Segmentation Mask 1.1 class masks to binary person PNGs."""

import argparse
from pathlib import Path
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/mask111.zip",
                        help="CVAT Segmentation Mask 1.1 export ZIP")
    parser.add_argument("--images", type=Path, default=ROOT / "PropainterInput/Dynamic03/Images",
                        help="Original images, used to validate sizes and fill empty frames")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cvat_corrected_masks",
                        help="New directory for 0/255 grayscale PNG masks")
    parser.add_argument("--label", default="person", help="Class to retain as white")
    return parser.parse_args()


def archive_layout(archive):
    names = archive.namelist()
    labelmaps = [name for name in names if name == "labelmap.txt" or name.endswith("/labelmap.txt")]
    if len(labelmaps) != 1:
        raise ValueError("Expected exactly one labelmap.txt in the CVAT ZIP")
    prefix = labelmaps[0][:-len("labelmap.txt")]
    subsets = [name for name in names if name.startswith(prefix + "ImageSets/Segmentation/")
               and name.endswith(".txt")]
    if not subsets:
        raise ValueError("Missing ImageSets/Segmentation/*.txt in the CVAT ZIP")
    stems = []
    for name in sorted(subsets):
        stems.extend(line.strip() for line in archive.read(name).decode("utf-8").splitlines()
                     if line.strip())
    if not stems or len(set(stems)) != len(stems):
        raise ValueError("Frame list is empty or contains duplicate names")
    class_prefix = prefix + "SegmentationClass/"
    class_masks = {Path(name).stem: name for name in names
                   if name.startswith(class_prefix) and name.lower().endswith(".png")}
    if len(class_masks) != sum(name.startswith(class_prefix) and name.lower().endswith(".png")
                              for name in names):
        raise ValueError("Duplicate class mask filenames in the CVAT ZIP")
    unexpected = set(class_masks) - set(stems)
    if unexpected:
        raise ValueError(f"Class mask is not in the frame list: {sorted(unexpected)[0]}")
    return labelmaps[0], stems, class_masks


def parse_labelmap(contents, label):
    entries = []
    for line in contents.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(":")
        if len(fields) < 2:
            raise ValueError(f"Invalid labelmap line: {line}")
        name = fields[0].strip()
        try:
            color = tuple(int(value) for value in fields[1].split(","))
        except ValueError as exc:
            raise ValueError(f"Invalid label color: {line}") from exc
        if len(color) != 3 or any(not 0 <= value <= 255 for value in color):
            raise ValueError(f"Invalid RGB label color: {line}")
        entries.append((name, color))
    matches = [index for index, (name, _) in enumerate(entries) if name == label]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label!r} label in labelmap.txt")
    return entries, matches[0]


def person_pixels(encoded, entries, person_index, cv2, np):
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Unreadable class mask PNG in CVAT ZIP")
    if image.ndim == 2:
        values = np.unique(image)
        if np.any(values >= len(entries)):
            raise ValueError(f"Unknown grayscale class indices: {values.tolist()}")
        return image == person_index
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported class mask shape: {image.shape}")

    bgr = image[:, :, :3]
    known = np.zeros(image.shape[:2], dtype=bool)
    for _, rgb in entries:
        known |= np.all(bgr == rgb[::-1], axis=2)
    if not known.all():
        y, x = np.argwhere(~known)[0]
        raise ValueError(f"Unknown RGB class color: {tuple(int(v) for v in bgr[y, x, ::-1])}")
    return np.all(bgr == entries[person_index][1][::-1], axis=2)


def main():
    args = parse_args()
    import cv2
    import numpy as np

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")

    image_files = [path for path in args.images.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    images = {path.stem: path for path in image_files}
    if len(images) != len(image_files):
        raise ValueError("Original image stems must be unique")
    with ZipFile(args.input) as archive:
        labelmap_name, stems, class_masks = archive_layout(archive)
        entries, person_index = parse_labelmap(archive.read(labelmap_name), args.label)
        missing = [stem for stem in stems if stem not in images]
        if missing:
            raise FileNotFoundError(f"Original image not found: {missing[0]}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".cvat_masks_", dir=args.output.parent) as temp_dir:
            temporary = Path(temp_dir)
            for stem in stems:
                original = cv2.imread(str(images[stem]))
                if original is None:
                    raise ValueError(f"Unreadable original image: {images[stem]}")
                if stem in class_masks:
                    selected = person_pixels(archive.read(class_masks[stem]), entries,
                                             person_index, cv2, np)
                    if selected.shape != original.shape[:2]:
                        raise ValueError(f"Class mask size does not match original image: {stem}")
                    mask = selected.astype(np.uint8) * 255
                else:
                    # CVAT omits PNGs for frames with no annotations.
                    mask = np.zeros(original.shape[:2], dtype=np.uint8)
                target = temporary / f"{stem}.png"
                if not cv2.imwrite(str(target), mask):
                    raise OSError(f"Failed to write: {target}")
            temporary.rename(args.output)

    print(f"Saved {len(stems)} binary masks to {args.output.resolve()} "
          f"({len(class_masks)} annotated, {len(stems) - len(class_masks)} empty)")


if __name__ == "__main__":
    main()
