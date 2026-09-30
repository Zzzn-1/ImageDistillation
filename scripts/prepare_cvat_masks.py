#!/usr/bin/env python3
"""Package a timestamp range of images and binary masks for CVAT editing."""

import argparse
from pathlib import Path
import re
import tempfile
from zipfile import ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANGE = (
    "1733458896_237117052.png",
    "1733458932_514494419.png",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LABELMAP = "background:0,0,0::\nperson:128,0,0::\n"


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def select_images(directory, first, last):
    images = sorted((path for path in directory.iterdir()
                     if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                    key=natural_key)
    by_stem = {path.stem: index for index, path in enumerate(images)}
    if len(by_stem) != len(images):
        raise ValueError("Image stems must be unique so masks can match by timestamp")
    first_stem, last_stem = Path(first).stem, Path(last).stem
    if first_stem not in by_stem or last_stem not in by_stem:
        raise FileNotFoundError(f"Frame range endpoints must exist in {directory}: {first}, {last}")
    start, stop = by_stem[first_stem], by_stem[last_stem]
    if start > stop:
        raise ValueError("Frame range must be in timestamp order")
    return images[start:stop + 1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "PropainterInput/Dynamic03/Images",
                        help="Original images directory")
    parser.add_argument("--masks", type=Path, default=ROOT / "outputs/dynamic03_masks_refined/masks",
                        help="Binary masks directory (0=background, 255=person)")
    parser.add_argument("--frame-range", nargs=2, metavar=("FIRST", "LAST"),
                        default=DEFAULT_RANGE,
                        help="Inclusive first and last image filenames; defaults to the requested range")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for images.zip and annotations.zip; "
                             "defaults to outputs/cvat_<first>_to_<last>")
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2
    import numpy as np

    for directory in (args.input, args.masks):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    images = select_images(args.input, *args.frame_range)
    mask_paths = [args.masks / f"{image.stem}.png" for image in images]
    missing = [path for path in mask_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} masks; first: {missing[0]}")

    output_dir = args.output_dir or (ROOT / "outputs" /
        f"cvat_{images[0].stem}_to_{images[-1].stem}")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cvat_", dir=output_dir.parent) as temp_dir:
        temporary = Path(temp_dir)
        with (ZipFile(temporary / "images.zip", "w", compression=ZIP_STORED) as image_zip,
              ZipFile(temporary / "annotations.zip", "w", compression=ZIP_STORED) as annotation_zip):
            annotation_zip.writestr("labelmap.txt", LABELMAP)
            annotation_zip.writestr("ImageSets/Segmentation/default.txt",
                                    "".join(f"{image.stem}\n" for image in images))
            for index, (image_path, mask_path) in enumerate(zip(images, mask_paths), start=1):
                image = cv2.imread(str(image_path))
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    raise ValueError(f"Unreadable image or mask: {image_path}, {mask_path}")
                if image.shape[:2] != mask.shape:
                    raise ValueError(f"Image/mask dimensions do not match: {image_path}, {mask_path}")
                if np.any((mask != 0) & (mask != 255)):
                    raise ValueError(f"Mask must contain only 0 and 255: {mask_path}")

                # CVAT's grayscale labelmap maps index 0 to background and 1 to person.
                indexed = (mask == 255).astype(np.uint8)
                success, encoded = cv2.imencode(".png", indexed)
                if not success:
                    raise OSError(f"Failed to encode mask: {mask_path}")
                image_zip.write(image_path, arcname=image_path.name)
                # Binary source masks have no per-person IDs; object index 1
                # represents the entire foreground in this frame.
                for directory in ("SegmentationClass", "SegmentationObject"):
                    annotation_zip.writestr(f"{directory}/{image_path.stem}.png",
                                            encoded.tobytes())
                if index % 100 == 0 or index == len(images):
                    print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        temporary.rename(output_dir)

    print(f"Saved {len(images)} frames to {output_dir.resolve()}", flush=True)
    print("Create the CVAT task with images.zip, then upload annotations.zip "
          "as Segmentation Mask 1.1 annotations.", flush=True)


if __name__ == "__main__":
    main()
