#!/usr/bin/env python3
"""Inpaint Dynamic01 using CVAT masks and temporally overlapping ProPainter clips.

Defaults:
  images: Input/Images
  masks:  Input/masks_cvat (0=keep, 255=remove)
  output: outputs/dynamic01_propainter_cvat

The local ProPainter implementation uses bidirectional optical flow, neighboring
frames before/after each inference center, and reference frames within each clip.
Only white-mask pixels are replaced. All run_propainter CLI options are accepted;
use --images and --masks to override this entry point's explicit source defaults.
--inference-mask-dilate expands ProPainter's working masks without modifying CVAT
masks or expanding the final replacement region. Its default remains zero.
--propagation-length overrides the separate internal image-propagation cap.
--debug-frames saves intermediate evidence for selected original filenames.
"""

from run_propainter import ROOT, main


if __name__ == '__main__':
    main(default_images=ROOT / 'Input/Images',
         default_masks=ROOT / 'Input/masks_cvat',
         default_output=ROOT / 'outputs/dynamic01_propainter_cvat',
         require_temporal_context=True)
