#!/usr/bin/env python3
"""Run local ProPainter on overlapping clips, restoring original image filenames."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from patch_mask_gaps import ROOT, scan_images


def plan_chunks(count, length, overlap):
    if count < 2 or length < 2 or not 0 <= overlap < length:
        raise ValueError('Need at least 2 frames and 0 <= overlap < chunk size')
    windows = []
    start = 0
    while True:
        end = min(start + length, count)
        windows.append((start, end))
        if end == count:
            break
        start = end - overlap
    # Avoid passing a single final frame to the optical-flow model.
    if windows[-1][1] - windows[-1][0] == 1:
        windows[-1] = (count - 2, count)
    boundaries = [0] + [(windows[i - 1][1] + windows[i][0]) // 2
                         for i in range(1, len(windows))] + [count]
    return [{'start': start, 'end': end, 'keep_start': boundaries[i],
             'keep_end': boundaries[i + 1]} for i, (start, end) in enumerate(windows)]


def validate_temporal_context(chunks, count, radius):
    """Each retained clip boundary must have neighbors from the selected sequence."""
    for chunk in chunks:
        before = chunk['keep_start'] - chunk['start']
        after = chunk['end'] - chunk['keep_end']
        required_before = min(radius, chunk['keep_start'])
        required_after = min(radius, count - chunk['keep_end'])
        if before < required_before or after < required_after:
            raise ValueError('Insufficient temporal context at a clip boundary; increase --overlap '
                             'to at least --neighbor-length (and keep --chunk-size > --overlap)')


def read_inputs(root, start, limit, *, images=None, masks=None):
    import cv2
    import numpy as np

    images = images if images is not None else root / 'Images'
    masks = masks if masks is not None else root / 'masks'
    for directory in (images, masks):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    files = scan_images(images)
    files = files[start:start + limit if limit else None]
    if len(files) < 2:
        raise ValueError('Select at least two input frames')
    if limit and len(files) != limit:
        raise ValueError(f'Requested {limit} frames; only {len(files)} are available')
    records = []
    shape = None
    fingerprint = hashlib.sha256()
    for index, image_path in enumerate(files, start):
        mask_path = masks / (image_path.stem + '.png')
        if not mask_path.is_file():
            raise FileNotFoundError(f'Missing matching mask: {mask_path}')
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f'Unreadable image or matching mask: {image_path.name}')
        if shape is None:
            shape = image.shape[:2]
        if image.shape[:2] != shape or mask.shape != shape:
            raise ValueError(f'Image/mask dimensions do not match: {image_path.name}')
        if not np.isin(mask, [0, 255]).all():
            raise ValueError(f'Non-binary mask: {mask_path}')
        for path in (image_path, mask_path):
            info = path.stat()
            fingerprint.update(f'{path}:{info.st_size}:{info.st_mtime_ns}\n'.encode())
        records.append({'image': str(image_path), 'mask': str(mask_path),
                        'output': image_path.stem + '.png', 'source_index': index})
    return records, shape, fingerprint.hexdigest()


def composite(original, predicted, mask):
    """Only change white-mask pixels; preserve known background exactly."""
    if original.shape != predicted.shape or mask.shape != original.shape[:2]:
        raise ValueError('Inpainted frame dimensions do not match input')
    result = original.copy()
    result[mask == 255] = predicted[mask == 255]
    return result


def main(argv=None, *, default_images=None, default_masks=None, default_output=None,
         require_temporal_context=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'Input',
                        help='Root containing Images/ and masks/ when explicit directories are absent')
    parser.add_argument('--images', type=Path, default=default_images,
                        help='Original image directory; overrides --input/Images')
    parser.add_argument('--masks', type=Path, default=default_masks,
                        help='Binary PNG mask directory (white=remove); overrides --input/masks')
    parser.add_argument('--output', type=Path, default=default_output, required=default_output is None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0, help='0: all remaining frames')
    parser.add_argument('--chunk-size', type=int, default=80)
    parser.add_argument('--overlap', type=int, default=20)
    parser.add_argument('--neighbor-length', type=int, default=10)
    parser.add_argument('--ref-stride', type=int, default=10)
    parser.add_argument('--inference-mask-dilate', type=int, default=0,
                        help='ProPainter mask dilation iterations for flow completion and inpainting only; '
                             'final compositing still uses the original mask (default: 0)')
    parser.add_argument('--propagation-length', type=int,
                        help='Override the official 100-frame image-propagation cap; bounded by chunk-size')
    parser.add_argument('--debug-frames', nargs='+', default=[], metavar='FILENAME',
                        help='Export propagation, remaining masks, flows and comparisons for these source filenames')
    parser.add_argument('--gpu', default='0', help='Single GPU index or UUID')
    parser.add_argument('--fp32', action='store_true', help='Disable default FP16 inference')
    parser.add_argument('--resume', action='store_true', help='Resume with identical inputs/settings')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and plan; no inference or writes')
    args = parser.parse_args(argv)
    if args.start < 0 or args.limit < 0 or args.neighbor_length < 2 or args.ref_stride < 1:
        parser.error('Invalid start/limit/neighbor-length/ref-stride')
    if args.chunk_size < 2 or not 0 <= args.overlap < args.chunk_size:
        parser.error('Require chunk-size >= 2 and 0 <= overlap < chunk-size')
    if args.inference_mask_dilate < 0:
        parser.error('inference-mask-dilate must be nonnegative')
    if args.propagation_length is not None and not 2 <= args.propagation_length <= args.chunk_size:
        parser.error('Require 2 <= propagation-length <= chunk-size')
    if not args.gpu or ',' in args.gpu:
        parser.error('Select one GPU per run')
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    custom_inputs = args.images is not None or args.masks is not None
    images = (args.images if args.images is not None else args.input / 'Images').resolve()
    masks = (args.masks if args.masks is not None else args.input / 'masks').resolve()
    repo = ROOT / 'third_party/ProPainter'
    if not (repo / 'inference_propainter.py').is_file():
        raise FileNotFoundError(repo / 'inference_propainter.py')
    for name in ('raft-things.pth', 'recurrent_flow_completion.pth', 'ProPainter.pth'):
        weight = repo / 'weights' / name
        if not weight.is_file() or weight.stat().st_size == 0:
            raise FileNotFoundError(f'Missing weight: {weight}')
    print('Validating input images and masks...', flush=True)
    records, shape, fingerprint = read_inputs(args.input, args.start, args.limit, images=images, masks=masks)
    selected_names = {Path(record['image']).name for record in records}
    debug_names = set(args.debug_frames)
    if debug_names - selected_names:
        raise ValueError(f'Debug frames not found in selected inputs: {sorted(debug_names - selected_names)}')
    use_adapter = args.propagation_length is not None or bool(debug_names)
    adapter = ROOT / 'scripts/propainter_runtime.py'
    if use_adapter and not adapter.is_file():
        raise FileNotFoundError(adapter)
    chunks = plan_chunks(len(records), args.chunk_size, args.overlap)
    if require_temporal_context:
        validate_temporal_context(chunks, len(records), args.neighbor_length // 2)
    height, width = shape
    if height % 8 or width % 8:
        raise ValueError('This wrapper requires image width/height divisible by 8')
    print(f'{len(records)} frames, {width}x{height}, {len(chunks)} clips; '
          f'chunk={args.chunk_size}, overlap={args.overlap}', flush=True)
    print(f'Images: {images}\nMasks: {masks}', flush=True)
    print(f'Inference mask dilation: {args.inference_mask_dilate}; '
          'final replacement region: original mask', flush=True)
    print(f'Image-propagation block: up to {min(args.propagation_length or 100, args.chunk_size)} frames; '
          f'diagnostic frames: {len(debug_names)}', flush=True)
    if args.dry_run:
        print('Validation passed. No inference performed or output written.')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import cv2

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this environment; check nvidia-smi and PyTorch CUDA support')
    config = {'version': 1, 'input': str(args.input), 'fingerprint': fingerprint,
              'start': args.start, 'limit': args.limit, 'width': width, 'height': height,
              'chunk_size': args.chunk_size, 'overlap': args.overlap,
              'neighbor_length': args.neighbor_length, 'ref_stride': args.ref_stride,
              'fp16': not args.fp32, 'mask_dilation': args.inference_mask_dilate,
              'chunks': chunks, 'frames': records}
    # Preserve the legacy config for unchanged --input runs and their resume files.
    if custom_inputs or require_temporal_context:
        config.update(version=2, images=str(images), masks=str(masks),
                      temporal_context_radius=args.neighbor_length // 2 if require_temporal_context else None)
    if use_adapter:
        config.update(version=3, propagation_length=args.propagation_length, debug_frames=sorted(debug_names),
                      runtime_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                      inference_sha256=hashlib.sha256((repo / 'inference_propainter.py').read_bytes()).hexdigest())
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Output exists; use a new directory or --resume')
        saved = json.loads((args.output / 'run_config.json').read_text())
        if config != saved:
            raise ValueError('Inputs/settings changed; use a new output directory')
    else:
        args.output.mkdir(parents=True)
        (args.output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    images_dir = args.output / 'images'
    done_dir = args.output / '.done'
    images_dir.mkdir(exist_ok=True)
    done_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    cache = ROOT / '.cache/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    env.setdefault('MPLCONFIGDIR', str(cache))
    print(f'Using GPU: {torch.cuda.get_device_name(0)}', flush=True)
    for chunk_index, chunk in enumerate(chunks):
        marker = done_dir / f'{chunk_index:05d}.json'
        kept = records[chunk['keep_start']:chunk['keep_end']]
        debug_directory = args.output / 'debug' / f'clip_{chunk_index:05d}'
        debug_frames = {str(i): {'name': r['output'], 'source_index': r['source_index']}
                        for i, r in enumerate(records[chunk['start']:chunk['end']])
                        if Path(r['image']).name in debug_names}
        debug_complete = all((debug_directory / Path(r['name']).stem / 'comparison.png').is_file()
                             for r in debug_frames.values())
        if marker.exists() and all((images_dir / r['output']).is_file() for r in kept) and debug_complete:
            print(f'[{chunk_index + 1}/{len(chunks)}] already completed', flush=True)
            continue
        print(f'[{chunk_index + 1}/{len(chunks)}] input frames '
              f'{args.start + chunk["start"]}..{args.start + chunk["end"] - 1}', flush=True)
        with tempfile.TemporaryDirectory(prefix='.clip_', dir=args.output) as temp:
            work = Path(temp)
            for subdir in ('frames', 'masks'):
                (work / subdir).mkdir()
            for local_index, record in enumerate(records[chunk['start']:chunk['end']]):
                source = Path(record['image'])
                (work / 'frames' / f'{local_index:04d}{source.suffix}').symlink_to(source)
                (work / 'masks' / f'{local_index:04d}.png').symlink_to(record['mask'])
            command = [sys.executable, str(repo / 'inference_propainter.py'),
                       '--video', str(work / 'frames'), '--mask', str(work / 'masks'),
                       '--output', str(work / 'results'), '--width', str(width), '--height', str(height),
                       '--neighbor_length', str(args.neighbor_length), '--ref_stride', str(args.ref_stride),
                       '--subvideo_length', str(args.chunk_size),
                       '--mask_dilation', str(args.inference_mask_dilate),
                       '--save_fps', '30', '--save_frames']
            if not args.fp32:
                command.append('--fp16')
            if use_adapter:
                entry = [sys.executable, str(adapter), '--repo', str(repo)]
                if args.propagation_length is not None:
                    entry += ['--propagation-length', str(args.propagation_length)]
                if debug_frames:
                    spec = work / 'debug_spec.json'
                    spec.write_text(json.dumps({'output': str(debug_directory), 'frames': debug_frames}) + '\n')
                    entry += ['--debug-spec', str(spec)]
                command = entry + ['--'] + command[2:]
            subprocess.run(command, cwd=repo, env=env, check=True)
            for index in range(chunk['keep_start'], chunk['keep_end']):
                record = records[index]
                local_index = index - chunk['start']
                predicted_path = work / 'results/frames/frames' / f'{local_index:04d}.png'
                predicted = cv2.imread(str(predicted_path))
                if predicted is None:
                    raise RuntimeError(f'Missing ProPainter output: {predicted_path}')
                original = cv2.imread(record['image'])
                # Always reload the ORIGINAL CVAT mask. ProPainter's temporary
                # dilation improves its context but must not expand saved edits.
                mask = cv2.imread(record['mask'], cv2.IMREAD_GRAYSCALE)
                result = composite(original, predicted, mask)
                target = images_dir / record['output']
                temporary = target.with_name(target.stem + '.partial.png')
                if not cv2.imwrite(str(temporary), result):
                    raise OSError(f'Failed to write {temporary}')
                temporary.replace(target)
        marker.write_text(json.dumps(chunk) + '\n')
    summary = {'frames': len(records), 'clips': len(chunks), 'output': str(images_dir)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Done: {images_dir}', flush=True)


if __name__ == '__main__':
    main()
