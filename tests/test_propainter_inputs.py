"""Regression cases for CVAT mask selection and temporal clip boundaries."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_propainter import composite, plan_chunks, read_inputs, validate_temporal_context


class ProPainterInputTests(unittest.TestCase):
    def test_explicit_cvat_masks_override_legacy_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('Images', 'masks', 'masks_cvat'):
                (root / name).mkdir()
            for name in ('10.png', '2.png'):
                cv2.imwrite(str(root / 'Images' / name), np.zeros((8, 8, 3), np.uint8))
                cv2.imwrite(str(root / 'masks' / name), np.zeros((8, 8), np.uint8))
                cv2.imwrite(str(root / 'masks_cvat' / name), np.full((8, 8), 255, np.uint8))
            records, shape, _ = read_inputs(root, 0, 0, images=root / 'Images', masks=root / 'masks_cvat')
            self.assertEqual([r['output'] for r in records], ['2.png', '10.png'])
            self.assertEqual(shape, (8, 8))
            self.assertTrue(all(Path(r['mask']).parent == root / 'masks_cvat' for r in records))
            (root / 'masks_cvat' / '2.png').unlink()
            with self.assertRaises(FileNotFoundError):
                read_inputs(root, 0, 0, masks=root / 'masks_cvat')

    def test_overlap_retains_context_and_exactly_one_output_per_frame(self):
        for count in (2, 80, 81, 160, 161, 5251):
            chunks = plan_chunks(count, length=80, overlap=20)
            validate_temporal_context(chunks, count, radius=5)
            emitted = [i for chunk in chunks for i in range(chunk['keep_start'], chunk['keep_end'])]
            self.assertEqual(emitted, list(range(count)))

    def test_insufficient_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_temporal_context(plan_chunks(160, length=80, overlap=0), 160, radius=5)

    def test_composite_preserves_unmasked_background(self):
        original = np.full((8, 8, 3), 17, np.uint8)
        prediction = np.full_like(original, 201)
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 2:6] = 255
        result = composite(original, prediction, mask)
        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(result[mask == 255], prediction[mask == 255])
        self.assertTrue(np.all(original == 17))


if __name__ == '__main__':
    unittest.main()
