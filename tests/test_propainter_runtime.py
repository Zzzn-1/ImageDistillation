"""Adapter regressions on synthetic source; no model imports or inference."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from propainter_runtime import instrument_source


SOURCE = '''
subvideo_length_img_prop = min(100, args.subvideo_length)
ori_frames = frames_inp
comp_frames = ["generated"]
if args.save_frames:
    saved = True
'''


class RuntimeTests(unittest.TestCase):
    def execute_fixture(self, length, chunk=180):
        events = []
        diagnostics = SimpleNamespace(
            propagated=lambda *values: events.append(('propagated', values[-1])),
            finished=lambda frames: events.append(('finished', frames)))
        namespace = {name: None for name in ('frames_inp', 'flow_masks', 'masks_dilated',
                                             'updated_frames', 'updated_masks', 'gt_flows_bi', 'pred_flows_bi')}
        namespace.update(args=SimpleNamespace(subvideo_length=chunk, save_frames=True), _diagnostics=diagnostics)
        exec(instrument_source(SOURCE, '<synthetic>', length), namespace)
        return namespace, events

    def test_default_diagnostics_do_not_change_propagation_length(self):
        namespace, events = self.execute_fixture(None)
        self.assertEqual(namespace['subvideo_length_img_prop'], 100)
        self.assertEqual(events, [('propagated', 100), ('finished', ['generated'])])
        self.assertTrue(namespace['saved'])

    def test_requested_length_and_chunk_bound(self):
        namespace, events = self.execute_fixture(180)
        self.assertEqual(namespace['subvideo_length_img_prop'], 180)
        self.assertEqual(events[0], ('propagated', 180))
        namespace, _ = self.execute_fixture(180, chunk=80)
        self.assertEqual(namespace['subvideo_length_img_prop'], 80)

    def test_changed_upstream_expression_is_rejected(self):
        with self.assertRaises(ValueError):
            instrument_source(SOURCE.replace('min(100, args.subvideo_length)', '200'), '<synthetic>', 180)

    def test_missing_or_duplicate_hooks_are_rejected(self):
        with self.assertRaises(ValueError):
            instrument_source(SOURCE.replace('ori_frames = frames_inp', 'other = frames_inp'), '<synthetic>', None)
        with self.assertRaises(ValueError):
            instrument_source(SOURCE + '\nori_frames = frames_inp\n', '<synthetic>', None)


if __name__ == '__main__':
    unittest.main()
