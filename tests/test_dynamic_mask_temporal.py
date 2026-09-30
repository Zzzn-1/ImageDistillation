"""Temporal ID-switch regressions using synthetic masks; no YOLO/SAM2 required."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dynamic_mask_temporal import TemporalRemoval, warp_masks


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((80, 120, 3), dtype=np.uint8)
        self.bike = np.zeros((80, 120), dtype=bool)
        self.bike[25:60, 10:45] = True
        self.parked = np.zeros_like(self.bike)
        self.parked[25:60, 65:100] = True
        self.temporal = TemporalRemoval(hold=5)
        # Geometry/state tests use zero displacement. The flow itself is tested below.
        self.warper = patch("dynamic_mask_temporal.warp_masks",
                            side_effect=lambda old, new, masks: [m.copy() for m in masks])
        self.warper.start()
        self.addCleanup(self.warper.stop)

    def frame(self, index, items, keep=()):
        objects = [dict(id=tid, **spec) for tid, action, mask, spec in items]
        decisions = {str(tid): {"action": action} for tid, action, mask, spec in items}
        seeds = [mask for tid, action, mask, spec in items]
        self.temporal.update(self.image, objects, seeds, decisions, set(map(str, keep)),
                             index, f"{index}.png")
        return decisions

    def seed_removal(self):
        for i in range(2):
            self.frame(i, [(55, "remove", self.bike, {"class": 1})])

    def test_id_and_bicycle_motorcycle_class_switch_continue_removal(self):
        self.seed_removal()
        result = self.frame(2, [(-17, "keep", self.bike, {"class": 3}),
                                (9, "keep", self.parked, {"class": 3})])
        self.assertEqual(result["-17"]["action"], "remove")
        self.assertEqual(result["-17"]["temporal"]["source_id"], "55")
        self.assertEqual(result["9"]["action"], "keep")
        result = self.frame(3, [(81, "keep", self.bike, {"class": 1}),
                                (9, "keep", self.parked, {"class": 3})])
        self.assertEqual(result["81"]["action"], "remove")
        self.assertEqual(result["9"]["action"], "keep")

    def test_single_remove_frame_does_not_seed_propagation(self):
        self.frame(0, [(55, "remove", self.bike, {"class": 1})])
        result = self.frame(1, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "keep")

    def test_inherited_decisions_expire_without_renewing_themselves(self):
        self.seed_removal()
        for i in range(2, 7):
            result = self.frame(i, [(81, "keep", self.bike, {"class": 1})])
            self.assertEqual(result["81"]["action"], "remove")
        result = self.frame(7, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "keep")

    def test_fresh_rider_comotion_can_renew_a_confirmed_history(self):
        self.seed_removal()
        for i in range(2, 7):
            result = self.frame(i, [(81, "keep", self.bike,
                                    {"class": 1, "rider_matches": [46] if i == 6 else []})])
            self.assertEqual(result["81"]["action"], "remove")
        self.assertTrue(result["81"]["temporal"]["renewed_by_rider_match"])
        result = self.frame(7, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "remove")

    def test_explicit_keep_blocks_and_clears_history(self):
        self.seed_removal()
        result = self.frame(2, [(81, "keep", self.bike, {"class": 1})], keep=[81])
        self.assertEqual(result["81"]["action"], "keep")
        result = self.frame(3, [(90, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["90"]["action"], "keep")

    def test_ambiguous_neighboring_detections_do_not_inherit_removal(self):
        self.seed_removal()
        result = self.frame(2, [(81, "keep", self.bike, {"class": 1}),
                                (9, "keep", self.bike, {"class": 3})])
        self.assertEqual([r["action"] for r in result.values()], ["keep", "keep"])
        result = self.frame(3, [(9, "keep", self.bike, {"class": 3})])
        self.assertEqual(result["9"]["action"], "keep")

    def test_same_id_at_another_location_does_not_inherit(self):
        self.seed_removal()
        result = self.frame(2, [(55, "keep", self.parked, {"class": 1})])
        self.assertEqual(result["55"]["action"], "keep")

    def test_two_missing_detections_bridge_but_do_not_emit_ghost_masks(self):
        self.seed_removal()
        self.assertEqual(self.frame(2, []), {})
        self.assertEqual(self.frame(3, []), {})
        result = self.frame(4, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "remove")

    def test_long_gap_or_nonconsecutive_input_breaks_history(self):
        self.seed_removal()
        for i in range(2, 5):
            self.frame(i, [])
        result = self.frame(5, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "keep")
        self.temporal = TemporalRemoval()
        self.seed_removal()
        result = self.frame(3, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "keep")

    def test_person_removal_does_not_seed_bicycle_removal(self):
        for i in range(2):
            self.frame(i, [(46, "remove", self.bike, {"class": 0})])
        result = self.frame(2, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"]["action"], "keep")

    def test_disabled_mode_preserves_base_decisions(self):
        self.temporal = TemporalRemoval(hold=0)
        self.seed_removal()
        result = self.frame(2, [(81, "keep", self.bike, {"class": 1})])
        self.assertEqual(result["81"], {"action": "keep"})


class FlowTests(unittest.TestCase):
    def test_warp_follows_image_translation_and_links_new_id(self):
        rng = np.random.default_rng(17)
        gray = rng.integers(0, 256, (120, 160), dtype=np.uint8)
        mask = np.zeros_like(gray, dtype=bool)
        mask[35:80, 30:70] = True
        temporal = TemporalRemoval()
        for i, (tid, action) in enumerate([(55, "remove"), (55, "remove"), (81, "keep")]):
            transform = np.array([[1, 0, i * 5], [0, 1, 0]], dtype=np.float32)
            image = cv2.warpAffine(gray, transform, (160, 120))
            moved = cv2.warpAffine(mask.astype(np.uint8), transform, (160, 120)).astype(bool)
            decisions = {str(tid): {"action": action}}
            temporal.update(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), [{"id": tid, "class": 1}],
                            [moved], decisions, set(), i, f"{i}.png")
        self.assertEqual(decisions["81"]["action"], "remove")
        self.assertGreater(decisions["81"]["temporal"]["iou"], 0.9)

    def test_inconsistent_forward_backward_flow_is_rejected(self):
        image = np.zeros((40, 60), dtype=np.uint8)
        mask = np.zeros_like(image, dtype=bool)
        mask[10:30, 10:30] = True
        bad_flow = np.full((*image.shape, 2), 5, dtype=np.float32)
        with patch("dynamic_mask_temporal.cv2.calcOpticalFlowFarneback", return_value=bad_flow):
            self.assertFalse(warp_masks(image, image, [mask])[0].any())


if __name__ == "__main__":
    unittest.main()
