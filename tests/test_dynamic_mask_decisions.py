"""Synthetic regressions; no model weights or dataset images are needed.

These tests were authored but not executed, as requested by the user.
"""

from collections import Counter, defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_dynamic_masks as masks

masks.cv2, masks.np = cv2, np


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(min_votes=3, moving_ratio=0.6, motion_px=2.5, motion_relative=0.02)

    def track(self, classes, valid=12, moving=6, rider_votes=0):
        return {"classes": set(classes), "first": "first.png", "last": "last.png", "frames": 60,
                "valid": set(range(valid)), "moving": set(range(moving)),
                "riders": defaultdict(set, {229: set(range(rider_votes))}),
                "motion_sources": Counter(), "candidate": True}

    def test_bicycle_motorcycle_label_flip_and_low_motion_ratio(self):
        track = self.track([1, 3], rider_votes=5)
        decision = masks.summarize_tracks({"237": track}, self.args)["237"]
        self.assertEqual(decision["action"], "remove")
        self.assertEqual(decision["actions_by_class"], {"1": "remove", "3": "remove"})

    def test_person_near_parked_motorcycle_is_not_enough(self):
        track = self.track([3], moving=0, rider_votes=0)
        decision = masks.summarize_tracks({"452": track}, self.args)["452"]
        self.assertEqual(decision["action"], "keep")

    def test_moving_unoccupied_bicycle_is_not_automatically_removed(self):
        track = self.track([1], moving=12, rider_votes=0)
        self.assertEqual(masks.summarize_tracks({"1": track}, self.args)["1"]["action"], "keep")

    def test_road_vehicle_ratio_is_not_relaxed(self):
        track = self.track([2, 5, 7], moving=6, rider_votes=5)
        self.assertEqual(masks.summarize_tracks({"1": track}, self.args)["1"]["action"], "keep")

    def test_mixed_person_vehicle_id_does_not_delete_vehicle_by_person_label(self):
        track = self.track([0, 3], valid=1, moving=1)
        decision = masks.summarize_tracks({"403": track}, self.args)["403"]
        self.assertEqual(decision["action"], "mixed")
        self.assertEqual(masks.automatic_action({"class": 0}, decision), "remove")
        self.assertEqual(masks.automatic_action({"class": 3}, decision), "keep")

    def test_spaced_votes_do_not_require_fixed_modulo_and_do_not_overcount(self):
        votes = set()
        for index in [1, 2, 3, 6, 7, 11, 12]:
            masks.add_spaced_vote(votes, index, lag=5)
        self.assertEqual(votes, {1, 6, 11})

    def test_box_motion_compensates_camera_and_rejects_clipping(self):
        old = {"class": 3, "confidence": 0.9, "box": [100, 100, 180, 160]}
        camera = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        stationary = dict(old, box=[120, 100, 200, 160])
        moving = dict(old, box=[145, 100, 225, 160])
        clipped = dict(old, box=[0, 100, 80, 160])
        evidence = lambda obj: masks.box_motion_evidence(old, obj, camera, np.zeros(2), 0.2, (480, 640), self.args)
        self.assertFalse(evidence(stationary)["moving"])
        self.assertTrue(evidence(moving)["moving"])
        self.assertEqual(evidence(moving)["source"], "box_fallback")
        self.assertIsNone(evidence(clipped))

    def test_rider_must_share_motion_not_only_overlap(self):
        def obj(tid, cls, box, vector, moving):
            return {"id": tid, "class": cls, "box": box, "rider_candidates": [], "rider_matches": [],
                    "motion": {"vector": vector, "pixels": float(np.linalg.norm(vector)), "moving": moving}}
        person = obj(403, 0, [30, 10, 60, 90], [14, 0], True)
        bike = obj(452, 3, [20, 55, 100, 100], [12, 0], True)
        masks.rider_candidates([person, bike])
        self.assertEqual(bike["rider_matches"], [403])
        parked = obj(453, 3, [20, 55, 100, 100], [0, 0], False)
        masks.rider_candidates([person, parked])
        self.assertEqual(parked["rider_candidates"], [403])
        self.assertEqual(parked["rider_matches"], [])

    def test_family_tracking_maps_subset_indices_and_namespaces_ids(self):
        class Boxes:
            def __init__(self, classes):
                self.cls = np.asarray(classes)

            def cpu(self):
                return self

            def numpy(self):
                return self

            def __getitem__(self, indices):
                return Boxes(self.cls[indices])

        class Tracker:
            def __init__(self):
                self.seen = []

            def update(self, boxes, image):
                self.seen.append(boxes.cls.tolist())
                return np.array([[0, 0, 10, 10, i+1, 0.9, cls, i] for i, cls in enumerate(boxes.cls)])

        tracker = masks.FamilyTracker.__new__(masks.FamilyTracker)
        tracker.trackers = {name: Tracker() for name in ("person", "two_wheeler", "road_vehicle")}
        tracker.ids = {}
        first = tracker.update(Boxes([3, 0, 2]), None)
        second = tracker.update(Boxes([1, 0, 2]), None)
        self.assertEqual(set(first), {0, 1, 2})
        self.assertEqual(len(set(first.values())), 3)
        self.assertEqual(first, second)  # bicycle/motorcycle share one family
        self.assertEqual(tracker.trackers["person"].seen, [[0], [0]])
        self.assertEqual(tracker.trackers["two_wheeler"].seen, [[3], [1]])
        tracker.update(Boxes([]), None)
        self.assertTrue(all(t.seen[-1] == [] for t in tracker.trackers.values()))


if __name__ == "__main__":
    unittest.main()
