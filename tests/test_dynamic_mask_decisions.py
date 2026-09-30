"""Synthetic regressions; no model weights or dataset images are needed."""

from collections import Counter, defaultdict
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import Dynamic03_generate_masks as masks

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
        for cls in (1, 3):
            with self.subTest(cls=cls):
                track = self.track([cls], moving=0, rider_votes=0)
                decision = masks.summarize_tracks({"452": track}, self.args)["452"]
                self.assertEqual(decision["action"], "keep")

    def test_moving_unoccupied_bicycle_is_not_automatically_removed(self):
        track = self.track([1], moving=12, rider_votes=0)
        self.assertEqual(masks.summarize_tracks({"1": track}, self.args)["1"]["action"], "keep")

    def test_people_and_road_vehicles_removed_without_motion_or_rider(self):
        for cls in (0, 2, 5, 7):
            for valid, moving in ((0, 0), (12, 0), (12, 6), (12, 12)):
                with self.subTest(cls=cls, valid=valid, moving=moving):
                    track = self.track([cls], valid=valid, moving=moving)
                    decision = masks.summarize_tracks({"1": track}, self.args)["1"]
                    self.assertEqual(decision["action"], "remove")
                    self.assertFalse(decision["review"])

    def test_mixed_car_bicycle_id_keeps_unridden_bicycle(self):
        decision = masks.summarize_tracks({"1": self.track([2, 1], moving=0)}, self.args)["1"]
        self.assertEqual(decision["action"], "mixed")
        self.assertEqual(masks.automatic_action({"class": 2}, decision), "remove")
        self.assertEqual(masks.automatic_action({"class": 1}, decision), "keep")

    def test_keep_overrides_cannot_retain_mandatory_classes(self):
        overrides = {"tracks": {"1": "keep"}, "frames": {"a.png": {"1": "keep"}}}
        for cls in (0, 2, 5, 7):
            obj = {"class": cls, "id": 1}
            self.assertEqual(masks.removal_action(obj, {"action": "keep"}, overrides, "a.png"), "remove")

    def test_parked_two_wheeler_frame_override_wins_over_track_removal(self):
        overrides = {"tracks": {"1": "remove"}, "frames": {"parked.png": {"1": "keep"}}}
        for cls in (1, 3):
            obj = {"class": cls, "id": 1}
            self.assertEqual(masks.removal_action(obj, {"action": "remove"}, overrides, "parked.png"), "keep")

    def test_reject_keep_overrides_for_people_and_road_vehicles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overrides.json"
            records = [{"name": "a.png", "objects": [{"id": 1}]}]
            for cls in (0, 2, 5, 7):
                for overrides in ({"tracks": {"1": "keep"}},
                                  {"frames": {"a.png": {"1": "keep"}}}):
                    path.write_text(json.dumps(overrides))
                    with self.subTest(cls=cls, overrides=overrides), self.assertRaisesRegex(ValueError, "must be removed"):
                        masks.read_overrides(path, {"1": {"classes": [cls]}}, records)

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
