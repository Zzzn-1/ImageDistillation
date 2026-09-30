"""Regressions for rider recovery, carried-object association and render resume."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import Dynamic03_generate_masks as main
from dynamic_mask_refinement import RiderRecovery, CarriedItems, carried_by, remove_duplicate_bikes
from dynamic_mask_resume import completed_prefix

main.cv2, main.np = cv2, np


class RefinementTests(unittest.TestCase):
    def test_adjacent_motion_recovers_fast_rider_but_not_parked_bike(self):
        for bike_speed in (0, 7):
            rng = np.random.default_rng(42)
            background = rng.integers(0, 256, (320, 480, 3), dtype=np.uint8)
            person_texture = rng.integers(0, 256, (100, 70, 3), dtype=np.uint8)
            bike_texture = rng.integers(0, 256, (65, 120, 3), dtype=np.uint8)
            recovery = RiderRecovery(main.track_points, main.background_mask)
            for i in range(3):
                image = background.copy()
                x, bx = 125 + 7 * i, 100 + bike_speed * i
                image[70:170, x:x+70] = person_texture
                image[180:245, bx:bx+120] = bike_texture
                seeds = np.zeros((2, 320, 480), dtype=bool)
                seeds[0, 70:170, x:x+70] = True
                seeds[1, 180:245, bx:bx+120] = True
                objects = [{"id": 1, "class": 0, "box": [x,70,x+70,170]},
                           {"id": 2, "class": 1, "box": [bx,180,bx+120,245], "rider_candidates": [1]}]
                decisions = {"1": {"action": "remove"}, "2": {"action": "keep"}}
                recovery.update(image, objects, seeds, decisions, set(), i)
            self.assertEqual(decisions["2"]["action"], "remove" if bike_speed else "keep")

    def test_duplicate_fragment_does_not_protect_confirmed_bike(self):
        masks = np.zeros((3, 30, 30), dtype=bool)
        masks[0, 3:20, 3:20] = True
        masks[1, 3:12, 3:12] = True
        masks[2, 23:28, 23:28] = True
        objects = [{"id": i, "class": 1} for i in range(3)]
        decisions = {str(i): {"action": "remove" if i == 0 else "keep"} for i in range(3)}
        self.assertEqual(remove_duplicate_bikes(objects, masks, decisions, set()), 1)
        self.assertEqual(decisions["1"]["action"], "remove")
        self.assertEqual(decisions["2"]["action"], "keep")
        decisions["1"] = {"action": "keep"}
        self.assertEqual(remove_duplicate_bikes(objects, masks, decisions, {"1"}), 0)

    def test_carried_item_needs_person_contact_not_just_box_overlap(self):
        person = np.zeros((80, 80), dtype=bool)
        person[10:65, 30:40] = True
        objects = [{"id": 1, "class": 0, "box": [20, 10, 60, 65]}]
        bag = np.zeros_like(person)
        bag[20:35, 40:46] = True
        self.assertEqual(carried_by([40,20,46,35], bag, objects, [person]), 1)
        remote = np.zeros_like(person)
        remote[20:35, 60:66] = True
        self.assertIsNone(carried_by([60,20,66,35], remote, objects, [person]))

    def test_newspaper_prompt_uses_sam_mask_not_a_filled_box(self):
        detector = CarriedItems.__new__(CarriedItems)
        detector.model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(masks=None)])
        detector.world, detector.device, detector.conf = None, "cpu", .15
        detector.prompts = {"frame.png": [{"box": [39,20,49,40], "label": "newspaper"}]}
        person = np.zeros((80,80), dtype=bool)
        person[10:65, 30:40] = True
        paper = np.zeros_like(person)
        paper[22:36, 40:44] = True
        predictor = SimpleNamespace(predict=lambda **k: (paper[None], np.array([.9]), None))
        found = detector.detect(np.zeros((80,80,3), dtype=np.uint8),
            [{"id": 1, "class": 0, "box": [20,10,60,65]}], [person], predictor, "frame.png")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0]["label"], "newspaper")
        self.assertEqual(found[0][0]["carried_by"], 1)
        np.testing.assert_array_equal(found[0][1], paper)
        predictor.predict = lambda **k: (paper[None], np.array([.1]), None)
        self.assertEqual(detector.detect(np.zeros((80,80,3), dtype=np.uint8),
            [{"id": 1, "class": 0, "box": [20,10,60,65]}], [person], predictor, "frame.png"), [])


class ResumeIntegrationTests(unittest.TestCase):
    def test_interrupt_resume_replays_temporal_state_without_rewriting_complete_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            analysis, output = root / "analysis", root / "output"
            analysis.mkdir()
            (analysis / "seeds").mkdir()
            mask = np.zeros((32,32), dtype=bool)
            mask[8:24,8:24] = True
            image = np.random.default_rng(19).integers(0,256,(32,32,3),dtype=np.uint8)
            records = []
            for i in range(4):
                source = root / f"{i}.png"
                cv2.imwrite(str(source), image)
                seed_file = f"seeds/{i}.npz"
                np.savez_compressed(analysis / seed_file, packed=np.packbits(mask[None],axis=-1))
                records.append({"index": i, "name": source.name, "source": str(source),
                                "shape": [32,32], "size": source.stat().st_size,
                                "mtime_ns": source.stat().st_mtime_ns, "seeds": seed_file,
                                "objects": [{"id": 1 if i < 2 else 2, "class": 1, "box": [8,8,24,24]}]})
            tracks = {"1": {"action": "remove", "classes": [1]}, "2": {"action": "keep", "classes": [1]}}
            main.write_json(analysis / "tracks.json", tracks)
            main.write_json(analysis / "summary.json", {"complete": True, "decision_version": 3, "frames": 4})
            (analysis / "frames.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in records))
            weight = root / "dummy.pt"
            weight.write_bytes(b"test")
            args = SimpleNamespace(stage="render", analysis=analysis, output=output, overrides=None,
                                   sam_checkpoint=weight, sam_config="test", device="cpu", dilate=0,
                                   sam_min_score=.5, overlay_every=1, temporal_hold=5, temporal_iou=.5,
                                   resume=False, no_rider_recovery=True, no_carried_items=True,
                                   carried_yolo=weight, carried_conf=.15, carried_world=None, carried_prompts=None)

            class Predictor:
                calls = 0
                interrupt = True
                def __init__(self, model):
                    pass
                def set_image(self, frame):
                    Predictor.calls += 1
                    if Predictor.interrupt and Predictor.calls == 3:
                        raise KeyboardInterrupt()
                def predict(self, **kwargs):
                    return mask[None].copy(), np.array([1.0]), None

            modules = {"torch": SimpleNamespace(device=lambda s: SimpleNamespace(type="cpu"), inference_mode=nullcontext),
                       "sam2.build_sam": SimpleNamespace(build_sam2=lambda *a,**k: None),
                       "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=Predictor)}
            with patch.dict(sys.modules, modules):
                with self.assertRaises(KeyboardInterrupt):
                    main.render(args)
                self.assertEqual(len((output / "manifest.jsonl").read_text().splitlines()), 2)
                before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in output.rglob('*.png')}
                # Simulate a torn final line left by abrupt process termination.
                with (output / "manifest.jsonl").open('ab') as f:
                    f.write(b'{"source":')
                Predictor.interrupt = False
                Predictor.calls = 0
                args.resume = True
                main.render(args)
                self.assertEqual(Predictor.calls, 2)  # Completed frames did not invoke SAM2.
                for name, value in before.items():
                    p = Path(name)
                    self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), value)
                manifest = [json.loads(s) for s in (output / "manifest.jsonl").read_text().splitlines()]
                self.assertEqual(len(manifest), 4)
                self.assertEqual(manifest[2]['decisions']['2']['action'], 'remove')
                self.assertEqual(json.loads((output / 'summary.json').read_text())['frames'], 4)
                # A completed resume is a no-op for models.
                Predictor.calls = 0
                main.render(args)
                self.assertEqual(Predictor.calls, 0)
            saved = json.loads((output / 'config.json').read_text())
            self.assertEqual(len(completed_prefix(output, records, saved, tracks)[0]), 4)
            changed = dict(saved, dilate=3)
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                completed_prefix(output, records, changed, tracks)
            path = output / 'manifest.jsonl'
            full = path.read_bytes()
            path.write_bytes(full.rstrip(b'\n'))
            self.assertEqual(len(completed_prefix(output, records, saved, tracks)[0]), 3)
            path.write_bytes(full)
            (output / 'masks/2.png').write_bytes(b'incomplete PNG')
            self.assertEqual(len(completed_prefix(output, records, saved, tracks)[0]), 2)


if __name__ == '__main__':
    unittest.main()
