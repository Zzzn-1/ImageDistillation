"""Adjacent-frame rider evidence and person-associated carried-object masks."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


class RiderRecovery:
    """Recover fast/clipped riders using consecutive-frame flow, without ID-wide deletion."""

    def __init__(self, track_points, background_mask):
        self.track_points, self.background_mask = track_points, background_mask
        self.previous = None
        self.objects = []
        self.index = None
        self.votes = {}

    def update(self, image, objects, seeds, decisions, keep_ids, index):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        votes, changed = {}, 0
        candidates = any(o["class"] in (1, 3) and o.get("rider_candidates")
                         and str(o["id"]) not in keep_ids for o in objects)
        if candidates and self.previous is not None and index == self.index + 1 and gray.shape == self.previous.shape:
            bg = self.background_mask(gray.shape, objects)
            p, q = self.track_points(gray, self.previous, bg)
            if len(p):
                old_bg = self.background_mask(gray.shape, self.objects)
                valid = old_bg[q[:, 1].astype(int), q[:, 0].astype(int)] > 0
                p, q = p[valid], q[valid]
            if len(p) >= 30:
                background = np.median(q - p, axis=0)
                noise = float(np.quantile(np.linalg.norm(q - p - background, axis=1), 0.8))
                cells = (p / np.array([gray.shape[1] / 4, gray.shape[0] / 4])).astype(int)
                # This fallback assumes near-translation. Reject rotation/parallax-heavy backgrounds.
                if noise <= 2.5 and len(set(map(tuple, cells))) >= 3:
                    by_id = {obj["id"]: (obj, seed) for obj, seed in zip(objects, seeds)}
                    motions = {}

                    def motion(tid):
                        if tid not in motions:
                            _, seed = by_id[tid]
                            interior = cv2.erode(seed.astype(np.uint8), np.ones((3, 3), np.uint8))
                            a, b = self.track_points(gray, self.previous, interior, max_corners=120)
                            result = None
                            if len(a) >= 6:
                                residual = b - a - background
                                vector = np.median(residual, axis=0)
                                speed = float(np.linalg.norm(vector))
                                agreement = (np.linalg.norm(residual - vector, axis=1) < max(2, speed * 0.6)).mean()
                                if speed > max(3.0, 4 * noise) and agreement >= 0.7:
                                    result = (vector, speed)
                            motions[tid] = result
                        return motions[tid]

                    for obj in objects:
                        tid = str(obj["id"])
                        if obj["class"] not in (1, 3) or tid in keep_ids:
                            continue
                        for pid in obj.get("rider_candidates", []):
                            if pid not in by_id or by_id[pid][0]["class"] != 0:
                                continue
                            bm, pm = motion(obj["id"]), motion(pid)
                            if bm is None or pm is None or np.linalg.norm(bm[0] - pm[0]) >= max(3, 0.5 * bm[1]):
                                continue
                            key = (obj["id"], pid)
                            votes[key] = self.votes.get(key, 0) + 1
                            if votes[key] >= 2 and decisions[tid]["action"] == "keep":
                                decisions[tid].update(action="remove", base_action="keep",
                                    rider_recovery={"person_id": pid, "consecutive_votes": votes[key],
                                                    "bike_motion_px": round(bm[1], 3),
                                                    "person_motion_px": round(pm[1], 3)})
                                changed += 1
        self.previous, self.objects, self.index, self.votes = gray, objects, index, votes
        return changed


def remove_duplicate_bikes(objects, seeds, decisions, keep_ids):
    """A near-contained duplicate must not protect part of a confirmed removal."""
    sources = [(o, s) for o, s in zip(objects, seeds)
               if o["class"] in (1, 3) and decisions[str(o["id"])]["action"] == "remove"]
    changed = 0
    for obj, seed in zip(objects, seeds):
        tid = str(obj["id"])
        if obj["class"] not in (1, 3) or tid in keep_ids or decisions[tid]["action"] != "keep":
            continue
        area = int(seed.sum())
        for source, mask in sources:
            coverage = float(np.count_nonzero(seed & mask) / max(1, area))
            if area and area <= mask.sum() * 1.05 and coverage >= 0.95:
                decisions[tid].update(action="remove", base_action="keep",
                                      duplicate_of=str(source["id"]), duplicate_coverage=coverage)
                changed += 1
                break
    return changed


def carried_by(box, mask, objects, seeds):
    """Require containment near a person AND contact with that person's mask."""
    best = None
    area = max(1, np.count_nonzero(mask))
    for person, seed in zip(objects, seeds):
        if person["class"] != 0:
            continue
        x1, y1, x2, y2 = person["box"]
        pw, ph = x2 - x1, y2 - y1
        h, w = mask.shape
        region = np.zeros_like(mask)
        region[max(0, int(y1 - .1 * ph)):min(h, int(y2 + .1 * ph) + 1),
               max(0, int(x1 - .2 * pw)):min(w, int(x2 + .2 * pw) + 1)] = True
        coverage = np.count_nonzero(mask & region) / area
        radius = max(3, min(16, round(ph * .06)))
        near = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)).astype(bool)
        contact = np.count_nonzero(mask & near) / area
        if coverage >= .6 and contact >= .02 and (best is None or contact > best[0]):
            best = (contact, person["id"])
    return best[1] if best else None


class CarriedItems:
    LABELS = {24: "backpack", 26: "handbag", 28: "suitcase", 73: "book"}
    TEXT_LABELS = ["backpack", "handbag", "newspaper", "sheet of paper"]

    def __init__(self, model_path, device, conf=.15, world_path=None, prompts_path=None):
        from ultralytics import YOLO
        if not Path(model_path).is_file():
            raise FileNotFoundError(model_path)
        self.model = YOLO(str(model_path))
        if self.model.task != "segment":
            raise ValueError("Carried items require a YOLO segmentation checkpoint")
        if any(self.model.names.get(cls) != label for cls, label in self.LABELS.items()):
            raise ValueError("Carried items require COCO backpack/handbag/suitcase/book classes")
        self.device, self.conf = device, conf
        self.world = None
        if world_path:
            if not Path(world_path).is_file():
                raise FileNotFoundError(world_path)
            if importlib.util.find_spec("clip") is None:
                raise RuntimeError("YOLO-World requires CLIP: pip install git+https://github.com/ultralytics/CLIP.git")
            from ultralytics import YOLOWorld
            self.world = YOLOWorld(str(world_path))
            self.world.set_classes(self.TEXT_LABELS)
        self.prompts = json.loads(Path(prompts_path).read_text()) if prompts_path else {}

    def detect(self, image, objects, seeds, predictor, frame_name):
        h, w = image.shape[:2]
        people = [o for o in objects if o["class"] == 0]
        if not people:
            return []
        candidates = []
        # Full frame plus large-person crops; crops recover small bags in full-image inference.
        rois = [(0, 0, w, h)]
        for p in people:
            x1, y1, x2, y2 = p["box"]
            if y2 - y1 >= 100:
                dx, dy = .15 * (x2 - x1), .1 * (y2 - y1)
                rois.append((max(0, int(x1 - dx)), max(0, int(y1 - dy)),
                             min(w, int(x2 + dx) + 1), min(h, int(y2 + dy) + 1)))
        for x1, y1, x2, y2 in dict.fromkeys(rois):
            result = self.model.predict(image[y1:y2, x1:x2], classes=list(self.LABELS),
                conf=self.conf, imgsz=960, retina_masks=True, device=self.device, verbose=False)[0]
            if result.masks is None:
                continue
            for box, cls, score, local in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                            result.boxes.conf.cpu().numpy(), result.masks.data.cpu().numpy()):
                mask = np.zeros((h, w), dtype=bool)
                mask[y1:y2, x1:x2] = cv2.resize(local, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST) > .5
                candidates.append((box + [x1, y1, x1, y1], self.LABELS[int(cls)], float(score), mask))
        boxes = list(self.prompts.get(frame_name, []))
        if self.world is not None:
            result = self.world.predict(image, conf=self.conf, imgsz=960, device=self.device, verbose=False)[0]
            boxes += [{"box": b.tolist(), "label": self.TEXT_LABELS[int(c)], "score": float(s)}
                      for b, c, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                         result.boxes.conf.cpu().numpy())]
        for entry in boxes:
            box = np.asarray(entry["box"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError(f"Invalid carried-object box in {frame_name}: {box}")
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(bool)
            bounds = np.zeros_like(mask)
            a, b, c, d = box.astype(int)
            bounds[b:d+1, a:c+1] = True
            mask &= bounds
            if np.isfinite(scores[0]) and scores[0] >= .5 and mask.any():
                candidates.append((box, entry.get("label", "newspaper"), entry.get("score", 1), mask))
        selected = []
        for box, label, score, mask in candidates:
            pid = carried_by(box, mask, objects, seeds)
            if pid is None:
                continue
            if any(np.count_nonzero(mask & old[1]) / max(1, min(mask.sum(), old[1].sum())) > .85 for old in selected):
                continue
            obj = {"id": f"carried_{len(selected)}", "class": -1, "label": label,
                   "box": box.tolist(), "confidence": score, "carried_by": pid}
            selected.append((obj, mask))
        return selected
