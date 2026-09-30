"""Short-lived two-wheeler removal decisions linked by optical-flow mask overlap.

Only match existing detections; never synthesize masks for an undetected object.
Propagated decisions do not renew their own expiry time.
"""

import cv2
import numpy as np


def warp_masks(previous, current, masks):
    """Warp masks with backward flow, rejecting inconsistent/occluded pixels."""
    forward = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(current, previous, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    h, w = current.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = x + backward[..., 0], y + backward[..., 1]
    sampled = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (np.linalg.norm(sampled + backward, axis=2) < 1.5)
    valid &= (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    warped = []
    for mask in masks:
        moved = cv2.remap(mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) & valid
        # Large changes indicate clipping, occlusion, or unreliable flow.
        if not 0.5 * mask.sum() <= moved.sum() <= 2 * mask.sum():
            moved[:] = False
        warped.append(moved)
    return warped


def mask_matches(previous, current, min_iou):
    """Mutual best matches with a margin over both competing objects and histories."""
    scores = np.zeros((len(previous), len(current)), dtype=np.float32)
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            intersection = np.count_nonzero(old & new)
            if intersection:
                scores[i, j] = intersection / np.count_nonzero(old | new)
    matches = {}
    if not scores.size:
        return matches, scores
    for i in range(len(previous)):
        j = int(scores[i].argmax())
        score = float(scores[i, j])
        if score < min_iou or int(scores[:, j].argmax()) != i:
            continue
        row_other = float(np.max(np.delete(scores[i], j), initial=0))
        col_other = float(np.max(np.delete(scores[:, j], i), initial=0))
        if score - max(row_other, col_other) < 0.1:
            continue
        matches[j] = (i, score)
    return matches, scores


class TemporalRemoval:
    """Confirm on two matched remove frames, then bridge short ID disruptions.

Fresh base removal or fresh rider co-motion can renew a confirmed history.
Otherwise hold counts elapsed source frames, including missing detections.
Explicit keep clears matched history; ambiguous matches never propagate.
"""

    def __init__(self, hold=5, min_iou=0.5):
        if hold < 0 or not 0 < min_iou <= 1:
            raise ValueError("Require hold >= 0 and 0 < min_iou <= 1")
        self.hold = hold
        self.min_iou = min_iou
        self.states = []
        self.previous = None
        self.previous_index = None

    def update(self, image, objects, seeds, decisions, keep_ids, index, frame_name):
        """Update decisions in place and return the number changed from keep to remove."""
        if not self.hold:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = min(1.0, 640 / max(h, w))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        if scale < 1:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        bikes, masks = [], []
        for obj, seed in zip(objects, seeds):
            if obj["class"] in (1, 3):
                bikes.append(obj)
                mask = seed.astype(np.uint8)
                if scale < 1:
                    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                masks.append(mask.astype(bool))
        if (self.previous is None or self.previous.shape != gray.shape
                or index != self.previous_index + 1):
            self.states = []
        warped, matches = [], {}
        scores = np.zeros((len(self.states), len(bikes)), dtype=np.float32)
        if any(state["confirmed"] for state in self.states):
            warped = warp_masks(self.previous, gray, [s["mask"] for s in self.states])
            matches, scores = mask_matches(warped, masks, self.min_iou)

        next_states, used = [], set()
        changed = 0
        for j, (obj, mask) in enumerate(zip(bikes, masks)):
            tid = str(obj["id"])
            base_action = decisions[tid]["action"]
            old = None
            if j in matches:
                i, iou = matches[j]
                used.add(i)
                old = self.states[i]
            alive = old is not None and index - old["anchor_index"] <= self.hold
            confirmed = 0
            anchor_index, anchor_id, anchor_frame = index, tid, frame_name
            if tid not in keep_ids:
                if base_action == "remove":
                    # A single old observation followed by a gap is not two consecutive frames.
                    confirmed = 1
                    if alive and (old["confirmed"] >= 2 or old["seen"] == index - 1):
                        confirmed = min(2, old["confirmed"] + 1)
                elif alive and old["confirmed"] >= 2:
                    decisions[tid].update(action="remove", base_action=base_action,
                                          temporal={"source_id": old["anchor_id"],
                                                    "source_frame": old["anchor_frame"],
                                                    "previous_id": old["id"],
                                                    "iou": round(iou, 4),
                                                    "age": index - old["anchor_index"]})
                    changed += 1
                    confirmed = 2
                    anchor_index = old["anchor_index"]
                    anchor_id, anchor_frame = old["anchor_id"], old["anchor_frame"]
                    # This is new analysis evidence, not a propagated decision voting for itself.
                    if any(pid >= 0 for pid in obj.get("rider_matches", [])):
                        anchor_index, anchor_id, anchor_frame = index, tid, frame_name
                        decisions[tid]["temporal"]["renewed_by_rider_match"] = True
            next_states.append({"mask": mask, "id": tid, "confirmed": confirmed,
                                "anchor_index": anchor_index, "anchor_id": anchor_id,
                                "anchor_frame": anchor_frame, "seen": index})

        # Carry a confirmed history through at most two absent detections, without emitting a mask.
        # Discard any overlapping unmatched history: ambiguity or keep must not be bypassed later.
        for i, old in enumerate(self.states):
            if (i not in used and old["confirmed"] >= 2 and warped[i].any()
                    and index - old["seen"] <= 2 and index - old["anchor_index"] <= self.hold
                    and float(np.max(scores[i], initial=0)) < 0.1):
                next_states.append(dict(old, mask=warped[i]))
        self.states = next_states
        self.previous, self.previous_index = gray, index
        return changed
