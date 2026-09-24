#!/usr/bin/env python3
"""Analyze vehicle tracks, then render reviewed YOLO11 + SAM2 removal masks.

No single-image classifier can establish physical motion or rider identity.
Automatic decisions are conservative heuristics; inspect tracks before rendering.
"""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DECISION_VERSION = 2


def class_family(cls):
    if cls == 0:
        return "person"
    return "two_wheeler" if cls in (1, 3) else "road_vehicle"


class FamilyTracker:
    """Prevent a rider ID from being reassigned to the vehicle or vice versa."""

    def __init__(self, config):
        from ultralytics.trackers.bot_sort import BOTSORT

        self.trackers = {family: BOTSORT(SimpleNamespace(**config))
                         for family in ("person", "two_wheeler", "road_vehicle")}
        self.ids = {}

    def update(self, boxes, image):
        detected = boxes.cpu().numpy()
        classes = detected.cls.astype(int)
        ids = {}
        for family, tracker in self.trackers.items():
            indices = np.array([i for i, cls in enumerate(classes) if class_family(cls) == family], dtype=int)
            # Empty updates also age lost tracks in this family.
            for row in tracker.update(detected[indices], image):
                key = (family, int(row[4]))
                if key not in self.ids:
                    self.ids[key] = len(self.ids) + 1
                ids[int(indices[int(row[-1])])] = self.ids[key]
        return ids


def add_spaced_vote(votes, index, lag):
    """Accept the first available evidence, then enforce spacing between votes."""
    if not votes or index - max(votes) >= lag:
        votes.add(index)


def summarize_tracks(tracks, args):
    decisions = {}
    for tid, t in tracks.items():
        valid, moving = len(t["valid"]), len(t["moving"])
        riders = {str(pid): len(votes) for pid, votes in t["riders"].items()}
        rider_votes = max(riders.values(), default=0)
        dynamic = moving >= args.min_votes and moving / max(valid, 1) >= args.moving_ratio
        # Each rider vote ALREADY requires BOTH objects to move together after
        # camera compensation. A whole-track moving ratio must not veto it.
        riding = rider_votes >= args.min_votes
        actions, reasons = {}, {}
        for cls in sorted(t["classes"]):
            action, reason = "keep", "uncertain_or_no_confirmed_motion"
            if cls == 0:
                action, reason = "remove", "person"
            elif cls in (2, 5, 7) and dynamic:
                action, reason = "remove", "motion_evidence"
            elif cls in (1, 3) and riding:
                action, reason = "remove", "rider_geometry_and_comotion"
            actions[str(cls)], reasons[str(cls)] = action, reason
        unique_actions = set(actions.values())
        action = next(iter(unique_actions)) if len(unique_actions) == 1 else "mixed"
        unique_reasons = set(reasons.values())
        reason = next(iter(unique_reasons)) if len(unique_reasons) == 1 else "per_class_decision"
        decisions[tid] = {"classes": sorted(t["classes"]), "action": action, "reason": reason,
                          "actions_by_class": actions, "reasons_by_class": reasons,
                          "review": t["classes"] != {0}, "first": t["first"], "last": t["last"],
                          "frames": t["frames"], "valid_motion_votes": valid, "moving_votes": moving,
                          "rider_votes": rider_votes, "rider_votes_by_person": riders,
                          "motion_sources": dict(t["motion_sources"]),
                          "rider_candidate": t["candidate"], "decision_version": DECISION_VERSION}
    return decisions


def automatic_action(obj, track):
    if obj["class"] == 0:
        return "remove"
    return track.get("actions_by_class", {}).get(str(obj["class"]), track["action"])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fresh_directory(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    analyze = sub.add_parser("analyze", help="YOLO tracking and motion evidence; no SAM2")
    analyze.add_argument("--input", type=Path, default=ROOT / "dataset/Dynamic03_raw")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--yolo", type=Path, default=ROOT / "models/yolo11x-seg.pt")
    analyze.add_argument("--tracker", default="botsort.yaml")
    analyze.add_argument("--device", default="cuda:0")
    analyze.add_argument("--conf", type=float, default=0.15)
    analyze.add_argument("--imgsz", type=int, default=960)
    analyze.add_argument("--start", type=int, default=0)
    analyze.add_argument("--limit", type=int, default=0)
    analyze.add_argument("--motion-lag", type=int, default=5, help="Compare frames this far apart")
    analyze.add_argument("--motion-px", type=float, default=2.5, help="Minimum residual over motion-lag, original pixels")
    analyze.add_argument("--motion-relative", type=float, default=0.02, help="Residual / object box diagonal")
    analyze.add_argument("--min-votes", type=int, default=3, help="Motion/rider votes spaced by at least motion-lag frames")
    analyze.add_argument("--moving-ratio", type=float, default=0.6)
    analyze.add_argument("--overlay-every", type=int, default=10)
    render = sub.add_parser("render", help="SAM2 masks from analyzed tracks and optional decisions")
    render.add_argument("--analysis", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--overrides", type=Path, help='JSON: {"tracks":{"42":"keep"},"frames":{"name.png":{"43":"remove"}}}')
    render.add_argument("--sam-checkpoint", type=Path, default=ROOT / "models/sam2.1_hiera_base_plus.pt")
    render.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    render.add_argument("--device", default="cuda:0")
    render.add_argument("--dilate", type=int, default=2)
    render.add_argument("--sam-min-score", type=float, default=0.5)
    render.add_argument("--overlay-every", type=int, default=10)
    args = parser.parse_args()
    if args.overlay_every < 0:
        parser.error("overlay-every must be nonnegative")
    if args.stage == "analyze":
        if min(args.start, args.limit) < 0 or min(args.imgsz, args.motion_lag, args.min_votes) < 1:
            parser.error("Invalid frame range, image size, lag or vote count")
        if not 0 < args.conf <= 1 or not 0 < args.moving_ratio <= 1:
            parser.error("conf and moving-ratio must be in (0, 1]")
        if args.motion_px <= 0 or args.motion_relative < 0:
            parser.error("motion-px must be positive; motion-relative must be nonnegative")
    elif args.dilate < 0 or not 0 <= args.sam_min_score <= 1:
        parser.error("Invalid dilation or SAM score threshold")
    return args


def track_points(old, new, mask, max_corners=1000):
    """Forward/backward checked KLT, coordinates in original image pixels."""
    points = cv2.goodFeaturesToTrack(old, maxCorners=max_corners, qualityLevel=0.01,
                                    minDistance=5, mask=mask.astype(np.uint8) * 255)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 2))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None)
    if forward is None:
        return np.empty((0, 2)), np.empty((0, 2))
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(new, old, forward, None)
    if backward is None:
        return np.empty((0, 2)), np.empty((0, 2))
    p, q, r = points[:, 0], forward[:, 0], backward[:, 0]
    good = (status[:, 0] > 0) & (reverse_status[:, 0] > 0)
    good &= np.isfinite(q).all(axis=1) & (np.linalg.norm(p - r, axis=1) < 1.0)
    good &= (q[:, 0] >= 0) & (q[:, 0] < new.shape[1])
    good &= (q[:, 1] >= 0) & (q[:, 1] < new.shape[0])
    return p[good], q[good]


def background_mask(shape, objects):
    mask = np.ones(shape, dtype=np.uint8)
    h, w = shape
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        mask[max(0, int(y1)-8):min(h, int(y2)+9),
             max(0, int(x1)-8):min(w, int(x2)+9)] = 0
    return mask


def box_motion_evidence(old_obj, obj, homography, correction, noise, shape, args):
    """Fallback only for two-wheelers/riders; use a higher jitter threshold."""
    if obj["class"] not in (0, 1, 3) or min(old_obj["confidence"], obj["confidence"]) < 0.35:
        return None
    h, w = shape
    old_box, box = np.asarray(old_obj["box"]), np.asarray(obj["box"])
    for bounds in (old_box, box):
        if bounds[0] < 2 or bounds[1] < 2 or bounds[2] > w-2 or bounds[3] > h-2:
            return None  # Image-edge clipping can create a large false center displacement.
    x1, y1, x2, y2 = old_box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    projected = cv2.perspectiveTransform(corners[None], homography)[0] + correction
    if not np.isfinite(projected).all():
        return None
    expected_size = projected.max(axis=0) - projected.min(axis=0)
    size = box[2:] - box[:2]
    if min(size) < 10 or min(expected_size) < 10:
        return None
    ratio = size / expected_size
    if np.any(ratio < 0.65) or np.any(ratio > 1.55):
        return None  # Unstable detections/occlusion should not count as object motion.
    vector = (box[:2] + box[2:]) / 2 - (projected.min(axis=0) + projected.max(axis=0)) / 2
    speed = float(np.linalg.norm(vector))
    threshold = max(args.motion_px * 2, max(0.06, args.motion_relative) * float(np.linalg.norm(size)), 4 * noise)
    return {"vector": vector.tolist(), "pixels": speed, "threshold": threshold,
            "points": 0, "moving": speed > threshold, "source": "box_fallback"}


def motion_evidence(previous, gray, objects, seeds, args):
    old_gray, old_objects, old_seeds = previous
    old_bg = background_mask(gray.shape, old_objects)
    new_bg = background_mask(gray.shape, objects)
    p, q = track_points(old_gray, gray, old_bg)
    if len(p):
        good = new_bg[q[:, 1].astype(int), q[:, 0].astype(int)] > 0
        p, q = p[good], q[good]
    if len(p) < 40:
        return "insufficient_background"
    homography, inliers = cv2.findHomography(p, q, cv2.RANSAC, 2.0)
    if homography is None or not np.isfinite(homography).all():
        return "camera_fit_failed"
    inliers = inliers.ravel().astype(bool)
    h, w = gray.shape
    cells = (p[inliers] / np.array([w / 4, h / 4])).astype(int)
    if inliers.sum() < 30 or inliers.mean() < 0.6 or len(set(map(tuple, cells))) < 6:
        return "unreliable_camera_fit"
    bg_residual = q - cv2.perspectiveTransform(p.astype(np.float32)[None], homography)[0]
    old_by_id = {obj["id"]: (obj, mask) for obj, mask in zip(old_objects, old_seeds)}
    for obj, seed in zip(objects, seeds):
        pair = old_by_id.get(obj["id"])
        if (pair is None or class_family(pair[0]["class"]) != class_family(obj["class"])
                or obj["id"] < 0):
            continue
        old_obj, old_seed = pair
        box = np.asarray(old_obj["box"])
        center = (box[:2] + box[2:]) / 2
        diagonal = float(np.linalg.norm(box[2:] - box[:2]))
        distance = np.linalg.norm(p - center, axis=1)
        near = np.argsort(distance)[:80]
        near = near[distance[near] < max(100, diagonal * 2)]
        if len(near) < 15:
            continue
        # Local residuals partly compensate parallax; retain outliers to estimate noise.
        correction = np.median(bg_residual[near], axis=0)
        noise = float(np.quantile(np.linalg.norm(bg_residual[near] - correction, axis=1), 0.9))
        fallback = box_motion_evidence(old_obj, obj, homography, correction, noise, gray.shape, args)
        if fallback is not None:
            obj["motion"] = fallback
        interior = cv2.erode(old_seed.astype(np.uint8), np.ones((3, 3), np.uint8))
        a, b = track_points(old_gray, gray, interior, max_corners=120)
        if len(a):
            good = seed[b[:, 1].astype(int), b[:, 0].astype(int)]
            a, b = a[good], b[good]
        if len(a) < 6:
            continue
        residual = b - cv2.perspectiveTransform(a.astype(np.float32)[None], homography)[0] - correction
        vector = np.median(residual, axis=0)
        speed = float(np.linalg.norm(vector))
        threshold = max(args.motion_px, args.motion_relative * diagonal, 3 * noise)
        agreement = float((np.linalg.norm(residual - vector, axis=1) < max(2, speed * 0.5)).mean())
        if agreement < 0.7:
            continue  # Retain the stricter box fallback if mask flow is incoherent.
        obj["motion"] = {"vector": vector.tolist(), "pixels": speed, "threshold": threshold,
                         "points": len(a), "moving": speed > threshold, "source": "mask_flow"}
    return "ok"


def rider_candidates(objects):
    """Box geometry + co-motion is evidence, not a trained riding-action classifier."""
    people = [obj for obj in objects if obj["class"] == 0]
    for bike in (obj for obj in objects if obj["class"] in (1, 3)):
        x1, y1, x2, y2 = bike["box"]
        for person in people:
            a, b, c, d = person["box"]
            pw, ph = max(c-a, 1), max(d-b, 1)
            overlap = max(0, min(x2, c) - max(x1, a)) / min(max(x2-x1, 1), pw)
            hip_x, hip_y = (a+c)/2, b + 0.6 * ph
            plausible = (overlap > 0.45 and b < y1 + 0.25 * (y2-y1)
                         and x1 <= hip_x <= x2 and y1 - 0.2 * ph <= hip_y <= y2
                         and d > y1 and d < y2 + 0.3 * ph)
            if not plausible:
                continue
            bike["rider_candidates"].append(person["id"])
            bm, pm = bike.get("motion"), person.get("motion")
            if bm and pm and bm["moving"] and pm["moving"]:
                error = np.linalg.norm(np.asarray(bm["vector"]) - pm["vector"])
                if error < max(2.5, 0.4 * bm["pixels"]):
                    bike["rider_matches"].append(person["id"])


def draw_objects(image, objects, decisions=None):
    canvas = image.copy()
    for obj in objects:
        decision = decisions.get(str(obj["id"]), {}) if decisions else {}
        color = (0, 0, 255) if decision.get("action") == "remove" else (0, 255, 255)
        x1, y1, x2, y2 = map(int, obj["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        label = f'{obj["id"]} {CLASSES[obj["class"]]} {decision.get("action", "")}'
        cv2.putText(canvas, label, (x1, max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return canvas


def analyze(args):
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    if not args.yolo.is_file():
        raise FileNotFoundError(args.yolo)
    files = sorted((p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                   key=lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)])
    files = files[args.start:args.start + args.limit if args.limit else None]
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError("No frames, or duplicate image stems")
    model = YOLO(str(args.yolo))
    if model.task != "segment" or any(model.names[k] != v for k, v in CLASSES.items()):
        raise ValueError("Use COCO YOLO11 segmentation weights (*-seg.pt)")
    tracker_config = YAML.load(check_yaml(args.tracker))
    if tracker_config["tracker_type"] != "botsort" or tracker_config.get("with_reid", False):
        raise ValueError("Use BoT-SORT with with_reid: false")
    tracker = FamilyTracker(tracker_config)
    fresh_directory(args.output)
    for name in ("seeds", "overlays"):
        (args.output / name).mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(decision_version=DECISION_VERSION, tracking="separate_class_families")
    write_json(args.output / "config.json", config)
    history = deque(maxlen=args.motion_lag)
    tracks = {}
    shape = None
    missing_id = -1
    camera_counts = Counter()
    with (args.output / "frames.jsonl").open("w") as manifest:
        for index, path in enumerate(files):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read {path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if shape is not None and gray.shape != shape:
                raise ValueError("All frames must have the same dimensions")
            shape = gray.shape
            result = model.predict(image, classes=list(CLASSES), conf=args.conf, imgsz=args.imgsz,
                                   retina_masks=True, device=args.device, verbose=False)[0]
            boxes = result.boxes
            # Keep EVERY raw detection. model.track() can filter tentative detections,
            # which would lose new people and untracked parked-vehicle protection.
            ids = tracker.update(boxes, image)
            objects, masks = [], []
            if len(boxes) and result.masks is None:
                raise RuntimeError("Detector returned boxes without instance masks")
            for j in range(len(boxes)):
                tid = ids.get(j, missing_id)
                if tid < 0:
                    missing_id -= 1
                mask = result.masks.data[j].cpu().numpy()
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
                objects.append({"id": tid, "class": int(boxes.cls[j].item()),
                                "box": boxes.xyxy[j].cpu().tolist(), "confidence": float(boxes.conf[j].item()),
                                "rider_candidates": [], "rider_matches": []})
                masks.append(mask)
            seeds = np.stack(masks) if masks else np.zeros((0, *shape), dtype=bool)
            camera = motion_evidence(history[0], gray, objects, seeds, args) if len(history) == args.motion_lag else "warmup"
            camera_counts[camera] += 1
            rider_candidates(objects)
            for obj in objects:
                tid = str(obj["id"])
                track = tracks.setdefault(tid, {"classes": set(), "first": path.name, "last": path.name,
                                               "frames": 0, "valid": set(), "moving": set(),
                                               "riders": defaultdict(set), "candidate": False,
                                               "motion_sources": Counter()})
                track["classes"].add(obj["class"])
                track["last"] = path.name
                track["frames"] += 1
                if obj.get("motion"):
                    track["motion_sources"][obj["motion"]["source"]] += 1
                    if not track["valid"] or index - max(track["valid"]) >= args.motion_lag:
                        track["valid"].add(index)
                        if obj["motion"]["moving"]:
                            track["moving"].add(index)
                # Pair evidence has its own sampling schedule: a good rider match
                # must not be discarded because motion was sampled one frame earlier.
                for rider in obj["rider_matches"]:
                    if rider >= 0:
                        add_spaced_vote(track["riders"][rider], index, args.motion_lag)
                track["candidate"] |= bool(obj["rider_candidates"])
            seed_file = f"seeds/{index:06d}.npz"
            np.savez_compressed(args.output / seed_file, packed=np.packbits(seeds, axis=-1))
            stat = path.stat()
            record = {"index": args.start + index, "source": str(path.resolve()), "name": path.name,
                      "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "shape": list(shape),
                      "seeds": seed_file, "camera": camera, "objects": objects}
            manifest.write(json.dumps(record) + "\n")
            history.append((gray, objects, seeds))
            if args.overlay_every and index % args.overlay_every == 0:
                save_image(args.output / "overlays" / (path.stem + ".png"), draw_objects(image, objects))
            if index % 20 == 0 or index == len(files)-1:
                print(f"Analyze {index+1}/{len(files)}: {path.name}, camera={camera}", flush=True)
    decisions = summarize_tracks(tracks, args)
    write_json(args.output / "tracks.json", decisions)
    write_json(args.output / "overrides.example.json", {"tracks": {}, "frames": {}})
    write_json(args.output / "summary.json", {"frames": len(files), "tracks": len(tracks), "decision_version": DECISION_VERSION,
                                               "camera": dict(camera_counts), "complete": True})


def read_overrides(path, tracks, records):
    data = json.loads(path.read_text()) if path else {"tracks": {}, "frames": {}}
    if set(data) - {"tracks", "frames"}:
        raise ValueError("Overrides accepts only tracks and frames")
    present = {r["name"]: {str(obj["id"]) for obj in r["objects"]} for r in records}
    groups = [(data.get("tracks", {}), set(tracks))]
    for name, group in data.get("frames", {}).items():
        if name not in present:
            raise ValueError(f"Unknown override frame: {name}")
        groups.append((group, present[name]))
    for group, valid in groups:
        for tid, action in group.items():
            if tid not in valid or action not in {"keep", "remove"}:
                raise ValueError(f"Invalid override: {tid}={action}")
            if tracks[tid]["classes"] == [0] and action == "keep":
                raise ValueError("All people must be removed; cannot keep a person track")
    return data


def refine(predictor, obj, seed, negative, min_score):
    """Use a YOLO interior point, box and other-instance negative points."""
    if not seed.any():
        return seed, "empty_yolo_seed"
    distance = cv2.distanceTransform(seed.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(distance.argmax(), seed.shape)
    points = [[x, y]] + negative
    labels = [1] + [0] * len(negative)
    masks, scores, _ = predictor.predict(box=np.asarray(obj["box"], dtype=np.float32),
                                        point_coords=np.asarray(points, dtype=np.float32),
                                        point_labels=np.asarray(labels), multimask_output=False)
    mask = masks[0].astype(bool)
    # A SAM prompt may otherwise absorb an adjacent parked vehicle or rider.
    h, w = seed.shape
    x1, y1, x2, y2 = obj["box"]
    bounds = np.zeros_like(seed)
    bounds[max(0, int(y1)-2):min(h, int(y2)+3), max(0, int(x1)-2):min(w, int(x2)+3)] = True
    mask &= bounds
    intersection = np.count_nonzero(mask & seed)
    if (not np.isfinite(scores[0]) or scores[0] < min_score or intersection < 0.5 * seed.sum()
            or mask.sum() > 2.5 * seed.sum() or not mask.any()):
        return seed, "yolo_fallback"
    return mask, "sam2"


def render(args):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    summary = json.loads((args.analysis / "summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Analysis is incomplete")
    tracks = json.loads((args.analysis / "tracks.json").read_text())
    records = [json.loads(line) for line in (args.analysis / "frames.jsonl").read_text().splitlines()]
    if len(records) != summary["frames"]:
        raise ValueError("Analysis frame count mismatch")
    overrides = read_overrides(args.overrides, tracks, records)
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(args.sam_checkpoint)
    for record in records:
        stat = Path(record["source"]).stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f'Source changed since analysis: {record["source"]}')
        if not (args.analysis / record["seeds"]).is_file():
            raise FileNotFoundError(record["seeds"])
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device)
    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint), device=str(device)))
    fresh_directory(args.output)
    for name in ("masks", "raw_masks", "colmap_masks", "overlays"):
        (args.output / name).mkdir()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(overrides_applied=overrides, temporal_mask_propagation=False)
    write_json(args.output / "config.json", config)
    write_json(args.output / "tracks.json", tracks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate * 2 + 1,) * 2)
    counts = Counter()
    with torch.inference_mode(), (args.output / "manifest.jsonl").open("w") as manifest:
        for index, record in enumerate(records):
            image = cv2.imread(record["source"])
            if image is None or list(image.shape[:2]) != record["shape"]:
                raise ValueError(f'Invalid source: {record["source"]}')
            h, w = record["shape"]
            with np.load(args.analysis / record["seeds"], allow_pickle=False) as archive:
                seeds = np.unpackbits(archive["packed"], axis=-1, count=w).astype(bool)
            objects = record["objects"]
            if seeds.shape != (len(objects), h, w):
                raise ValueError("Corrupt seed cache")
            decisions = {}
            for obj in objects:
                tid = str(obj["id"])
                action = overrides.get("frames", {}).get(record["name"], {}).get(tid,
                         overrides.get("tracks", {}).get(tid, automatic_action(obj, tracks[tid])))
                if obj["class"] == 0:
                    action = "remove"  # Even when an ID switched between a vehicle and a person.
                decisions[tid] = {"action": action}
            remove = np.zeros((h, w), dtype=bool)
            protect = np.zeros_like(remove)
            person_seed = np.zeros_like(remove)
            for obj, seed in zip(objects, seeds):
                if obj["class"] == 0:
                    person_seed |= seed
            autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
            with autocast:
                if objects:
                    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                for obj, seed in zip(objects, seeds):
                    negative = []
                    for other, other_seed in zip(objects, seeds):
                        if other is obj or decisions[str(other["id"])]["action"] == decisions[str(obj["id"])]["action"]:
                            continue
                        region = other_seed & ~seed
                        if region.any():
                            dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
                            y, x = np.unravel_index(dist.argmax(), region.shape)
                            negative.append([int(x), int(y)])
                    mask, method = refine(predictor, obj, seed, negative, args.sam_min_score)
                    decisions[str(obj["id"])]["segmenter"] = method
                    counts[method] += 1
                    if decisions[str(obj["id"])]["action"] == "remove":
                        remove |= mask
                    else:
                        protect |= mask | seed
            # Keep visible person pixels; protected vehicle pixels win elsewhere, including dilation.
            protect &= ~person_seed
            remove &= ~protect
            raw = remove.astype(np.uint8) * 255
            removal = cv2.dilate(raw, kernel) if args.dilate else raw.copy()
            removal[protect] = 0
            name = Path(record["name"]).stem + ".png"
            save_image(args.output / "raw_masks" / name, raw)
            save_image(args.output / "masks" / name, removal)
            save_image(args.output / "colmap_masks" / (record["name"] + ".png"), 255-removal)
            if args.overlay_every and index % args.overlay_every == 0:
                overlay = image.copy()
                active = removal > 0
                overlay[active] = (0.5 * overlay[active] + np.array([0, 0, 127.5])).astype(np.uint8)
                save_image(args.output / "overlays" / name, draw_objects(overlay, objects, decisions))
            manifest.write(json.dumps({"source": record["source"], "mask": "masks/"+name,
                                       "decisions": decisions, "masked_fraction": float((removal > 0).mean())}) + "\n")
            counts["frames"] += 1
            if index % 20 == 0 or index == len(records)-1:
                print(f'Render {index+1}/{len(records)}: {record["name"]}', flush=True)
    write_json(args.output / "summary.json", dict(counts))


def main():
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    global cv2, np
    import cv2
    import numpy as np
    (analyze if args.stage == "analyze" else render)(args)


if __name__ == "__main__":
    main()
