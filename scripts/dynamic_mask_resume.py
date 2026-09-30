"""Validate a render run and reuse its complete manifest prefix."""

from collections import Counter
import json
from pathlib import Path
import struct


def complete_png(path, shape):
    """Check dimensions and the final PNG chunk without decoding thousands of previews."""
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                return False
            width, height = struct.unpack(">II", header[16:24])
            if [height, width] != shape:
                return False
            stream.seek(-12, 2)
            return stream.read() == b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
    except (OSError, ValueError):
        return False


def comparable(config):
    config = dict(config)
    config.pop("resume", None)
    for key in ("analysis", "output", "sam_checkpoint", "overrides", "carried_yolo", "carried_world", "carried_prompts"):
        if config.get(key) is not None:
            config[key] = str(Path(config[key]).resolve())
    # Old v3 renders had neither recovery nor carried-object inference.
    config.setdefault("no_rider_recovery", True)
    config.setdefault("no_carried_items", True)
    if config["no_rider_recovery"] and config["no_carried_items"]:
        config.pop("refinement_version", None)
    if config["no_carried_items"]:
        for key in ("carried_yolo", "carried_conf", "carried_world", "carried_prompts", "carried_prompts_applied"):
            config.pop(key, None)
    return config


def completed_prefix(output, records, config, tracks):
    """Read only. Return committed frames and the byte offset safe to append from."""
    saved = json.loads((output / "config.json").read_text())
    if comparable(saved) != comparable(config):
        raise ValueError("Resume settings changed; use the original flags or a new output directory")
    if json.loads((output / "tracks.json").read_text()) != tracks:
        raise ValueError("Analysis tracks changed; use a new output directory")
    complete, offset = [], 0
    manifest = output / "manifest.jsonl"
    if not manifest.exists():
        return complete, offset
    with manifest.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                break  # Only newline-terminated records are committed.
            try:
                entry = json.loads(line)
            except (ValueError, UnicodeError):
                if stream.read().strip():
                    raise ValueError("Corrupt manifest before its final line")
                break
            if index >= len(records):
                raise ValueError("Manifest has more frames than the analysis")
            record = records[index]
            name = Path(record["name"]).stem + ".png"
            if entry.get("source") != record["source"] or entry.get("mask") != "masks/" + name:
                raise ValueError("Manifest frame order differs from analysis")
            paths = [output / sub / name for sub in ("raw_masks", "masks")]
            paths.append(output / "colmap_masks" / (record["name"] + ".png"))
            if config["overlay_every"] and index % config["overlay_every"] == 0:
                paths.append(output / "overlays" / name)
            if not all(complete_png(path, record["shape"]) for path in paths):
                break  # Recompute from the first missing/incomplete frame, including any later files.
            complete.append(entry)
            offset = stream.tell()
            if len(complete) % 500 == 0:
                print(f'Validated {len(complete)} completed render frames...', flush=True)
    return complete, offset


def frame_counts(entry):
    counts = Counter(frames=1)
    for d in entry["decisions"].values():
        if "segmenter" in d:
            counts[d["segmenter"]] += 1
        for key, field in (("temporal_removals", "temporal"), ("rider_recoveries", "rider_recovery"),
                           ("duplicate_removals", "duplicate_of"), ("carried_items", "carried_by")):
            counts[key] += int(field in d)
    return counts
