#!/usr/bin/env python3
"""Adapt the local official inference entry point without editing its checkout.

Only three explicit AST locations are supported: propagation length, the end of
image propagation, and final frame export. Refuse incompatible source layouts
before loading models. The diagnostic hooks do not modify inference tensors.
"""

import argparse
import ast
import json
from pathlib import Path
import sys


class InferenceAdapter(ast.NodeTransformer):
    def __init__(self, propagation_length):
        self.propagation_length = propagation_length
        self.matches = {"length": 0, "propagated": 0, "final": 0}

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        name = node.targets[0].id
        if name == "subvideo_length_img_prop":
            expected = ast.parse("min(100, args.subvideo_length)", mode="eval").body
            if ast.dump(node.value) != ast.dump(expected):
                raise ValueError("Unsupported ProPainter image-propagation length expression")
            self.matches["length"] += 1
            if self.propagation_length is not None:
                node.value = ast.parse(f"min({self.propagation_length}, args.subvideo_length)", mode="eval").body
        if name == "ori_frames":
            if not isinstance(node.value, ast.Name) or node.value.id != "frames_inp":
                raise ValueError("Unsupported ProPainter propagation export location")
            self.matches["propagated"] += 1
            hook = ast.parse("_diagnostics.propagated(frames_inp, flow_masks, masks_dilated, "
                             "updated_frames, updated_masks, gt_flows_bi, pred_flows_bi, "
                             "subvideo_length_img_prop)").body[0]
            return [hook, node]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if ast.dump(node.test) == ast.dump(ast.parse("args.save_frames", mode="eval").body):
            self.matches["final"] += 1
            hook = ast.parse("_diagnostics.finished(comp_frames)").body[0]
            return [hook, node]
        return node


def instrument_source(source, filename, propagation_length):
    tree = ast.parse(source, filename=filename)
    adapter = InferenceAdapter(propagation_length)
    tree = adapter.visit(tree)
    if adapter.matches != {"length": 1, "propagated": 1, "final": 1}:
        raise ValueError(f"Unsupported ProPainter source layout: {adapter.matches}")
    return compile(ast.fix_missing_locations(tree), filename, "exec")


class Diagnostics:
    def __init__(self, spec):
        self.spec = spec

    def propagated(self, originals, flow_masks, masks, propagated, remaining,
                   raw_flows, completed_flows, propagation_length):
        if not self.spec or not self.spec["frames"]:
            return
        import cv2
        import numpy as np

        def save(directory, name, array):
            if not cv2.imwrite(str(directory / name), array):
                raise OSError(f"Cannot write diagnostic image: {directory / name}")

        count = len(originals)
        for key, record in self.spec["frames"].items():
            index = int(key)
            directory = Path(self.spec["output"]) / Path(record["name"]).stem
            directory.mkdir(parents=True, exist_ok=True)
            mask = masks[0, index, 0].detach().float().cpu().numpy() > 0.5
            unknown = remaining[0, index, 0].detach().float().cpu().numpy() > 0.5
            flow_mask = flow_masks[0, index, 0].detach().float().cpu().numpy() > 0.5
            original = cv2.cvtColor(originals[index], cv2.COLOR_RGB2BGR)
            rgb = propagated[0, index].detach().float().cpu().permute(1, 2, 0).numpy()
            rgb = np.clip((rgb + 1) * 127.5, 0, 255).astype(np.uint8)
            overlay = original.copy()
            overlay[mask] = (0.5 * original[mask] + np.array([0, 0, 127.5])).astype(np.uint8)
            save(directory, "original.png", original)
            save(directory, "inference_mask.png", mask.astype(np.uint8) * 255)
            save(directory, "flow_mask.png", flow_mask.astype(np.uint8) * 255)
            save(directory, "mask_overlay.png", overlay)
            save(directory, "propagated.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            save(directory, "remaining_mask.png", unknown.astype(np.uint8) * 255)
            save(directory, "propagated_valid.png", (mask & ~unknown).astype(np.uint8) * 255)
            # Save only pairs adjacent to the selected frame, not the whole clip.
            fields = {}
            for pair_index, label in ((index-1, "previous_pair"), (index, "next_pair")):
                if 0 <= pair_index < count-1:
                    for stage, flows in (("raw", raw_flows), ("completed", completed_flows)):
                        for direction, flow in zip(("forward", "backward"), flows):
                            fields[f"{stage}_{label}_{direction}"] = flow[0, pair_index].detach().float().cpu().numpy()
            np.savez_compressed(directory / "flows.npz", **fields)
            if count > propagation_length:
                start = index // propagation_length * propagation_length
                context = [max(0, start-10), min(count, start+propagation_length+10)]
            else:
                context = [0, count]
            metadata = dict(record, local_index=index, propagation_length=propagation_length,
                            propagation_context_local_half_open=context,
                            remaining_fraction_inside_mask=float((unknown & mask).sum() / max(1, mask.sum())))
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def finished(self, frames):
        if not self.spec or not self.spec["frames"]:
            return
        import cv2
        import numpy as np

        for key, record in self.spec["frames"].items():
            directory = Path(self.spec["output"]) / Path(record["name"]).stem
            final = cv2.cvtColor(frames[int(key)], cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(directory / "model_output.png"), final):
                raise OSError(f"Cannot write diagnostic output: {directory}")
            panels = []
            for name, label in (("original", "Original"), ("mask_overlay", "Actual inference mask"),
                                ("propagated", "After image propagation"), ("remaining_mask", "White = still unknown"),
                                ("propagated_valid", "White = propagation accepted"), ("model_output", "After transformer")):
                panel = cv2.imread(str(directory / (name + ".png")))
                if panel is None:
                    raise ValueError(f"Missing diagnostic panel: {directory / name}")
                panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(panel, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(panel)
            comparison = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
            if not cv2.imwrite(str(directory / "comparison.png"), comparison):
                raise OSError(f"Cannot write diagnostic comparison: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--propagation-length", type=int)
    parser.add_argument("--debug-spec", type=Path)
    args, inference_args = parser.parse_known_args()
    if args.propagation_length is not None and args.propagation_length < 2:
        parser.error("propagation-length must be at least 2")
    if inference_args[:1] == ["--"]:
        inference_args = inference_args[1:]
    entry = args.repo.resolve() / "inference_propainter.py"
    code = instrument_source(entry.read_text(), str(entry), args.propagation_length)
    spec = json.loads(args.debug_spec.read_text()) if args.debug_spec else None
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *inference_args]
    namespace = {"__name__": "__main__", "__file__": str(entry), "_diagnostics": Diagnostics(spec)}
    exec(code, namespace)


if __name__ == "__main__":
    main()
