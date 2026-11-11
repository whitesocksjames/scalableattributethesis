"""Verify self-contained loading and optional original/export physical parity."""
import argparse
import gc
import json
import os
from pathlib import Path
import tempfile

from scalable_attribute.runtime.checkpoints import POINTS, load_model, read_checkpoint


def reference_model(manifest, source_root, point, device):
    """Restore the original stages in their original order as a reference."""
    from scalable_attribute.export import load_source
    from scalable_attribute.models.base import CanonicalBaseModel
    from scalable_attribute.models.config import BaseSynthesisConfig
    from scalable_attribute.models.scalable import CanonicalScalableModel
    op = next(x for x in manifest["operating_points"] if x["point"] == point)
    released = manifest["released_checkpoints"][op["released_profile"]]
    # Check frozen identities before model construction, including the 4k parent.
    load_source(source_root, released)
    base_record = op.get("bootstrap_checkpoint", op["base_checkpoint"]) if point == "4k" else op["base_checkpoint"]
    base_state = load_source(source_root, base_record)
    full_state = load_source(source_root, op["full_checkpoint"])
    base = CanonicalBaseModel(str(source_root / released["path"]),
                              BaseSynthesisConfig(**base_state["config"]))
    if base_state["architecture"] == "canonical_base_predict_correct":
        base.base_synthesis.load_state_dict(base_state["base_synthesis"], strict=True)
    else:
        base.load_state_dict(base_state["base_model"], strict=True)
    base.freeze()
    model = CanonicalScalableModel(base, op["lambda"])
    if point == "4k":
        model.load_state_dict(full_state["scalable_model"], strict=True)
    else:
        model.enhancement.vae.load_state_dict(full_state["enhancement_vae"], strict=True)
    return model.to(device).requires_grad_(False).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--point", choices=list(POINTS) + ["all"], default="all")
    parser.add_argument("--load-only", action="store_true", help="CPU strict-state loading only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--input", type=Path, default=Path("examples/example.ply"))
    parser.add_argument("--gpcc", type=Path, default=Path("third_party/tmc3_v21"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if args.output.exists():
        parser.error("Refusing to overwrite validation evidence")
    if not args.load_only and (not args.source_manifest or not args.source_root):
        parser.error("Physical parity requires --source-manifest and --source-root")
    if not args.load_only and args.device == "cuda" and not torch.cuda.is_available():
        args.output.write_text(json.dumps({"status": "NOT_RUN", "reason": "CUDA unavailable"}, indent=2) + "\n")
        return 2
    points = list(POINTS) if args.point == "all" else [args.point]
    device = "cpu" if args.load_only else args.device
    rows = []
    for point in points:
        path = args.models / ("ours_" + point + ".pth")
        state = read_checkpoint(path)
        model = load_model(path, device=device)
        actual = model.state_dict()
        if set(actual) != set(state["model_state"]):
            raise RuntimeError("Strict load state keys differ")
        for key, tensor in state["model_state"].items():
            if actual[key].dtype != tensor.dtype or not torch.equal(actual[key].cpu(), tensor):
                raise RuntimeError("Strict load changed tensor: " + key)
        row = {"point": point, "strict_independent_load": "PASS", "tensors": len(actual)}
        if not args.load_only:
            from scalable_attribute.evaluate import read_rgb_ply
            from scalable_attribute.runtime.codec import (
                encode_base, decode_base, encode_full, decode_full, assert_sparse_equal)
            import MinkowskiEngine as ME
            manifest = json.loads(args.source_manifest.read_text())
            reference = reference_model(manifest, args.source_root, point, device)
            attribute = read_rgb_ply(args.input, device)
            gpcc = args.gpcc.resolve()
            previous = Path.cwd()
            with tempfile.TemporaryDirectory(prefix="ours-parity-") as directory:
                Path(directory, "tmc3_v21").symlink_to(gpcc)
                try:
                    os.chdir(directory)
                    old = reference.hard_reconstruct(attribute)
                    payload, base = encode_base(model, attribute)
                    enhancement = encode_full(model, attribute, base)
                finally:
                    os.chdir(previous)
            geometry = ME.SparseTensor(features=torch.zeros_like(attribute.F),
                                       coordinates=attribute.C, device=device)
            independent = decode_base(model, payload, geometry)
            full = decode_full(model, independent, enhancement["enhancement_payload"])
            for left, right, label in ((old["Base"], base["Base"], "original/export Base"),
                                       (old["Full"], full["x_out"], "original/export Full"),
                                       (base["Base"], independent["Base"], "independent Base decode")):
                assert_sparse_equal(left, right, label)
            bits = payload["bits"]["base"]
            full_bits = bits + enhancement["enhancement_bits"]
            if bits != old["base_bits"] or full_bits != old["full_bits"]:
                raise RuntimeError("Original/export physical bits differ")
            if enhancement["enhancement_payload"]["strings"] != old["enhancement_payload"]["strings"]:
                raise RuntimeError("Original/export EL bytes differ")
            row.update(physical_equivalence="PASS", base_independent_decode="PASS",
                       full_reuses_base="PASS", base_bits=bits, full_bits=full_bits)
            del reference, attribute, old, base, independent, enhancement, full
        rows.append(row)
        from scalable_attribute.export import sha256
        args.output.write_text(json.dumps({"status": "PASS" if len(rows) == len(points) else "PARTIAL",
                                          "mode": "load-only" if args.load_only else "physical",
                                          "device": device,
                                          "input_sha256": None if args.load_only else sha256(args.input),
                                          "models": rows}, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del model, state, actual
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
