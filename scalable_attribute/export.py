"""Lossless export from trusted frozen thesis states; no GPU required.

Supply the research manifest explicitly. It remains in the research archive.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scalable_attribute.runtime.checkpoints import FORMAT, POINTS


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(root, record):
    import torch
    path = (root / record["path"]).resolve()
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError("Frozen source identity mismatch: " + record["path"])
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assemble(operation, released, base, full):
    """Match the original strict loader order without changing a tensor."""
    point = operation["point"]
    if operation["lambda"] != POINTS[point]:
        raise ValueError("Frozen point/lambda mismatch")
    if operation["full_loader"] == "joint_scalable_full":
        if full.get("architecture") != "canonical_scalable_mvub_finetune_v1":
            raise ValueError("Joint architecture mismatch")
        if full.get("conditioning_lambda") != operation["lambda"]:
            raise ValueError("Joint lambda mismatch")
        return full["scalable_model"], full["base_config"]
    weights = {"base.prefix.model." + k: v for k, v in released["model"].items()}
    architecture = base.get("architecture")
    if base.get("base_lambda", base.get("conditioning_lambda")) != operation["lambda"]:
        raise ValueError("Base lambda mismatch")
    if architecture == "canonical_base_predict_correct":
        weights.update({"base.base_synthesis." + k: v
                        for k, v in base["base_synthesis"].items()})
    elif architecture == "canonical_base_rescue_v1":
        weights = {"base." + k: v for k, v in base["base_model"].items()}
    else:
        raise ValueError("Unsupported Base architecture")
    if full.get("architecture") != "canonical_independent_enhancement":
        raise ValueError("Unsupported EL architecture")
    if full.get("conditioning_lambda") != operation["lambda"]:
        raise ValueError("EL lambda mismatch")
    weights.update({"enhancement.vae." + k: v
                    for k, v in full["enhancement_vae"].items()})
    return weights, base["config"]


def export_point(manifest, root, output, point):
    import torch
    from scalable_attribute.runtime.checkpoints import read_checkpoint
    operation = next(p for p in manifest["operating_points"] if p["point"] == point)
    records = {"released": manifest["released_checkpoints"][operation["released_profile"]],
               "base": operation["base_checkpoint"], "full": operation["full_checkpoint"]}
    target = output / ("ours_" + point + ".pth")
    if target.exists():
        raise FileExistsError("Refusing to replace export: " + str(target))
    loaded, by_hash = {}, {}
    for role, record in records.items():
        if record["sha256"] not in by_hash:
            by_hash[record["sha256"]] = load_source(root, record)
        loaded[role] = by_hash[record["sha256"]]
    weights, config = assemble(operation, loaded["released"], loaded["base"], loaded["full"])
    state = {"format": FORMAT, "point": point,
             "conditioning_lambda": operation["lambda"],
             "prefix_config": {"scale": 5, "stage": 1, "vmode": 1},
             "base_config": config, "model_state": weights,
             "provenance": {"contract_id": manifest["contract_id"],
                            "released_profile": operation["released_profile"],
                            "source_sha256": {k: v["sha256"] for k, v in records.items()}}}
    output.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pth.tmp")
    if temporary.exists():
        raise FileExistsError("Preserve interrupted export: " + str(temporary))
    torch.save(state, temporary)
    actual = read_checkpoint(temporary)["model_state"]
    if set(weights) != set(actual):
        raise RuntimeError("Export changed state keys")
    for key, tensor in weights.items():
        other = actual[key]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise RuntimeError("Export changed tensor: " + key)
    temporary.rename(target)
    record = {"point": point, "lambda": operation["lambda"], "filename": target.name,
              "size_bytes": target.stat().st_size, "sha256": sha256(target),
              "tensor_count": len(weights), "tensor_equivalence": "PASS",
              "inference_equivalence": "NOT_RUN", "url": None,
              "source_sha256": state["provenance"]["source_sha256"]}
    (output / ("ours_" + point + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point", choices=list(POINTS), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    print(json.dumps(export_point(manifest, args.source_root, args.output, args.point)), flush=True)


if __name__ == "__main__":
    main()
