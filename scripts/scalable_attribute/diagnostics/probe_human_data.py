#!/usr/bin/env python3
"""No-update compatibility probe for partitioned RGB human point clouds."""

import argparse
import json
import os

import MinkowskiEngine as ME
import torch

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import make_data_loader
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import UncachedPCDataset, h5_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument("--enhancement-checkpoint", required=True)
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    files = h5_files(args.data_root, args.file_list)
    dataset = UncachedPCDataset(files, color_format="yuv", normalize=True)
    loader = make_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0)
    base_state = torch.load(args.base_synthesis_checkpoint, map_location="cpu")
    base = CanonicalBaseModel(
        args.released_checkpoint,
        BaseSynthesisConfig(**base_state["config"])).cuda()
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)
    model = CanonicalScalableModel(base, args.conditioning_lambda).cuda()
    enhancement_state = torch.load(
        args.enhancement_checkpoint, map_location="cpu")
    model.enhancement.vae.load_state_dict(
        enhancement_state["enhancement_vae"], strict=True)
    model.eval()
    model.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for index, (coords, feats) in enumerate(loader):
            attribute = ME.SparseTensor(
                features=feats, coordinates=coords, tensor_stride=1,
                device="cuda")
            result = model(attribute)
            likelihood = result["likelihood_E"]
            rate = get_bits(likelihood) / len(attribute)
            mse = torch.mean((attribute.F - result["Full"].F) ** 2)
            if not torch.isfinite(rate) or not torch.isfinite(mse):
                raise RuntimeError("Non-finite human-data forward")
            rows.append({
                "block": index, "points": len(attribute),
                "coord_min": attribute.C[:, 1:].min(dim=0).values.tolist(),
                "coord_max": attribute.C[:, 1:].max(dim=0).values.tolist(),
                "rgb_yuv_min": attribute.F.min(dim=0).values.tolist(),
                "rgb_yuv_max": attribute.F.max(dim=0).values.tolist(),
                "estimated_enhancement_bpp": float(rate.item()),
                "full_mse": float(mse.item()),
            })
    summary = {
        "status": "PASS", "optimizer_steps": 0,
        "num_h5": len(files), "total_points": sum(r["points"] for r in rows),
        "base_prefix_forward": True, "enhancement_forward": True,
        "base_frozen": not any(p.requires_grad for p in model.base.parameters()),
        "finite": True, "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
