#!/usr/bin/env python3
"""One-time real-RWTT coordinate gate for the approved EL DSx2/USx2 topology."""

import json
import os

import numpy as np
import torch
import MinkowskiEngine as ME

from basic_models.backbone import Backbone
from data_utils.dataloaders.attribute_dataloader import PCDataset, make_data_loader
from lossy_attribute.model import MultiscaleVAE


DATA_ROOT = os.environ["UNICORN_PROBE_DATA_ROOT"]
CHECKPOINT = os.environ["UNICORN_PROBE_CHECKPOINT"]
OUTPUT = os.environ.get("UNICORN_PROBE_OUTPUT", "probe_el_backbone_coordinates.json")
LMB = int(os.environ.get("UNICORN_PROBE_LMB", "2048"))


def coordinate_set(x):
    coords = x.C.detach().cpu().numpy()
    order = np.lexsort(tuple(coords[:, i] for i in range(coords.shape[1] - 1, -1, -1)))
    return coords[order]


def main():
    files = []
    for root, _, names in os.walk(DATA_ROOT):
        files.extend(os.path.join(root, n) for n in names if n.endswith(".h5"))
        if files:
            break
    files.sort()
    if not files:
        raise RuntimeError("No RWTT HDF5 found")

    dataset = PCDataset(files[:1], color_format="yuv", normalize=True)
    coords, feats = next(iter(make_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0)))
    A = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device="cuda")

    base = MultiscaleVAE().cuda().eval()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")["model"]
    if set(checkpoint) != set(base.state_dict()):
        raise RuntimeError("Released Base checkpoint does not exactly match the model")
    base.load_state_dict(checkpoint)
    base.requires_grad_(False)
    with torch.no_grad():
        base_out = base(A, training=False, lmb=LMB, real_coding=False)
        B = base_out["out_list"][0]
        down = Backbone(scale=2, in_channels=3, channels=128, out_channels=64).cuda().eval()
        up = Backbone(scale=-2, in_channels=64, channels=128, out_channels=128).cuda().eval()
        y_E = down(B)
        decoded = up(y_E)

    same_support = np.array_equal(coordinate_set(B), coordinate_set(decoded))
    report = {
        "sample": files[0],
        "B_points": len(B),
        "B_stride": list(B.tensor_stride),
        "y_E_points": len(y_E),
        "y_E_stride": list(y_E.tensor_stride),
        "decoded_points": len(decoded),
        "decoded_stride": list(decoded.tensor_stride),
        "decoded_matches_B_support": same_support,
        "pass": same_support and list(B.tensor_stride) == [1, 1, 1]
                and list(y_E.tensor_stride) == [4, 4, 4]
                and list(decoded.tensor_stride) == [1, 1, 1],
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError("EL DSx2/USx2 coordinate gate failed")


if __name__ == "__main__":
    main()
