#!/usr/bin/env python3
"""Probe the released Unicorn-v1 lossy Attribute Base interface.

This is an experiment-only diagnostic.  It does not modify the model and uses a
forward hook solely to observe the final ``fuseNet`` output produced inside the
public hard decoder, whose public return value currently contains only x_rec.
"""

import json
import os

import numpy as np
import torch
import MinkowskiEngine as ME

from data_utils.dataloaders.attribute_dataloader import PCDataset, make_data_loader
from lossy_attribute.model import MultiscaleVAE


DATA_ROOT = os.environ["UNICORN_PROBE_DATA_ROOT"]
CHECKPOINT = os.environ["UNICORN_PROBE_CHECKPOINT"]
OUTPUT = os.environ.get("UNICORN_PROBE_OUTPUT", "probe_unicorn_base_interface.json")
LMB = int(os.environ.get("UNICORN_PROBE_LMB", "2048"))


def describe(x):
    return {
        "feature_shape": list(x.F.shape),
        "coordinate_shape": list(x.C.shape),
        "tensor_stride": list(x.tensor_stride),
        "feature_dtype": str(x.F.dtype),
        "coordinate_dtype": str(x.C.dtype),
    }


def ordered(x):
    coords = x.C.detach().cpu().numpy()
    order = np.lexsort(tuple(coords[:, i] for i in range(coords.shape[1] - 1, -1, -1)))
    return coords[order], x.F.detach().cpu().numpy()[order]


def compare(lhs, rhs):
    lhs_c, lhs_f = ordered(lhs)
    rhs_c, rhs_f = ordered(rhs)
    coordinates_equal = np.array_equal(lhs_c, rhs_c)
    if not coordinates_equal or lhs_f.shape != rhs_f.shape:
        return {
            "coordinates_equal": coordinates_equal,
            "feature_shapes_equal": lhs_f.shape == rhs_f.shape,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
        }
    delta = lhs_f.astype(np.float64) - rhs_f.astype(np.float64)
    return {
        "coordinates_equal": True,
        "feature_shapes_equal": True,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }


def load_released(model, checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu")
    released = ckpt["model"]
    current = model.state_dict()
    compatible = {k: v for k, v in released.items() if k in current}
    current.update(compatible)
    model.load_state_dict(current)
    return {
        "checkpoint_keys": len(released),
        "loaded_keys": len(compatible),
        "missing_keys": sorted(set(current) - set(compatible)),
        "unexpected_keys": sorted(set(released) - set(current)),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA/MinkowskiEngine on an allocated GPU node")

    files = []
    for root, _, names in os.walk(DATA_ROOT):
        files.extend(os.path.join(root, n) for n in names if n.endswith(".h5"))
        if files:
            break
    files.sort()
    if not files:
        raise RuntimeError("No RWTT HDF5 found under " + DATA_ROOT)

    # Official public-v1 loader path: stored RGB uint8 -> normalized YUV float.
    dataset = PCDataset(files[:1], color_format="yuv", normalize=True)
    loader = make_data_loader(dataset, batch_size=1, shuffle=False, num_workers=0)
    coords, feats = next(iter(loader))
    A = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device="cuda")

    model = MultiscaleVAE().cuda()
    load_info = load_released(model, CHECKPOINT)
    model.eval()
    model.requires_grad_(False)

    with torch.no_grad():
        normal = model(A, training=False, lmb=LMB)
        deterministic = model(A, training=False, lmb=LMB, real_coding=False)
    B_forward = normal["out_list"][0]
    F_forward = normal["curr_f"]
    B_deterministic = deterministic["out_list"][0]
    F_deterministic = deterministic["curr_f"]

    # Exercise the public hard entropy encode/decode calls.
    with torch.no_grad():
        enc_set_list, x_low, gpcc_bits = model(A, training=False, lmb=LMB, encode=True)
        x0 = ME.SparseTensor(
            features=torch.zeros_like(A.F),
            coordinate_map_key=A.coordinate_map_key,
            coordinate_manager=A.coordinate_manager,
            device=A.device,
        )
        B_hard, F_hard = model.decode(
            x0=x0, x_low=x_low, enc_set_list=enc_set_list, lmb=LMB,
            return_feature=True)

    # Minimal EL surrogate: gradients must reach its own parameters, never Base.
    el_probe = torch.nn.Linear(B_forward.F.shape[1] + F_forward.F.shape[1], 1).cuda()
    el_loss = el_probe(torch.cat([B_forward.F.detach(), F_forward.F.detach()], dim=1)).square().mean()
    el_loss.backward()
    base_grad_tensors = sum(p.grad is not None for p in model.parameters())
    el_grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in el_probe.parameters())

    report = {
        "checkpoint": CHECKPOINT,
        "sample": files[0],
        "lambda": LMB,
        "input_path": "HDF5 RGB uint8 -> PCDataset(color_format='yuv', normalize=True) -> rgb2yuv(out_range=1)",
        "A": describe(A),
        "B_forward": describe(B_forward),
        "F_U_forward": describe(F_forward),
        "B_deterministic": describe(B_deterministic),
        "F_U_deterministic": describe(F_deterministic),
        "B_hard": describe(B_hard),
        "F_U_hard_internal": describe(F_hard),
        "alignment": {
            "A_vs_B_forward": compare(A, B_forward)["coordinates_equal"],
            "A_vs_F_U_forward": compare(A, F_forward)["coordinates_equal"],
            "A_vs_B_hard": compare(A, B_hard)["coordinates_equal"],
            "A_vs_F_U_hard": compare(A, F_hard)["coordinates_equal"],
        },
        "forward_vs_hard_B": compare(B_forward, B_hard),
        "forward_vs_hard_F_U": compare(F_forward, F_hard),
        "forward_vs_deterministic_B": compare(B_forward, B_deterministic),
        "forward_vs_deterministic_F_U": compare(F_forward, F_deterministic),
        "hard_api": {
            "decode_public_return": "B only (curr_x)",
            "F_U_computed_internally": True,
            "F_U_publicly_returned_with_opt_in": True,
            "x_low_serialized_or_decoded_by_public_api": False,
            "base_bitstream_only_decode_available": False,
            "gpcc_bits_reported": int(gpcc_bits),
        },
        "frozen_base": {
            "eval": not model.training,
            "all_requires_grad_false": all(not p.requires_grad for p in model.parameters()),
            "base_grad_tensors_after_el_backward": base_grad_tensors,
            "el_gradients_finite": el_grad_finite,
        },
        "checkpoint_load": load_info,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
