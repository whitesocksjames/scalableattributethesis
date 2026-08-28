#!/usr/bin/env python3
"""Static/runtime gate for the single zero-centered synthesis experiment."""

import argparse
import json
import os

import MinkowskiEngine as ME
import numpy as np
import torch

from data_utils.attribute.color_format import rgb2yuv
from data_utils.attribute.inout import read_h5
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.model import ScalableAttributeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-lambda", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample(path):
    coords, rgb = read_h5(os.path.expandvars(path))
    yuv = rgb2yuv(rgb.astype("float32"), out_range=1).astype("float32")
    coords, feats = ME.utils.sparse_collate([coords], [yuv])
    return ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1, device="cuda")


def make_model(base_checkpoint, zero_centered):
    config = EnhancementConfig(
        synthesis_condition="b_fu",
        zero_centered_synthesis=zero_centered,
    )
    return ScalableAttributeModel(
        os.path.expandvars(base_checkpoint), config).cuda().eval()


@torch.no_grad()
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    A = sample(args.h5)

    original = make_model(args.base_checkpoint, zero_centered=False)
    original.train()
    original_B, _ = original.base_adapter(A, args.base_lambda)
    original_B_F = original_B.F.detach().clone()
    original_B_C = original_B.C.detach().clone()
    del original
    torch.cuda.empty_cache()

    model = make_model(args.base_checkpoint, zero_centered=True)
    model.train()
    B, F_U = model.base_adapter(A, args.base_lambda)
    base_max_abs = float((B.F - original_B_F).abs().max().item())
    base_coordinates_equal = bool(torch.equal(B.C, original_B_C))

    model.eval()
    layer = model.enhancement
    y_E = layer._analysis(A, B, F_U)
    zero_y = ME.SparseTensor(
        features=torch.zeros_like(y_E.F),
        coordinate_map_key=y_E.coordinate_map_key,
        coordinate_manager=y_E.coordinate_manager,
        device=y_E.device,
    )
    _, Full_zero = layer._synthesis(zero_y, B, F_U)
    zero_max_abs = float((Full_zero.F - B.F).abs().max().item())
    zero_support_equal = bool(
        zero_y.coordinate_map_key == y_E.coordinate_map_key
        and zero_y.coordinate_manager == y_E.coordinate_manager
        and list(zero_y.tensor_stride) == list(y_E.tensor_stride)
        and torch.equal(zero_y.C, y_E.C))

    deterministic = layer(A, B, F_U)["Full"]
    encoded, hard_encoder = layer.hard_encode(A, B, F_U)
    hard_decoder = layer.hard_decode(encoded, B, F_U)
    deterministic_hard_max_abs = float(
        (deterministic.F - hard_encoder.F).abs().max().item())
    hard_roundtrip_max_abs = float(
        (hard_encoder.F - hard_decoder.F).abs().max().item())

    frozen = (
        not model.base_adapter.training
        and not model.base_adapter.base.training
        and all(not parameter.requires_grad
                for parameter in model.base_adapter.parameters())
        and not B.F.requires_grad
        and not F_U.F.requires_grad
    )
    result = {
        "h5": os.path.expandvars(args.h5),
        "base_checkpoint": os.path.expandvars(args.base_checkpoint),
        "base_lambda": args.base_lambda,
        "zero_symbols_full_vs_base_max_abs": zero_max_abs,
        "zero_sparse_support_identical": zero_support_equal,
        "deterministic_vs_hard_encoder_max_abs": deterministic_hard_max_abs,
        "hard_encoder_vs_decoder_max_abs": hard_roundtrip_max_abs,
        "base_before_vs_after_max_abs": base_max_abs,
        "base_coordinates_equal": base_coordinates_equal,
        "base_frozen": frozen,
    }
    passed = (
        zero_max_abs <= 1e-7
        and zero_support_equal
        and deterministic_hard_max_abs <= 1e-7
        and hard_roundtrip_max_abs <= 1e-7
        and base_max_abs == 0.0
        and base_coordinates_equal
        and frozen
    )
    result["pass"] = passed
    output = os.path.abspath(os.path.expandvars(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("zero-centered synthesis static gate failed")


if __name__ == "__main__":
    main()
