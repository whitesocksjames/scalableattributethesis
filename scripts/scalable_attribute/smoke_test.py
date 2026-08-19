#!/usr/bin/env python3
"""One real-RWTT core smoke test for the scalable Attribute prototype."""

import json
import os

import torch
import MinkowskiEngine as ME

from data_utils.dataloaders.attribute_dataloader import PCDataset, make_data_loader
from scalable_attribute.coder import ScalableAttributeCoder
from scalable_attribute.config import EnhancementConfig
from scalable_attribute.losses import rate_distortion_loss
from scalable_attribute.model import ScalableAttributeModel


def sparse_difference(lhs, rhs):
    if not torch.equal(lhs.C, rhs.C):
        raise RuntimeError("Smoke comparison coordinates differ")
    return float((lhs.F - rhs.F).abs().max().item())


def main():
    data_root = os.environ["SCALABLE_DATA_ROOT"]
    checkpoint = os.environ["SCALABLE_BASE_CHECKPOINT"]
    output_path = os.environ["SCALABLE_SMOKE_OUTPUT"]
    base_lambda = int(os.environ.get("SCALABLE_BASE_LAMBDA", "2048"))
    rd_lambda = float(os.environ.get("SCALABLE_RD_LAMBDA", "1000"))

    files = []
    for root, _, names in os.walk(data_root):
        files.extend(os.path.join(root, name) for name in names if name.endswith(".h5"))
        if files:
            break
    files.sort()
    dataset = PCDataset(files[:1], color_format="yuv", normalize=True)
    coords, feats = next(iter(make_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0)))
    A = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device="cuda")

    config = EnhancementConfig()
    model = ScalableAttributeModel(checkpoint, config).cuda()
    model.train()
    output = model(A, base_lambda)
    loss, rate, distortion = rate_distortion_loss(
        A, output["Full"], output["likelihood"], rd_lambda)
    loss.backward()

    base_grad_count = sum(parameter.grad is not None for parameter in model.base_adapter.parameters())
    enhancement_gradients = [
        parameter.grad for parameter in model.enhancement.parameters() if parameter.grad is not None]
    enhancement_gradients_finite = bool(enhancement_gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in enhancement_gradients)

    shapes = {
        "A": [list(A.F.shape), list(A.tensor_stride)],
        "B": [list(output["B"].F.shape), list(output["B"].tensor_stride)],
        "F_U": [list(output["F_U"].F.shape), list(output["F_U"].tensor_stride)],
        "y_E": [list(output["y_E"].F.shape), list(output["y_E"].tensor_stride)],
        "y_hat": [list(output["y_hat"].F.shape), list(output["y_hat"].tensor_stride)],
        "delta_A": [list(output["delta_A"].F.shape), list(output["delta_A"].tensor_stride)],
        "Full": [list(output["Full"].F.shape), list(output["Full"].tensor_stride)],
    }

    del output
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        estimated = model(A, base_lambda)
        hard = ScalableAttributeCoder(model).test(A, base_lambda)

    report = {
        "sample": files[0],
        "config": config.to_dict(),
        "shapes": shapes,
        "loss": float(loss.detach().cpu()),
        "estimated_R_E": float(rate.detach().cpu()),
        "distortion": float(distortion.detach().cpu()),
        "base_grad_count": base_grad_count,
        "enhancement_gradients_finite": enhancement_gradients_finite,
        "estimated_vs_hard_B_max_abs": sparse_difference(estimated["B"], hard["B"]),
        "estimated_vs_hard_F_U_max_abs": sparse_difference(estimated["F_U"], hard["F_U"]),
        "estimated_vs_hard_Full_max_abs": sparse_difference(estimated["Full"], hard["Full"]),
        "hard_max_abs_difference": hard["hard_max_abs_difference"],
        "base_bits": hard["base_bits"],
        "el_bits": hard["el_bits"],
        "full_bits": hard["full_bits"],
        "R_base": hard["R_base"],
        "R_E": hard["R_E"],
        "R_full": hard["R_full"],
    }
    report["pass"] = (
        base_grad_count == 0
        and enhancement_gradients_finite
        and report["estimated_vs_hard_B_max_abs"] == 0
        and report["estimated_vs_hard_F_U_max_abs"] == 0
        and report["estimated_vs_hard_Full_max_abs"] == 0
        and report["hard_max_abs_difference"] == 0
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError("Scalable Attribute core smoke failed")


if __name__ == "__main__":
    main()
