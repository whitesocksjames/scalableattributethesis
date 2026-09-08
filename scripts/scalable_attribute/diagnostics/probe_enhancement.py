#!/usr/bin/env python3
"""E0-B correctness probe for the canonical independent EnhancementVAE."""

import argparse
import hashlib
import json
import os
import shlex
import sys

import MinkowskiEngine as ME
import numpy as np
import torch

from basic_models.loss import get_bits
from data_utils.dataloaders.attribute_dataloader import PCDataset
from scalable_attribute.canonical.config import BaseSynthesisConfig
from scalable_attribute.canonical.model import CanonicalBaseModel
from scalable_attribute.canonical.scalable_model import (
    CanonicalScalableModel, load_frozen_base)
from scalable_attribute.data import h5_files
from scalable_attribute.evaluation import psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--file-list", required=True)
    parser.add_argument("--released-checkpoint", required=True)
    parser.add_argument("--base-synthesis-checkpoint", required=True)
    parser.add_argument(
        "--require-base-architecture",
        choices=("canonical_base_predict_correct", "canonical_base_rescue_v1"))
    parser.add_argument("--require-base-rescue-step", type=int)
    parser.add_argument(
        "--require-base-rescue-sampling", choices=("uniform", "high_energy"))
    parser.add_argument(
        "--require-base-rescue-trainable-scope",
        choices=("base_path", "base_synthesis_only"))
    parser.add_argument("--conditioning-lambda", type=int, required=True)
    parser.add_argument("--tmc3-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sparse_difference(left, right, label):
    same_support(left, right, label)
    return float((left.F - right.F).abs().max().item())


def same_support(left, right, label):
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")


def quality(reference, reconstruction):
    difference = sparse_difference(reference, reconstruction, "quality")
    del difference
    mse = float(torch.mean((reference.F - reconstruction.F) ** 2).item())
    if not np.isfinite(mse):
        raise RuntimeError("Non-finite reconstruction MSE")
    return mse, psnr(mse)


def module_gradient_norm(module, label):
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(label + " has non-finite gradients")
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    if squared is None or squared.item() == 0:
        raise RuntimeError(label + " received no nonzero gradient")
    return float(squared.sqrt().item())


@torch.no_grad()
def native_full_anchor(prefix, attribute, conditioning_lambda):
    encoded, x_low, gpcc_bits = prefix.model(
        attribute, training=False, lmb=conditioning_lambda, encode=True)
    if len(encoded) != 5:
        raise RuntimeError("Native Full did not encode five residual streams")
    x0 = ME.SparseTensor(
        features=torch.zeros_like(attribute.F),
        coordinate_map_key=attribute.coordinate_map_key,
        coordinate_manager=attribute.coordinate_manager,
        device=attribute.device)
    reconstruction = prefix.model.decode(
        x0=x0, x_low=x_low, enc_set_list=encoded,
        lmb=conditioning_lambda)
    stream_bits = [int(len(item["strings"]) * 8) for item in encoded]
    mse, value_psnr = quality(attribute, reconstruction)
    return {
        "Full": reconstruction,
        "mse": mse,
        "psnr": value_psnr,
        "r5_bits": stream_bits[4],
        "residual_bits": stream_bits,
        "gpcc_bits": int(gpcc_bits),
        "total_bits": int(gpcc_bits + sum(stream_bits)),
    }


def parameter_independence(released, enhancement, expected_state=None):
    released_state = (released.state_dict()
                      if expected_state is None else expected_state)
    enhancement_state = enhancement.state_dict()
    if set(released_state) != set(enhancement_state):
        raise RuntimeError("Enhancement/released state_dict keys differ")
    max_difference = 0.0
    for name in released_state:
        difference = float((
            released_state[name].detach().cpu() -
            enhancement_state[name].detach().cpu()
        ).abs().max().item())
        max_difference = max(max_difference, difference)
    released_parameters = dict(released.named_parameters())
    enhancement_parameters = dict(enhancement.named_parameters())
    for name, parameter in released_parameters.items():
        candidate = enhancement_parameters[name]
        if id(parameter) == id(candidate):
            raise RuntimeError("Shared Parameter object at " + name)
        if parameter.data_ptr() == candidate.data_ptr():
            raise RuntimeError("Shared Parameter storage at " + name)
    return {
        "state_max_abs_difference": max_difference,
        "parameter_objects_distinct": True,
        "parameter_storage_distinct": True,
        "parameter_count": sum(
            parameter.numel() for parameter in enhancement.parameters()),
    }


def clone_state_dict_cpu(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_base_checkpoint_contract(state, args):
    architecture = state.get("architecture")
    if (args.require_base_architecture is not None and
            architecture != args.require_base_architecture):
        raise ValueError("Required Base checkpoint architecture mismatch")
    rescue_requirements = (
        args.require_base_rescue_step,
        args.require_base_rescue_sampling,
        args.require_base_rescue_trainable_scope,
    )
    if not any(value is not None for value in rescue_requirements):
        return
    if architecture != "canonical_base_rescue_v1":
        raise ValueError("Base-rescue requirements need a rescue checkpoint")
    if (args.require_base_rescue_step is not None and
            int(state.get("step", -1)) != args.require_base_rescue_step):
        raise ValueError("Required Base rescue step mismatch")
    if (args.require_base_rescue_trainable_scope is not None and
            state.get("trainable_scope") !=
            args.require_base_rescue_trainable_scope):
        raise ValueError("Required Base rescue trainable scope mismatch")
    if args.require_base_rescue_sampling is not None:
        sampling = state.get("sampling") or {}
        high_weight = float(sampling.get("high_weight", float("nan")))
        actual = "uniform" if high_weight == 1.0 else "high_energy"
        if actual != args.require_base_rescue_sampling:
            raise ValueError("Required Base rescue sampling mismatch")


def main():
    args = parse_args()
    if args.max_samples not in (1, 2):
        raise ValueError("E0-B max-samples must be 1 or 2")
    if os.path.exists(args.output_dir):
        raise FileExistsError("E0-B output directory already exists")
    os.makedirs(args.output_dir)
    if not os.path.isfile(args.tmc3_path) or not os.access(
            args.tmc3_path, os.X_OK):
        raise FileNotFoundError("Executable tmc3 not found: " + args.tmc3_path)
    os.symlink(os.path.realpath(args.tmc3_path), os.path.join(
        args.output_dir, "tmc3_v21"))
    os.chdir(args.output_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.file_list, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    files = h5_files(args.data_root, args.file_list)[:args.max_samples]
    entries = entries[:args.max_samples]
    with open("selected_h5.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(entries) + "\n")
    with open("resolved_args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open("command.txt", "w", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable] + sys.argv) + "\n")

    base_checkpoint = torch.load(
        args.base_synthesis_checkpoint, map_location="cpu")
    require_base_checkpoint_contract(base_checkpoint, args)
    config = BaseSynthesisConfig(**base_checkpoint["config"])
    base = CanonicalBaseModel(args.released_checkpoint, config).cuda()
    released_enhancement_initialization = clone_state_dict_cpu(
        base.prefix.model.VAE)
    load_frozen_base(
        base, args.base_synthesis_checkpoint, args.released_checkpoint,
        args.conditioning_lambda)

    dataset = PCDataset(files, color_format="yuv", normalize=True)
    coords, feats = dataset[0]
    batch_coords, batch_feats = ME.utils.sparse_collate([coords], [feats])
    first_attribute = ME.SparseTensor(
        features=batch_feats, coordinates=batch_coords, tensor_stride=1,
        device="cuda")
    base_before = base(first_attribute, args.conditioning_lambda)["Base"]

    model = CanonicalScalableModel(
        base, conditioning_lambda=args.conditioning_lambda,
        enhancement_initialization_state=
        released_enhancement_initialization).cuda()
    model.train()
    base_after = model.base_forward(first_attribute)["Base"]
    base_invariance = sparse_difference(
        base_before, base_after, "Base before/after Enhancement creation")
    if base_invariance != 0.0:
        raise RuntimeError("Adding Enhancement changed Base")

    independence = parameter_independence(
        model.base.prefix.model.VAE, model.enhancement.vae,
        expected_state=released_enhancement_initialization)
    if independence["state_max_abs_difference"] != 0.0:
        raise RuntimeError("Enhancement initialization differs from released VAE")

    enhancement_parameters = list(model.enhancement.parameters())
    optimizer = torch.optim.Adam(enhancement_parameters, lr=1e-4)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups
        for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in enhancement_parameters}:
        raise RuntimeError("Optimizer parameter contract failed")
    if optimizer_ids & {id(parameter) for parameter in model.base.parameters()}:
        raise RuntimeError("Optimizer contains frozen Base parameters")

    frozen_contract = {
        "prefix_eval": not model.base.prefix.training,
        "prefix_requires_grad": any(
            parameter.requires_grad for parameter in model.base.prefix.parameters()),
        "base_synthesis_eval": not model.base.base_synthesis.training,
        "base_synthesis_requires_grad": any(
            parameter.requires_grad
            for parameter in model.base.base_synthesis.parameters()),
        "enhancement_train": model.enhancement.training,
        "enhancement_all_trainable": all(
            parameter.requires_grad for parameter in enhancement_parameters),
        "optimizer_enhancement_only": True,
    }
    if not (frozen_contract["prefix_eval"]
            and not frozen_contract["prefix_requires_grad"]
            and frozen_contract["base_synthesis_eval"]
            and not frozen_contract["base_synthesis_requires_grad"]
            and frozen_contract["enhancement_train"]
            and frozen_contract["enhancement_all_trainable"]):
        raise RuntimeError("Freeze/train contract failed")

    anchors = []
    deterministic_hard_max = {"Full": 0.0, "F_Full": 0.0, "dec_E": 0.0}
    encode_decode_max = {"Full": 0.0, "F_Full": 0.0, "dec_E": 0.0}
    total_points = 0
    total_base_bits = 0
    total_enhancement_bits = 0
    total_full_bits = 0
    total_estimated_bits = 0.0
    tensor_contract = None

    for index in range(len(dataset)):
        coords, feats = dataset[index]
        batch_coords, batch_feats = ME.utils.sparse_collate([coords], [feats])
        attribute = ME.SparseTensor(
            features=batch_feats, coordinates=batch_coords, tensor_stride=1,
            device="cuda")
        native = native_full_anchor(
            model.base.prefix, attribute, args.conditioning_lambda)
        deterministic = model.deterministic_forward(attribute)
        hard = model.hard_reconstruct(attribute)

        likelihood = deterministic["likelihood_E"]
        if likelihood is None or not torch.isfinite(likelihood).all():
            raise RuntimeError("Enhancement likelihood is missing/non-finite")
        estimated_bits = float(get_bits(likelihood).item())
        distortion, deterministic_psnr = quality(
            attribute, deterministic["Full"])
        hard_mse, hard_psnr = quality(attribute, hard["Full"])
        base_mse, base_psnr = quality(attribute, hard["Base"])
        for key in deterministic_hard_max:
            difference = sparse_difference(
                deterministic[key], hard[key], "deterministic/hard " + key)
            deterministic_hard_max[key] = max(
                deterministic_hard_max[key], difference)
            encoded_key = {
                "Full": "encoded_Full", "F_Full": "encoded_F_Full",
                "dec_E": "encoded_dec_E",
            }[key]
            encode_decode_max[key] = max(
                encode_decode_max[key], sparse_difference(
                    hard[encoded_key], hard[key], "encode/decode " + key))
        if max(deterministic_hard_max.values()) != 0.0:
            raise RuntimeError("Deterministic/hard Enhancement states differ")
        if max(encode_decode_max.values()) != 0.0:
            raise RuntimeError("Enhancement encode/decode states differ")
        if hard["prefix_rate"]["num_residual_streams"] != 4:
            raise RuntimeError("Scalable Base did not use exactly four streams")
        if hard["full_bits"] != hard["base_bits"] + hard["enhancement_bits"]:
            raise RuntimeError("Scalable physical rate identity failed")

        tensors = {
            "GT": attribute, "B": hard["Base"], "F_B": hard["F_B"],
            "d5p": hard["d5p"], "Full": hard["Full"],
            "F_Full": hard["F_Full"], "dec_E": hard["dec_E"],
            "latent": deterministic["Qlatent_E"],
        }
        expected_channels = {
            "GT": 3, "B": 3, "F_B": 128, "d5p": 128,
            "Full": 3, "F_Full": 128, "dec_E": 128, "latent": 32,
        }
        current_contract = {}
        for name, tensor in tensors.items():
            if tensor.F.shape[1] != expected_channels[name]:
                raise RuntimeError(name + " channel contract failed")
            if name != "latent":
                same_support(attribute, tensor, "GT/" + name)
                if any(stride != 1 for stride in tensor.tensor_stride):
                    raise RuntimeError(name + " is not stride 1")
            current_contract[name] = {
                "shape": list(tensor.F.shape),
                "tensor_stride": list(tensor.tensor_stride),
            }
        current_contract["prior_channels"] = 3 + 128 + 128
        if tensor_contract is None:
            tensor_contract = current_contract

        points = len(attribute)
        anchor = {
            "file": entries[index], "points": points,
            "native_full_psnr": native["psnr"],
            "native_r5_bits": native["r5_bits"],
            "native_total_bits": native["total_bits"],
            "native_total_bpp": native["total_bits"] / points,
            "base_psnr": base_psnr,
            "base_bits": hard["base_bits"],
            "base_bpp": hard["base_bits"] / points,
            "step0_full_psnr": hard_psnr,
            "step0_deterministic_psnr": deterministic_psnr,
            "step0_distortion": distortion,
            "step0_hard_mse": hard_mse,
            "step0_enhancement_estimated_bits": estimated_bits,
            "step0_enhancement_estimated_bpp": estimated_bits / points,
            "step0_enhancement_hard_bits": hard["enhancement_bits"],
            "step0_enhancement_hard_bpp": hard["enhancement_bits"] / points,
            "step0_full_bits": hard["full_bits"],
            "step0_full_bpp": hard["full_bits"] / points,
        }
        anchors.append(anchor)
        total_points += points
        total_base_bits += hard["base_bits"]
        total_enhancement_bits += hard["enhancement_bits"]
        total_full_bits += hard["full_bits"]
        total_estimated_bits += estimated_bits

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(first_attribute)
    likelihood = output["likelihood_E"]
    estimated_rate = get_bits(likelihood) / len(first_attribute)
    distortion = torch.mean((first_attribute.F - output["Full"].F) ** 2)
    diagnostic_loss = estimated_rate + distortion
    if not all(torch.isfinite(value) for value in (
            estimated_rate, distortion, diagnostic_loss)):
        raise RuntimeError("Non-finite differentiable Enhancement values")
    diagnostic_loss.backward()
    gradient_groups = {
        name: module_gradient_norm(getattr(model.enhancement.vae, name), name)
        for name in (
            "encoder", "EQlayer", "block_prior", "loc_net", "scale_net",
            "DQlayer", "decoder", "fuseNet", "outNet")
    }
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError("Frozen Base received gradients")

    results = {
        "status": "PASS",
        "metric": "normalized YUV 1:1:1 tensor MSE; PSNR peak=1",
        "conditioning_lambda": args.conditioning_lambda,
        "base_checkpoint": {
            "architecture": base_checkpoint["architecture"],
            "step": int(base_checkpoint.get("step", -1)),
            "sha256": sha256(args.base_synthesis_checkpoint),
        },
        "enhancement_initialization": {
            "source": "released_vae_exact_independent_clone",
            "released_checkpoint": args.released_checkpoint,
            "released_checkpoint_sha256": sha256(args.released_checkpoint),
        },
        "selected_h5": entries,
        "parameter_independence": independence,
        "freeze_contract": frozen_contract,
        "base_invariance_max_abs": base_invariance,
        "tensor_contract_first_sample": tensor_contract,
        "anchors": anchors,
        "aggregate_rate": {
            "points": total_points,
            "base_bits": total_base_bits,
            "base_bpp": total_base_bits / total_points,
            "enhancement_estimated_bits": total_estimated_bits,
            "enhancement_estimated_bpp": total_estimated_bits / total_points,
            "enhancement_hard_bits": total_enhancement_bits,
            "enhancement_hard_bpp": total_enhancement_bits / total_points,
            "full_bits": total_full_bits,
            "full_bpp": total_full_bits / total_points,
        },
        "deterministic_hard_max_abs": deterministic_hard_max,
        "encode_decode_max_abs": encode_decode_max,
        "gradient_gate": {
            "diagnostic_only_loss": float(diagnostic_loss.item()),
            "estimated_rate": float(estimated_rate.item()),
            "distortion": float(distortion.item()),
            "enhancement_group_norms": gradient_groups,
            "base_grad_none": True,
        },
        "scalable_base_residual_streams": 4,
        "native_r5_used_by_scalable_path": False,
        "min_max_excluded_from_rate": True,
    }
    with open("e0b_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("CANONICAL ENHANCEMENT E0-B PASS")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
