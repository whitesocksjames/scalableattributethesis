"""Payload-oriented Base/Full inference helpers.

The payload is a trusted torch serialization container. Its header and tensor
metadata are transport side information and are excluded from formal rates.
"""
import torch
import MinkowskiEngine as ME
import time


def _sync(tensor):
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _cpu_payload(items):
    return [{k: (v.detach().cpu() if torch.is_tensor(v) else v)
             for k, v in item.items() if k != "x_out" and k != "f_out" and k != "dec"}
            for item in items]


@torch.no_grad()
def encode_base(model, attribute):
    """Encode r1-r4 and Base synthesis; returns decoded Base plus payload."""
    model.eval()
    _sync(attribute)
    started = time.perf_counter()
    encoded, x_low, gpcc_bits = model.base.prefix.model(
        attribute, training=False, lmb=model.conditioning_lambda, encode=True,
        max_residual_stages=4)
    _sync(attribute)
    prefix_encode_seconds = time.perf_counter() - started
    streams = _cpu_payload(encoded)
    payload = {
        "schema": "scalable-attribute-base-v1",
        "x_low_F": x_low.F.detach().cpu(),
        "streams": streams,
        "geometry_C": attribute.C.detach().cpu(),
        "geometry_stride": list(attribute.tensor_stride),
    }
    _sync(attribute)
    started = time.perf_counter()
    state = model.base.prefix.model.decode(
        x0=_zero_sparse(attribute), x_low=x_low, enc_set_list=encoded,
        lmb=model.conditioning_lambda, max_residual_stages=4, return_state=True)
    completed = model.base.prefix._complete_state(*state)
    _sync(attribute)
    prefix_decode_seconds = time.perf_counter() - started
    _sync(attribute)
    started = time.perf_counter()
    base_result = model.base.reconstruct_from_state(completed)
    _sync(attribute)
    base_synthesis_seconds = time.perf_counter() - started
    payload["bits"] = {
        "x_low": int(gpcc_bits),
        "residual": [len(s["strings"]) * 8 for s in streams],
    }
    payload["bits"]["base"] = payload["bits"]["x_low"] + sum(payload["bits"]["residual"])
    payload["runtime_components"] = {
        "prefix_encode": prefix_encode_seconds,
        "prefix_decode": prefix_decode_seconds,
        "base_synthesis": base_synthesis_seconds,
    }
    return payload, base_result


def _zero_sparse(attribute):
    return ME.SparseTensor(features=torch.zeros_like(attribute.F),
                           coordinate_map_key=attribute.coordinate_map_key,
                           coordinate_manager=attribute.coordinate_manager,
                           device=attribute.device)


@torch.no_grad()
def decode_base(model, payload, geometry):
    """Decode Base with geometry support supplied independently by the caller."""
    if payload.get("schema") != "scalable-attribute-base-v1":
        raise ValueError("Unsupported Base payload schema")
    if list(payload.get("geometry_stride", [])) != list(geometry.tensor_stride):
        raise ValueError("Payload geometry stride does not match supplied geometry")
    if len(payload.get("streams", [])) != model.base.prefix.residual_stages:
        raise ValueError("Base payload must contain exactly four residual streams")
    if not torch.equal(payload["geometry_C"].to(geometry.C.device), geometry.C):
        raise ValueError("Payload geometry support does not match supplied geometry")
    model.eval()
    zero = _zero_sparse(geometry)
    xlow_feature = payload["x_low_F"].to(device=geometry.device, dtype=geometry.F.dtype)
    xlow = zero
    for pooling in model.base.prefix.model.pooling_list:
        xlow = pooling(xlow)
    xlow = ME.SparseTensor(features=xlow_feature, coordinate_map_key=xlow.coordinate_map_key,
                           coordinate_manager=xlow.coordinate_manager, device=geometry.device)
    streams = [{k: (v.to(geometry.device) if torch.is_tensor(v) else v)
                for k, v in item.items()} for item in payload["streams"]]
    state = model.base.prefix.model.decode(
        x0=zero, x_low=xlow, enc_set_list=streams, lmb=model.conditioning_lambda,
        max_residual_stages=4, return_state=True)
    return model.base.reconstruct_from_state(model.base.prefix._complete_state(*state))


@torch.no_grad()
def encode_full(model, attribute, base_result=None):
    """Encode enhancement conditioned on reconstructed Base state."""
    if base_result is None:
        base_payload, base_result = encode_base(model, attribute)
    else:
        base_payload = None
    b = base_result
    emb = model.base.prefix.lambda_embedding(model.conditioning_lambda, attribute.device)
    _sync(attribute)
    started = time.perf_counter()
    encoded = model.enhancement.encode(b["Base"], attribute, b["F_B"], b["d5p"], emb)
    _sync(attribute)
    enhancement_encode_seconds = time.perf_counter() - started
    enhancement_payload = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                           for k, v in encoded.items() if k not in ("x_out", "f_out", "dec")}
    decode_payload = {k: (v.to(attribute.device) if torch.is_tensor(v) else v)
                      for k, v in enhancement_payload.items()}
    _sync(attribute)
    started = time.perf_counter()
    decoded = model.enhancement.decode(decode_payload, b["Base"], b["F_B"], b["d5p"], emb)
    _sync(attribute)
    enhancement_decode_seconds = time.perf_counter() - started
    assert_sparse_equal(encoded["x_out"], decoded["x_out"], "Full enhancement encode/decode")
    return {"base_payload": base_payload, "enhancement_payload": enhancement_payload,
            "result": decoded, "encoded_result": encoded,
            "enhancement_bits": len(encoded["strings"]) * 8,
            "runtime_components": {
                **(base_payload["runtime_components"] if base_payload else {}),
                "enhancement_encode": enhancement_encode_seconds,
                "enhancement_decode": enhancement_decode_seconds,
            }}


@torch.no_grad()
def decode_full(model, base_result, payload):
    """Decode Full using the decoded Base state and enhancement payload."""
    emb = model.base.prefix.lambda_embedding(model.conditioning_lambda, base_result["Base"].device)
    data = {k: (v.to(base_result["Base"].device) if torch.is_tensor(v) else v)
            for k, v in payload.items()}
    return model.enhancement.decode(data, base_result["Base"], base_result["F_B"], base_result["d5p"], emb)


def assert_sparse_equal(left, right, label):
    """Require identical sparse support, stride, and decoded feature values."""
    if list(left.tensor_stride) != list(right.tensor_stride):
        raise RuntimeError(label + " tensor strides differ")
    if not torch.equal(left.C, right.C):
        raise RuntimeError(label + " coordinates differ")
    if not torch.equal(left.F, right.F):
        raise RuntimeError(label + " features differ")
