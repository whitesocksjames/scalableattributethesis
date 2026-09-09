#!/usr/bin/env python3
"""Run one exact physical static-RGB point for pinned upstream Unicorn.

The runner is intentionally self-contained.  The only model, PLY, colour, and
metric code it imports is loaded from ``--upstream-root`` after the command
line has been replaced with the released static lossy-RGB configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import types


JSON_NAME = "physical_rd.json"
CSV_NAME = "physical_rd.csv"
RECONSTRUCTION_NAME = "reconstruction.ply"
CHUNK_RESULT_NAME = "chunk_result.json"
CODEC_CWD_NAME = "codec_cwd"

CSV_FIELDS = (
    "dataset",
    "sequence",
    "rate_id",
    "checkpoint_profile",
    "lambda_value",
    "lambda",
    "points",
    "input_points",
    "reconstruction_points",
    "entropy_stream_bits",
    "entropy_bits",
    "gpcc_bits",
    "total_bits",
    "physical_bits",
    "physical_bpp",
    "bit_identity_pass",
    "y_psnr",
    "u_psnr",
    "v_psnr",
    "yuv_psnr_611",
    "Y",
    "U",
    "V",
    "YUV611",
    "input_ply",
    "reconstruction_ply",
    "checkpoint",
    "upstream_root",
    "upstream_commit",
    "input_ply_sha256",
    "checkpoint_sha256",
    "reconstruction_ply_sha256",
    "input_sha256",
    "reconstruction_sha256",
    "input_read_seconds",
    "checkpoint_load_seconds",
    "encode_seconds",
    "decode_seconds",
    "write_seconds",
    "pc_error_seconds",
    "total_seconds",
    "peak_vram_bytes",
    "peak_vram_gib",
)

_UPSTREAM_PACKAGES = (
    "basic_models",
    "cfg",
    "data_utils",
    "lossy_attribute",
    "pipelines",
    "third_party",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one exact physical static-RGB point using upstream Unicorn"
    )
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--lambda-value", required=True, type=int)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--rate-id", required=True)
    parser.add_argument("--checkpoint-profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-order", type=int)
    parser.add_argument("--chunk-path")
    parser.add_argument("--chunk-sha256")
    parser.add_argument("--chunk-count", type=int)
    return parser.parse_args(argv)


def sha256_file(path):
    """Return the SHA-256 digest of one file without loading it wholly."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _path_is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _clear_loaded_upstream_packages():
    """Drop same-named packages so an earlier thesis import cannot win."""

    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in _UPSTREAM_PACKAGES):
            sys.modules.pop(name, None)


def _isolate_import_path(upstream_root):
    """Put the supplied root first and remove this repository's import paths."""

    project_root = Path(__file__).resolve().parents[3]
    kept = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except (OSError, RuntimeError):
            kept.append(entry)
            continue
        if (_path_is_within(resolved, project_root)
                and resolved != Path(upstream_root).resolve()):
            continue
        kept.append(entry)
    sys.path[:] = [str(Path(upstream_root).resolve())] + kept


class _FailClosedOpen3D(types.ModuleType):
    """Allow upstream imports while forbidding an unintended Open3D code path."""

    def __getattr__(self, name):
        raise RuntimeError(
            "Unexpected Open3D API access in the ASCII static-RGB path: " + name
        )


def _ensure_open3d_importable():
    """Install a fail-closed shim because upstream imports Open3D eagerly."""

    try:
        importlib.import_module("open3d")
    except (ImportError, OSError):
        sys.modules["open3d"] = _FailClosedOpen3D("open3d")


@contextlib.contextmanager
def _upstream_import_args():
    """Supply only the pinned static lossy-RGB arguments to upstream parsers."""

    previous = list(sys.argv)
    sys.argv = [
        "evaluate_unicorn_official_physical",
        "--color_format", "yuv",
        "--normalize", "1",
        "--Vmode", "1",
        "--scale", "5",
        "--stage", "1",
        "--in_channels", "3",
        "--channels", "128",
        "--latent_channels", "32",
        "--kernel_size", "3",
        "--block_layers", "2",
        "--block_type", "resnet",
        "--gpcc_version", "21",
        "--DBG", "0",
    ]
    try:
        yield
    finally:
        sys.argv = previous


def _load_upstream(upstream_root):
    """Load all executed model helpers from the explicitly supplied root."""

    root = Path(upstream_root).expanduser().resolve()
    required = (
        root / "lossy_attribute" / "model.py",
        root / "lossy_attribute" / "coder.py",
        root / "data_utils" / "attribute" / "inout.py",
        root / "data_utils" / "attribute" / "color_format.py",
        root / "third_party" / "pc_error_attr.py",
        root / "third_party" / "tmc3_v21",
    )
    missing = [str(path) for path in required if not path.exists()]
    if not root.is_dir() or missing:
        detail = ", ".join(missing) if missing else str(root)
        raise FileNotFoundError("Invalid upstream Unicorn root: " + detail)

    _isolate_import_path(root)
    _clear_loaded_upstream_packages()
    _ensure_open3d_importable()

    with _upstream_import_args():
        torch = importlib.import_module("torch")
        minkowski_engine = importlib.import_module("MinkowskiEngine")
        model_module = importlib.import_module("lossy_attribute.model")
        coder_module = importlib.import_module("lossy_attribute.coder")
        inout_module = importlib.import_module("data_utils.attribute.inout")
        color_module = importlib.import_module("data_utils.attribute.color_format")
        metric_module = importlib.import_module("third_party.pc_error_attr")

    checked = (
        model_module,
        coder_module,
        inout_module,
        color_module,
        metric_module,
    )
    for module in checked:
        origin = getattr(module, "__file__", None)
        if origin is None or not _path_is_within(origin, root):
            raise RuntimeError(
                "Upstream import escaped --upstream-root: {}".format(module)
            )

    return {
        "torch": torch,
        "ME": minkowski_engine,
        "MultiscaleVAE": model_module.MultiscaleVAE,
        "LossyAttributeCoder": coder_module.LossyAttributeCoder,
        "read_ply_ascii": inout_module.read_ply_ascii,
        "rgb2yuv": color_module.rgb2yuv,
        "pc_error": metric_module.pc_error,
        "upstream_root": root,
    }


def _ascii_header_line(stream, path):
    raw = stream.readline()
    if not raw:
        raise ValueError("Truncated PLY header: {}".format(path))
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("PLY is not ASCII: {}".format(path)) from exc


def _validate_ascii_six_column_ply(path):
    """Validate an ASCII x/y/z/r/g/b PLY and return its declared point count."""

    path = Path(path)
    with open(path, "rb") as stream:
        if _ascii_header_line(stream, path) != "ply":
            raise ValueError("Expected PLY magic: {}".format(path))
        format_line = _ascii_header_line(stream, path)
        if format_line != "format ascii 1.0":
            raise ValueError("Expected ASCII PLY format: {}".format(path))

        vertex_count = None
        vertex_properties = []
        in_vertex = False
        while True:
            line = _ascii_header_line(stream, path)
            if line == "end_header":
                break
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "element":
                if len(fields) != 3:
                    raise ValueError("Malformed PLY element line: {}".format(path))
                in_vertex = fields[1] == "vertex"
                if in_vertex:
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and in_vertex:
                if len(fields) != 3 or fields[1] == "list":
                    raise ValueError(
                        "Expected scalar vertex properties: {}".format(path)
                    )
                vertex_properties.append(fields[2])

        if vertex_count is None or vertex_count <= 0:
            raise ValueError("PLY has no positive vertex count: {}".format(path))
        if vertex_properties != ["x", "y", "z", "red", "green", "blue"]:
            raise ValueError(
                "Expected six-column x/y/z/r/g/b PLY: {}".format(path)
            )

        for index in range(vertex_count):
            line = _ascii_header_line(stream, path)
            values = line.split()
            if len(values) != 6:
                raise ValueError(
                    "Expected six values at PLY row {}: {}".format(index, path)
                )
            try:
                for value in values:
                    float(value)
            except ValueError as exc:
                raise ValueError(
                    "Non-numeric PLY row {}: {}".format(index, path)
                ) from exc

        if stream.read().strip():
            raise ValueError("PLY has data beyond its vertex count: {}".format(path))
    return vertex_count


def ply_points(path):
    """Public stdlib-only PLY point-count helper used for hard gates."""

    return _validate_ascii_six_column_ply(path)


def _exact_integer(value, name):
    if hasattr(value, "item"):
        value = value.item()
    integer = int(value)
    if value != integer:
        raise ValueError("{} is not an exact integer: {!r}".format(name, value))
    return integer


@contextlib.contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _prepare_output_dir(output_dir):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = (
        output / JSON_NAME,
        output / CSV_NAME,
        output / RECONSTRUCTION_NAME,
        output / CHUNK_RESULT_NAME,
        output / CODEC_CWD_NAME,
    )
    existing = [str(path) for path in protected
                if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))
    return output


def _atomic_write_text(path, text):
    """Publish a complete text file with a same-directory atomic replacement."""

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite " + str(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError("Refusing to overwrite " + str(path))
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_text(row):
    values = dict(row)
    values["entropy_stream_bits"] = "|".join(
        str(value) for value in row["entropy_stream_bits"]
    )
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=CSV_FIELDS, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerow(values)
    return stream.getvalue()


def _to_python_number(value):
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _measure_pc_error(pc_error, input_ply, reconstruction_ply):
    started = time.perf_counter()
    raw = pc_error(str(input_ply), str(reconstruction_ply), res=1)
    elapsed = time.perf_counter() - started
    y = _to_python_number(raw["  c[0],PSNRF"])
    u = _to_python_number(raw["  c[1],PSNRF"])
    v = _to_python_number(raw["  c[2],PSNRF"])
    return {
        "y_psnr": y,
        "u_psnr": u,
        "v_psnr": v,
        "yuv_psnr_611": (6.0 * y + u + v) / 8.0,
        "pc_error_seconds": elapsed,
    }


def _load_attribute(modules, input_ply, expected_points):
    coords, rgb = modules["read_ply_ascii"](str(input_ply))
    if len(coords) != expected_points:
        raise RuntimeError(
            "Upstream ASCII reader point count mismatch: expected {}, got {}".format(
                expected_points, len(coords)
            )
        )
    if len(rgb.shape) != 2 or rgb.shape[1] != 3:
        raise ValueError("Upstream reader did not return three RGB channels")
    yuv = modules["rgb2yuv"](
        rgb.astype("float32"), out_range=1
    ).astype("float32")
    batch_coords, batch_feats = modules["ME"].utils.sparse_collate(
        [coords], [yuv]
    )
    attribute = modules["ME"].SparseTensor(
        features=batch_feats,
        coordinates=batch_coords,
        tensor_stride=1,
        device="cuda",
    )
    if len(attribute) != expected_points:
        raise RuntimeError(
            "Sparse input point count changed: expected {}, got {}".format(
                expected_points, len(attribute)
            )
        )
    return attribute


def _run_physical(modules, model, coder, attribute, input_ply, reconstruction_ply,
                  codec_cwd, lambda_value, input_points, output):
    torch = modules["torch"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    with _working_directory(codec_cwd):
        enc_set_list, x_low, gpcc_bits = model.forward(
            x=attribute, training=False, lmb=lambda_value, encode=True
        )
        torch.cuda.synchronize()
        encode_seconds = time.perf_counter() - encode_started

        entropy_stream_bits = [
            int(len(item["strings"]) * 8) for item in enc_set_list
        ]
        entropy_bits = int(sum(len(item["strings"]) * 8 for item in enc_set_list))
        gpcc_bits = _exact_integer(gpcc_bits, "gpcc_bits")
        total_bits = entropy_bits + gpcc_bits
        if total_bits != entropy_bits + gpcc_bits:
            raise RuntimeError("Physical bit identity failed")

        x0 = modules["ME"].SparseTensor(
            features=torch.zeros_like(attribute.F),
            coordinate_map_key=attribute.coordinate_map_key,
            coordinate_manager=attribute.coordinate_manager,
            device=attribute.device,
        )
        decode_started = time.perf_counter()
        reconstruction = model.decode(
            x0=x0,
            x_low=x_low,
            enc_set_list=enc_set_list,
            lmb=lambda_value,
        )
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        if len(reconstruction) != input_points:
            raise RuntimeError(
                "Reconstruction point-count gate failed: expected {}, got {}".format(
                    input_points, len(reconstruction)
                )
            )

        write_started = time.perf_counter()
        coder.write_file(
            x_rec=reconstruction, filedir_rec=str(reconstruction_ply)
        )
        write_seconds = time.perf_counter() - write_started

    written_points = ply_points(reconstruction_ply)
    if written_points != input_points:
        raise RuntimeError(
            "Written reconstruction point-count gate failed: expected {}, got {}".format(
                input_points, written_points
            )
        )
    torch.cuda.synchronize()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "entropy_stream_bits": entropy_stream_bits,
        "entropy_bits": entropy_bits,
        "gpcc_bits": gpcc_bits,
        "total_bits": total_bits,
        "bit_identity_pass": total_bits == entropy_bits + gpcc_bits,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "write_seconds": write_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gib": peak_vram_bytes / (1024 ** 3),
        "reconstruction_points": written_points,
    }


def main(argv=None):
    args = parse_args(argv)
    if not args.source_commit.strip():
        raise ValueError("--source-commit must not be empty")

    upstream_root = Path(args.upstream_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    input_ply = Path(args.input_ply).expanduser().resolve()
    output = _prepare_output_dir(args.output_dir)
    for path in (checkpoint, input_ply):
        if not path.is_file():
            raise FileNotFoundError(path)
    tmc3 = upstream_root / "third_party" / "tmc3_v21"
    if not tmc3.is_file():
        raise FileNotFoundError(tmc3)

    overall_started = time.perf_counter()
    input_points = ply_points(input_ply)
    input_read_seconds = time.perf_counter() - overall_started
    input_sha256 = sha256_file(input_ply)
    checkpoint_sha256 = sha256_file(checkpoint)
    chunk_values = (args.chunk_order, args.chunk_path, args.chunk_sha256,
                    args.chunk_count)
    chunk_metadata = None
    if any(value is not None for value in chunk_values):
        if any(value is None for value in chunk_values):
            raise ValueError(
                "chunk order/path/SHA-256/count must be supplied together")
        chunk_path = Path(args.chunk_path).expanduser().resolve()
        if args.chunk_order < 0 or args.chunk_count < 0:
            raise ValueError("chunk order/count must be non-negative")
        if chunk_path != input_ply:
            raise ValueError("chunk path must identify --input-ply")
        if args.chunk_sha256.lower() != input_sha256:
            raise ValueError("chunk SHA-256 does not match --input-ply")
        if args.chunk_count != input_points:
            raise ValueError("chunk count does not match evaluator input points")
        chunk_metadata = {
            "order": args.chunk_order,
            "path": str(chunk_path),
            "sha256": input_sha256,
            "count": args.chunk_count,
        }

    modules = _load_upstream(upstream_root)
    torch = modules["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this physical runner")
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")

    checkpoint_started = time.perf_counter()
    model = modules["MultiscaleVAE"]().to(device)
    checkpoint_state = torch.load(str(checkpoint), map_location="cpu")
    released_state = checkpoint_state["model"]
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in released_state.items() if key in model_state
    }
    model_state.update(filtered_state)
    model.load_state_dict(model_state)
    model.requires_grad_(False)
    model.eval()
    checkpoint_load_seconds = time.perf_counter() - checkpoint_started

    attribute = _load_attribute(modules, input_ply, input_points)
    codec_cwd = output / CODEC_CWD_NAME
    codec_cwd.mkdir()
    (codec_cwd / "tmc3_v21").symlink_to(tmc3)
    reconstruction_ply = output / RECONSTRUCTION_NAME

    coder = modules["LossyAttributeCoder"](model=model)
    physical = _run_physical(
        modules,
        model,
        coder,
        attribute,
        input_ply,
        reconstruction_ply,
        codec_cwd,
        args.lambda_value,
        input_points,
        output,
    )
    quality = _measure_pc_error(
        modules["pc_error"], input_ply, reconstruction_ply
    )
    reconstruction_sha256 = sha256_file(reconstruction_ply)
    total_seconds = time.perf_counter() - overall_started

    row = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "rate_id": args.rate_id,
        "checkpoint_profile": args.checkpoint_profile,
        "lambda_value": args.lambda_value,
        "lambda": args.lambda_value,
        "points": input_points,
        "input_points": input_points,
        "reconstruction_points": physical["reconstruction_points"],
        "entropy_stream_bits": physical["entropy_stream_bits"],
        "entropy_bits": physical["entropy_bits"],
        "gpcc_bits": physical["gpcc_bits"],
        "total_bits": physical["total_bits"],
        "physical_bits": physical["total_bits"],
        "physical_bpp": physical["total_bits"] / input_points,
        "bit_identity_pass": physical["bit_identity_pass"],
        "y_psnr": quality["y_psnr"],
        "u_psnr": quality["u_psnr"],
        "v_psnr": quality["v_psnr"],
        "yuv_psnr_611": quality["yuv_psnr_611"],
        "Y": quality["y_psnr"],
        "U": quality["u_psnr"],
        "V": quality["v_psnr"],
        "YUV611": quality["yuv_psnr_611"],
        "input_ply": str(input_ply),
        "reconstruction_ply": str(reconstruction_ply),
        "checkpoint": str(checkpoint),
        "upstream_root": str(upstream_root),
        "upstream_commit": args.source_commit,
        "input_ply_sha256": input_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "reconstruction_ply_sha256": reconstruction_sha256,
        "input_sha256": input_sha256,
        "reconstruction_sha256": reconstruction_sha256,
        "input_read_seconds": input_read_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "encode_seconds": physical["encode_seconds"],
        "decode_seconds": physical["decode_seconds"],
        "write_seconds": physical["write_seconds"],
        "pc_error_seconds": quality["pc_error_seconds"],
        "total_seconds": total_seconds,
        "peak_vram_bytes": physical["peak_vram_bytes"],
        "peak_vram_gib": physical["peak_vram_gib"],
    }

    summary = {
        "status": "PASS",
        "schema_id": "unicorn_static_lossy_rgb_physical_v1",
        "source": {
            "upstream_root": str(upstream_root),
            "upstream_commit": args.source_commit,
            "no_thesis_model_imports": True,
        },
        "contract": {
            "dataset": args.dataset,
            "sequence": args.sequence,
            "rate_id": args.rate_id,
            "checkpoint_profile": args.checkpoint_profile,
            "lambda_value": args.lambda_value,
            "color_format": "yuv",
            "normalize": 1,
            "Vmode": 1,
            "scale": 5,
            "stage": 1,
            "channels": 128,
            "latent_channels": 32,
            "input_format": "ASCII six-column x/y/z/r/g/b PLY",
            "metric": "upstream pc_error once; YUV611=(6Y+U+V)/8",
        },
        "artifacts": {
            "input_ply": str(input_ply),
            "checkpoint": str(checkpoint),
            "reconstruction_ply": str(reconstruction_ply),
            "input_points": input_points,
            "reconstruction_points": physical["reconstruction_points"],
            "input_sha256": input_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "reconstruction_sha256": reconstruction_sha256,
        },
        "hashes": {
            "input_ply_sha256": input_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "reconstruction_ply_sha256": reconstruction_sha256,
            "input_sha256": input_sha256,
            "reconstruction_sha256": reconstruction_sha256,
        },
        "checkpoint_load": {
            "released_keys": len(released_state),
            "compatible_keys_loaded": len(filtered_state),
            "released_keys_ignored_by_official_filter": sorted(
                set(released_state) - set(filtered_state)
            ),
            "model_keys_not_in_released_checkpoint": sorted(
                set(model_state) - set(released_state)
            ),
            "load_semantics": (
                "official filtered-state update followed by model.load_state_dict"
            ),
        },
        "bit_accounting": {
            "entropy_stream_bits": physical["entropy_stream_bits"],
            "entropy_bits": physical["entropy_bits"],
            "gpcc_bits": physical["gpcc_bits"],
            "total_bits": physical["total_bits"],
            "physical_bits": physical["total_bits"],
            "identity": "total_bits = entropy_bits + gpcc_bits",
            "bit_identity_pass": physical["bit_identity_pass"],
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "peak_vram_bytes": physical["peak_vram_bytes"],
            "peak_vram_gib": physical["peak_vram_gib"],
            "codec_cwd": str(codec_cwd),
            "codec_binary": str(codec_cwd / "tmc3_v21"),
            "seed": 0,
        },
        "timings": {
            "input_read_seconds": input_read_seconds,
            "checkpoint_load_seconds": checkpoint_load_seconds,
            "encode_seconds": physical["encode_seconds"],
            "decode_seconds": physical["decode_seconds"],
            "write_seconds": physical["write_seconds"],
            "pc_error_seconds": quality["pc_error_seconds"],
            "total_seconds": total_seconds,
        },
        "row": row,
        "rows": [row],
    }
    _atomic_write_text(
        output / JSON_NAME,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(output / CSV_NAME, _csv_text(row))
    if chunk_metadata is not None:
        chunk_result = {
            "schema_version": 1,
            "sample": args.sequence,
            "method": "Original Unicorn",
            "operating_point": args.rate_id,
            "chunk": chunk_metadata,
            "status": "PASS",
            "endpoints": {
                "Original Unicorn": {
                    "status": "FORMAL_REUSABLE",
                    "reconstruction_ply": str(reconstruction_ply),
                    "reconstruction_sha256": reconstruction_sha256,
                    "physical_bits": physical["total_bits"],
                    "components": {
                        "gpcc": physical["gpcc_bits"],
                        "entropy": physical["entropy_bits"],
                    },
                    "runtime_components": {
                        "encode": physical["encode_seconds"],
                        "decode": physical["decode_seconds"],
                    },
                }
            },
        }
        _atomic_write_text(
            output / CHUNK_RESULT_NAME,
            json.dumps(chunk_result, indent=2, sort_keys=True,
                       allow_nan=False) + "\n",
        )
    print(json.dumps({"status": "PASS", "json": str(output / JSON_NAME),
                      "csv": str(output / CSV_NAME)}))


if __name__ == "__main__":
    main()
