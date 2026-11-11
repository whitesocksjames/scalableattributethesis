"""Small standalone Base/Full inference command for RGB PLY samples."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import hashlib
import shlex


def read_rgb_ply(path, device):
    """Use the formal ASCII/binary PLY reader and original float32 YUV path."""
    import numpy as np
    import MinkowskiEngine as ME
    from data_utils.attribute.color_format import rgb2yuv
    from scripts.scalable_attribute.evaluation.prepare_ctc_formal_chunks import read_ply
    coords, rgb = read_ply(path)
    yuv = rgb2yuv(rgb.astype(np.float32), out_range=1).astype(np.float32)
    batched_coords, batched_feats = ME.utils.sparse_collate([coords.astype(np.int32)], [yuv])
    return ME.SparseTensor(features=batched_feats.contiguous(),
                           coordinates=batched_coords.contiguous(),
                           tensor_stride=1, device=device)


def run_pc_error(executable, reference, reconstruction):
    """Run pc_error_d and parse the standard Y/U/V MSE and PSNR rows."""
    command = [str(executable), "-a", str(reference), "-b", str(reconstruction),
               "--hausdorff=1", "--resolution=1", "--color=1"]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError("pc_error_d failed ({}): {}".format(proc.returncode, proc.stderr.strip()))
    keys = ("  c[0],    F", "  c[1],    F", "  c[2],    F",
            "  c[0],PSNRF", "  c[1],PSNRF", "  c[2],PSNRF")
    found = {}
    for line in proc.stdout.splitlines():
        for key in keys:
            if key in line:
                value = line.split(":", 1)[-1].strip().split()[0]
                found[key] = float(value)
    missing = [key for key in keys if key not in found]
    if missing:
        raise RuntimeError("pc_error_d output lacks metrics: " + ", ".join(missing))
    py, pu, pv = (found[key] for key in keys[3:])
    return {"y_mse": found[keys[0]], "u_mse": found[keys[1]], "v_mse": found[keys[2]],
            "y_psnr": py, "u_psnr": pu, "v_psnr": pv,
            "yuv_psnr_611": (6 * py + pu + pv) / 8}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _endpoint_record(reconstruction, bits, components, runtime):
    return {"status": "PASS", "reconstruction_ply": str(reconstruction.resolve()),
            "reconstruction_sha256": sha256_file(reconstruction),
            "physical_bits": int(bits), "components": components,
            "runtime_components": runtime, "bit_identity_pass": True}


def evaluate_chunk_manifest(args, parser):
    """Write aggregator-compatible per-chunk records; metric merge is one later call."""
    import numpy as np
    import torch
    import MinkowskiEngine as ME
    from scalable_attribute.runtime.codec import (
        encode_base, encode_full, decode_base, decode_full, assert_sparse_equal)
    from data_utils.attribute.inout import write_ply_ascii
    from data_utils.attribute.color_format import yuv2rgb
    from scalable_attribute.runtime.checkpoints import load_model

    manifest_path = Path(args.chunk_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1 or not manifest.get("union_verification", {}).get("verified"):
        parser.error("chunk manifest version or union verification is invalid")
    if len(manifest.get("sources", [])) != 1:
        parser.error("--chunk-manifest requires exactly one source record")
    source = manifest["sources"][0]
    source_path = Path(args.input).resolve()
    if source_path != Path(source["source_path"]).resolve() or sha256_file(source_path) != source["source_sha256"]:
        parser.error("input source path or SHA-256 does not match chunk manifest")
    chunks = source.get("chunks", [])
    if not chunks:
        parser.error("chunk manifest has no chunks")
    for expected_order, chunk in enumerate(chunks):
        chunk_path = (manifest_path.parent / chunk["path"]).resolve()
        if int(chunk["order"]) != expected_order or sha256_file(chunk_path) != chunk["sha256"]:
            parser.error("chunk order or SHA-256 mismatch at order {}".format(expected_order))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device=device)
    point_name = args.operating_point or str(model.conditioning_lambda)
    out = Path(args.output).resolve()
    out.mkdir(parents=True)
    records = []
    for chunk in chunks:
        order = int(chunk["order"])
        chunk_path = (manifest_path.parent / chunk["path"]).resolve()
        attribute = read_rgb_ply(chunk_path, device)
        if len(attribute) != int(chunk["count"]):
            raise RuntimeError("chunk point count mismatch at order {}".format(order))
        chunk_out = out / "chunks" / "{:06d}".format(order)
        chunk_out.mkdir(parents=True)
        (chunk_out / "tmc3_v21").symlink_to(Path(args.gpcc).resolve())
        old_cwd = os.getcwd()
        try:
            os.chdir(chunk_out)
            base_payload, base = encode_base(model, attribute)
            full = encode_full(model, attribute, base) if args.endpoint in ("full", "both") else None
        finally:
            os.chdir(old_cwd)
        torch.save(base_payload, chunk_out / "base.payload.pt")
        geometry = ME.SparseTensor(features=torch.zeros_like(attribute.F),
                                   coordinate_map_key=attribute.coordinate_map_key,
                                   coordinate_manager=attribute.coordinate_manager,
                                   device=attribute.device)
        decoded_base = decode_base(model, torch.load(chunk_out / "base.payload.pt", map_location="cpu", weights_only=False), geometry)
        assert_sparse_equal(base["Base"], decoded_base["Base"], "Base chunk encode/decode")
        base_ply = chunk_out / "base.ply"
        write_ply_ascii(str(base_ply), base["Base"].C[:, 1:].cpu().numpy(),
                        np.clip(yuv2rgb(torch.clamp(base["Base"].F, 0, 1), out_range=255).round().cpu().numpy(), 0, 255).astype(np.uint8))
        base_bits = base_payload["bits"]["base"]
        endpoint_data = {"Base": _endpoint_record(
            base_ply, base_bits,
            {"x_low": base_payload["bits"]["x_low"], **{
                "r{}".format(i + 1): bits for i, bits in enumerate(base_payload["bits"]["residual"]) }},
            base_payload["runtime_components"])}
        if full is not None:
            torch.save(full["enhancement_payload"], chunk_out / "enhancement.payload.pt")
            decoded_full = decode_full(model, decoded_base, torch.load(
                chunk_out / "enhancement.payload.pt", map_location="cpu", weights_only=False))
            assert_sparse_equal(full["result"]["x_out"], decoded_full["x_out"], "Full chunk encode/decode")
            full_ply = chunk_out / "full.ply"
            write_ply_ascii(str(full_ply), full["result"]["x_out"].C[:, 1:].cpu().numpy(),
                            np.clip(yuv2rgb(torch.clamp(full["result"]["x_out"].F, 0, 1), out_range=255).round().cpu().numpy(), 0, 255).astype(np.uint8))
            endpoint_data["Full"] = _endpoint_record(
                full_ply, base_bits + full["enhancement_bits"],
                {"base": base_bits, "enhancement": full["enhancement_bits"]},
                {**base_payload["runtime_components"], **full["runtime_components"]})
        result_path = out / "results" / "chunk_{:06d}.json".format(order)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "schema_version": 1, "sample": args.sample or Path(source_path).stem,
            "method": args.method, "operating_point": point_name,
            "chunk": {"order": order, "path": chunk["path"], "sha256": chunk["sha256"], "count": int(chunk["count"])},
            "endpoints": endpoint_data,
        }, indent=2) + "\n", encoding="utf-8")
        records.append(str(result_path.resolve()))
    result_manifest = out / "input_manifest.json"
    result_manifest.write_text(json.dumps({"schema_version": 1, "results": records}, indent=2) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    aggregator = repo_root / "scripts/scalable_attribute/evaluation/aggregate_formal_chunk_results.py"
    command = ["python", str(aggregator), "--source-ply", str(source_path),
               "--chunk-manifest", str(manifest_path), "--input-manifest", str(result_manifest),
               "--endpoint", "Base", "--output-dir", str(out / "aggregated"),
               "--merged-dir", str(out / "merged")]
    if "Full" in endpoint_data:
        command[command.index("--output-dir"):command.index("--output-dir")] = ["--endpoint", "Full"]
    if args.pc_error:
        command.extend(["--pc-error", str(Path(args.pc_error).resolve())])
    (out / "aggregate_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="full sample for ordinary inference or canonical source for chunk mode")
    parser.add_argument("--chunk-manifest", help="prepare_ctc_formal_chunks manifest; writes per-chunk records for the reducer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpcc", required=True, help="tmc3_v21 executable")
    parser.add_argument("--pc-error", help="optional pc_error_d executable")
    parser.add_argument("--endpoint", choices=("base", "full", "both"), default="both")
    parser.add_argument("--sample", help="aggregation sample identifier in chunk mode")
    parser.add_argument("--method", default="ours", help="aggregation method identifier in chunk mode")
    parser.add_argument("--operating-point", help="aggregation operating point; defaults to checkpoint lambda")
    args = parser.parse_args(argv)
    if not args.chunk_manifest and not args.input:
        parser.error("--input is required unless --chunk-manifest is supplied")
    if args.chunk_manifest and not args.input:
        parser.error("chunk mode requires --input as the canonical full source PLY")
    for label, filename in (("checkpoint", args.checkpoint), ("input", args.input), ("G-PCC", args.gpcc), ("chunk manifest", args.chunk_manifest)):
        if filename and not Path(filename).is_file():
            parser.error("{} path does not exist: {}".format(label, filename))
    if args.pc_error and (not Path(args.pc_error).is_file() or not os.access(args.pc_error, os.X_OK)):
        parser.error("pc_error_d path is missing or not executable: {}".format(args.pc_error))
    if Path(args.gpcc).is_file() and not os.access(args.gpcc, os.X_OK):
        parser.error("G-PCC path is not executable: {}".format(args.gpcc))
    if Path(args.output).exists():
        parser.error("output path already exists: {}".format(args.output))
    if args.chunk_manifest:
        return evaluate_chunk_manifest(args, parser)
    import MinkowskiEngine  # noqa: F401
    import numpy as np
    import torch
    from scalable_attribute.runtime.codec import (
        encode_base, encode_full, decode_base, decode_full, assert_sparse_equal)
    from data_utils.attribute.inout import write_ply_ascii
    from data_utils.attribute.color_format import yuv2rgb
    from scalable_attribute.runtime.checkpoints import load_model
    import MinkowskiEngine as ME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device=device)
    attribute = read_rgb_ply(args.input, device)
    out = Path(args.output)
    out.mkdir(parents=True)
    # The legacy G-PCC wrapper invokes ./tmc3_v21 from cwd. Keep all temporary
    # files local to this output and expose the requested executable there.
    gpcc_link = out / "tmc3_v21"
    gpcc_link.symlink_to(Path(args.gpcc).resolve())
    previous_cwd = os.getcwd()
    full = None
    try:
        os.chdir(out)
        base_payload, base = encode_base(model, attribute)
        full = encode_full(model, attribute, base) if args.endpoint in ("full", "both") else None
    finally:
        os.chdir(previous_cwd)
    torch.save(base_payload, out / "base.payload.pt")
    torch.save({"geometry_C": attribute.C.cpu(), "tensor_stride": list(attribute.tensor_stride)}, out / "geometry.pt")
    # Recreate geometry from coordinates only: decoding receives no reference RGB.
    geometry = ME.SparseTensor(features=torch.zeros_like(attribute.F),
                               coordinate_map_key=attribute.coordinate_map_key,
                               coordinate_manager=attribute.coordinate_manager,
                               device=attribute.device)
    loaded_base_payload = torch.load(out / "base.payload.pt", map_location="cpu", weights_only=False)
    decoded_base = decode_base(model, loaded_base_payload, geometry)
    assert_sparse_equal(base["Base"], decoded_base["Base"], "Base encode/decode")
    report = {
        "endpoint": args.endpoint,
        "rate_semantics": "Base counts G-PCC-reported x_low attribute bits plus r1-r4 strings; Full adds Enhancement strings; torch container and geometry support are excluded",
        "base_bits": base_payload["bits"]["base"],
        "base_bpp": base_payload["bits"]["base"] / len(attribute),
        "base_encode_decode_exact": True,
    }
    write_ply_ascii(str(out / "base.ply"), base["Base"].C[:, 1:].cpu().numpy(),
                    np.clip(yuv2rgb(torch.clamp(base["Base"].F, 0, 1), out_range=255).round().cpu().numpy(), 0, 255).astype(np.uint8))
    if full is not None:
        torch.save(full["enhancement_payload"], out / "enhancement.payload.pt")
        loaded_enhancement = torch.load(out / "enhancement.payload.pt", map_location="cpu", weights_only=False)
        decoded_full = decode_full(model, decoded_base, loaded_enhancement)
        assert_sparse_equal(full["result"]["x_out"], decoded_full["x_out"], "Full encode/decode")
        report["full_bits"] = report["base_bits"] + full["enhancement_bits"]
        report["enhancement_bits"] = full["enhancement_bits"]
        report["full_bpp"] = report["full_bits"] / len(attribute)
        report["full_encode_decode_exact"] = True
        write_ply_ascii(str(out / "full.ply"), full["result"]["x_out"].C[:, 1:].cpu().numpy(),
                        np.clip(yuv2rgb(torch.clamp(full["result"]["x_out"].F, 0, 1), out_range=255).round().cpu().numpy(), 0, 255).astype(np.uint8))
    if args.pc_error:
        report["pc_error"] = {"base": run_pc_error(args.pc_error, args.input, out / "base.ply")}
        if full is not None:
            report["pc_error"]["full"] = run_pc_error(args.pc_error, args.input, out / "full.ply")
    (out / "rates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
