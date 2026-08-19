"""Prepare the RWTT training data with the public Unicorn-v1 utilities.

This is a repository-compatible entry point for the two relevant stages in
the author's original ``dataset_processing.py``:

1. ``mesh2pc_color`` densely samples textured OBJ meshes and voxelizes them.
2. ``partition_color`` splits the resulting attribute HDF5 files into blocks.

Input and output roots are always supplied explicitly so that the script can
be used with different local and HPC directory layouts.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _input_files(rootdir: str, suffix: str, input_file: Optional[str] = None) -> list[str]:
    if input_file is not None:
        return [input_file]
    suffix = suffix.lstrip(".")
    return sorted(glob.glob(os.path.join(rootdir, "**", f"*.{suffix}"), recursive=True))


def _relative_stem(path: str, rootdir: str) -> Path:
    return Path(os.path.relpath(path, rootdir)).with_suffix("")


def mesh2pc_color(args: argparse.Namespace) -> None:
    import MinkowskiEngine as ME
    import numpy as np
    import torch
    from PIL import Image
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.ops import sample_points_from_meshes

    from data_utils.attribute.inout import read_h5, write_h5, write_ply_ascii
    from data_utils.geometry.quantize import quantize_resolution

    # RWTT contains legitimate very-high-resolution textures. Disabling this
    # Pillow guard does not rescale or otherwise alter their RGB values.
    Image.MAX_IMAGE_PIXELS = None

    input_files = _input_files(args.input_rootdir, args.input_format, args.input_file)
    if args.shuffle:
        random.Random(args.seed).shuffle(input_files)
    input_files = input_files[: args.input_length]
    print(f"Input meshes: {len(input_files)}")

    output_count = 0
    for input_path in input_files:
        mesh = load_objs_as_meshes([input_path], device="cpu")
        xyz, rgb = sample_points_from_meshes(
            meshes=mesh,
            num_samples=args.num_points,
            return_textures=True,
        )
        xyz = xyz[0].detach().cpu().numpy()
        rgb = rgb[0].detach().cpu().numpy()
        sampled_count = len(xyz)
        print(f"Sampled points: {sampled_count}")

        # The public V1 function returns coordinates plus dequantization
        # metadata. RWTT preprocessing only stores the quantized coordinates.
        coords, _, _ = quantize_resolution(xyz, resolution=args.resolution)

        coords_batch, feats_batch = ME.utils.sparse_collate(
            [torch.as_tensor(coords)], [torch.as_tensor(rgb)]
        )
        point_cloud = ME.SparseTensor(
            features=feats_batch,
            coordinates=coords_batch,
            tensor_stride=1,
            device="cpu",
        )
        coords = point_cloud.C.detach().cpu().numpy()[:, 1:]
        feats = np.clip(
            np.rint(point_cloud.F.detach().cpu().numpy() * 255.0), 0, 255
        ).astype("uint8")

        relative_stem = _relative_stem(input_path, args.input_rootdir)
        h5_path = Path(args.output_rootdir) / "h5" / Path(f"{relative_stem}.h5")
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        write_h5(str(h5_path), coords, feats)

        stored_coords, stored_feats = read_h5(str(h5_path))
        if not np.array_equal(stored_coords, coords):
            raise RuntimeError(f"HDF5 coordinate readback mismatch: {h5_path}")
        if not np.array_equal(stored_feats, feats):
            raise RuntimeError(f"HDF5 feature readback mismatch: {h5_path}")

        if args.write_ply:
            ply_path = Path(args.output_rootdir) / "ply" / Path(f"{relative_stem}.ply")
            ply_path.parent.mkdir(parents=True, exist_ok=True)
            write_ply_ascii(str(ply_path), coords, feats)

        output_count += 1
        print(f"[{output_count}/{len(input_files)}] {input_path}")
        print(f"Unique voxels: {len(coords)}")
        print(f"Coordinates: dtype={stored_coords.dtype}, shape={stored_coords.shape}")
        print(
            f"Features: dtype={stored_feats.dtype}, shape={stored_feats.shape}, "
            f"RGB range=[{stored_feats.min()}, {stored_feats.max()}]"
        )
        print(f"Colored HDF5: {h5_path}")
        print("Mesh-to-HDF5: PASS")
        if output_count >= args.output_length:
            break


def partition_color(args: argparse.Namespace) -> None:
    import numpy as np

    from data_utils.attribute.inout import read_h5, write_h5, write_ply_ascii
    from data_utils.attribute.partition import kdtree_partition

    input_files = _input_files(args.input_rootdir, args.input_format, args.input_file)
    input_files = input_files[: args.input_length]
    print(f"Input point clouds: {len(input_files)}")

    output_count = 0
    partition_sizes = []
    block_dir = str(args.max_points)
    for input_path in input_files:
        coords, feats = read_h5(input_path)
        points = np.concatenate((coords, feats), axis=1)
        parts = kdtree_partition(points, max_num=args.max_points)

        relative_stem = _relative_stem(input_path, args.input_rootdir)
        for part_index, part in enumerate(parts):
            part = part.copy()
            part[:, :3] -= np.min(part[:, :3], axis=0)
            part_coords = part[:, :3]
            part_feats = part[:, 3:]
            part_name = f"{relative_stem.name}_P{part_index}"
            relative_parent = relative_stem.parent

            h5_path = (
                Path(args.output_rootdir)
                / "h5"
                / block_dir
                / relative_parent
                / f"{part_name}.h5"
            )
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            write_h5(str(h5_path), part_coords, part_feats)

            stored_coords, stored_feats = read_h5(str(h5_path))
            if len(stored_coords) != len(part) or len(stored_feats) != len(part):
                raise RuntimeError(f"Partition HDF5 readback mismatch: {h5_path}")
            if len(part) > args.max_points:
                raise RuntimeError(
                    f"Partition exceeds max_points: {len(part)} > {args.max_points}"
                )

            if args.write_ply:
                ply_path = (
                    Path(args.output_rootdir)
                    / "ply"
                    / block_dir
                    / relative_parent
                    / f"{part_name}.ply"
                )
                ply_path.parent.mkdir(parents=True, exist_ok=True)
                write_ply_ascii(str(ply_path), part_coords, part_feats)

            output_count += 1
            partition_sizes.append(len(part))
            print(
                f"Partition {output_count}: points={len(part)}, "
                f"coords_dtype={stored_coords.dtype}, feats_dtype={stored_feats.dtype}, "
                f"RGB_range=[{stored_feats.min()}, {stored_feats.max()}], file={h5_path}"
            )
            if output_count >= args.output_length:
                print(f"Output partitions: {output_count}")
                print(f"Partition sizes: {partition_sizes}")
                print(f"Max partition size: {max(partition_sizes)}")
                print("KD-tree partition: PASS")
                return

    print(f"Output partitions: {output_count}")
    print(f"Partition sizes: {partition_sizes}")
    if partition_sizes:
        print(f"Max partition size: {max(partition_sizes)}")
    print("KD-tree partition: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RWTT for Unicorn-v1 attribute training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--process",
        choices=("mesh2pc_color", "partition_color"),
        required=True,
    )
    parser.add_argument("--input_rootdir", required=True)
    parser.add_argument("--output_rootdir", required=True)
    parser.add_argument("--input_file")
    parser.add_argument("--input_format", default=None)
    parser.add_argument("--input_length", type=int, default=1_000_000)
    parser.add_argument("--output_length", type=int, default=1_000_000)
    parser.add_argument("--num_points", type=int, default=800_000)
    parser.add_argument("--resolution", type=int, default=1023)
    parser.add_argument("--max_points", type=int, default=100_000)
    parser.add_argument("--write_ply", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.input_format is None:
        args.input_format = "obj" if args.process == "mesh2pc_color" else "h5"
    return args


def main() -> None:
    args = parse_args()
    if args.process == "mesh2pc_color":
        mesh2pc_color(args)
    else:
        partition_color(args)


if __name__ == "__main__":
    main()
