# Environment and installation

The project targets Linux x86-64 with a CUDA-capable GPU for codec evaluation. No personal HPC directory, environment script or cluster account is required. Plotting and BD-BR analysis use the Python standard library and run without model weights or CUDA.

## Validated versions

[environment.json](../configs/environment.json) records the actual tested runtime:

| Component | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 1.13.1+cu117; CUDA runtime 11.7 |
| MinkowskiEngine / torchac | 0.5.4 / 0.9.3 |
| NumPy / h5py / pandas | 1.24.4 / 3.10.0 / 1.5.3 |
| tqdm / Ninja / setuptools / pip | 4.66.5 / 1.13.0 / 59.8.0 / 24.3.1 |

The validated precompiled environment also had a CUDA 11.8.89 toolkit available and an environment `nvcc` package 11.7.99. These observations distinguish toolkit and PyTorch runtime versions; they are not a requirement to install two toolkits. Existing compiled-environment execution, fresh torchac cache compilation and the codec-tool builds were tested. A completely fresh dependency solve and MinkowskiEngine rebuild have **not** been certified. The original compiler used to build MinkowskiEngine is not recorded.

## General Linux setup

The following is a reproducible version-pinning recipe, not a claim of a tested clean installation. Install a CUDA toolkit and C/C++ compiler combination supported by the selected PyTorch/CUDA and MinkowskiEngine versions, plus OpenBLAS development headers and Git. Use an NVIDIA driver compatible with the CUDA 11.7 PyTorch build. Consult [MinkowskiEngine v0.5.4 installation instructions](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.5.4) for compiler and BLAS configuration.

```bash
conda create -n scalable-attribute python=3.8.20 pip=24.3.1 -y
conda activate scalable-attribute
python -m pip install setuptools==59.8.0
python -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Install the matching MinkowskiEngine source with your compatible local CUDA/compiler settings:

```bash
THESIS_ME_ROOT=$(mktemp -d)
git clone --depth 1 --branch v0.5.4 \
  https://github.com/NVIDIA/MinkowskiEngine.git "$THESIS_ME_ROOT/MinkowskiEngine"
(cd "$THESIS_ME_ROOT/MinkowskiEngine" && python setup.py install --blas=openblas --force_cuda)
python -c 'import torch, MinkowskiEngine, torchac; print(torch.__version__, MinkowskiEngine.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The tag resolves to `9f81ae66b33b883cd08ee4f64d08cf633608b118`. GCC 13.3.0 was used for the separate CPU tool/torchac builds below; it is not a certified CUDA compiler configuration for rebuilding MinkowskiEngine. CUDA toolkit/compiler paths are machine configuration, not repository settings. PyTorch3D and Open3D are unnecessary for the final resnet inference path.

The first torchac import may compile a small C++ extension. On hosts with limited build resources, set `MAX_JOBS=1` and choose a writable `TORCH_EXTENSIONS_DIR`. Runtime import failure should be resolved before attempting model evaluation.

## tmc3 and pc_error

Use the included Linux binaries `third_party/tmc3_v21` and `third_party/pc_error_d`, identified by [tools.json](../third_party/tools.json). They require standard C++/C shared libraries; `pc_error_d` also uses OpenMP. The legacy tmc3 help invocation prints usage and returns 1, while pc_error help returns 0.

The following pinned builds were executed successfully with CMake 3.28.3 and GCC/G++ 13.3.0. Keep upstream checkout/build directories outside this repository:

```bash
THESIS_BUILD_ROOT=$(mktemp -d)
git clone --depth 1 --branch release-v21.0-rc1 \
  https://github.com/MPEGGroup/mpeg-pcc-tmc13.git "$THESIS_BUILD_ROOT/tmc13"
cmake -S "$THESIS_BUILD_ROOT/tmc13" -B "$THESIS_BUILD_ROOT/tmc13-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$THESIS_BUILD_ROOT/tmc13-build" --parallel 2
git clone --depth 1 --branch release-v0.13.4 \
  https://github.com/MPEGGroup/mpeg-pcc-dmetric.git "$THESIS_BUILD_ROOT/dmetric"
cmake -S "$THESIS_BUILD_ROOT/dmetric/source" -B "$THESIS_BUILD_ROOT/dmetric-build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$THESIS_BUILD_ROOT/dmetric-build" --parallel 2
```

Outputs are `$THESIS_BUILD_ROOT/tmc13-build/tmc3/tmc3` and `$THESIS_BUILD_ROOT/dmetric/test/pc_error`. Pass these paths using `--gpcc` and `--pc-error`. Source commits are `f2f1c8e7215f70dd43c99db7df6075b1da0abea6` and `f54b6ca30507c8fa594fe79ed9bf1cf0fed4985c`. A 512-point CPU example produced the same physical bits and quality as the supplied tools. The full formal matrix was not rerun with rebuilt binaries; preserve version and backend when comparing with frozen measurements. Tool notices are in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Models and evaluation

Seven complete checkpoints are supplied as external [GitHub Release assets](https://github.com/whitesocksjames/scalableattributethesis/releases/tag/thesis-handover-v1), keeping ordinary Git files small. No personal HPC directory is part of the installation procedure. Datasets are not bundled.

Download a model and run the Quick Test from the cloned repository root:

```bash
python -m scalable_attribute.download --point 8k --output checkpoints
python -m scalable_attribute.evaluate \
  --checkpoint checkpoints/ours_8k.pth \
  --input examples/example.ply --output outputs/example \
  --gpcc third_party/tmc3_v21 --pc-error third_party/pc_error_d
```

The downloader checks the expected file size and SHA-256 before accepting a download or reusing a local file. Use `--point all` for all seven, then independently check `(cd checkpoints && sha256sum -c ../configs/SHA256SUMS)`. [checkpoints.json](../configs/checkpoints.json) contains exact filenames, sizes and hashes. [CHECKPOINTS.md](CHECKPOINTS.md) records platform availability; the loader format is defined by `scalable_attribute/runtime/checkpoints.py`.

The output directory must be new. Use one backend for encoding and decoding; CPU/GPU bit-exact equivalence and payload interchange are not certified. The synthetic example exercises the interface, while [EVALUATION.md](EVALUATION.md) defines the final benchmark protocol.

For tests, install `requirements-test.txt` in your own environment and run `python -m pytest -q`. Tests requiring PyTorch/torchac are skipped when those packages are absent. CTC preprocessing and aggregation require NumPy; result analysis requires only the standard library.
