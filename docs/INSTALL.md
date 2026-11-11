# FAU environment and Quick Test

The primary handover target is the **existing compiled FAU environment** used by GPU audit job `1871865` (RTX 3080; 8k, 2k and 4k). Its exact runtime versions are recorded in [fau_environment.json](../configs/fau_environment.json). Reuse that authorized environment, or an administrator-provisioned clone of it, rather than solving a new package combination.

| Component | Verified FAU version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 1.13.1+cu117; built-in CUDA runtime 11.7 |
| FAU module | `cuda/11.8.0` |
| `nvcc` selected after the environment is first on PATH | CUDA 11.7; package `cuda-nvcc=11.7.99` |
| MinkowskiEngine / torchac | 0.5.4 / 0.9.3 |
| NumPy / h5py / pandas | 1.24.4 / 3.10.0 / 1.5.3 |
| tqdm / Ninja / setuptools / pip | 4.66.5 / 1.13.0 / 59.8.0 / 24.3.1 |
| OpenBLAS / libopenblas packages in the environment | 0.3.33 |

The module's toolkit is CUDA 11.8, whereas PyTorch and the environment compiler are CUDA 11.7. This is the observed working FAU setup. Do not replace it with a new CUDA/PyTorch pairing. The installed MinkowskiEngine is reused; its original source-build compiler has not been reconstructed. The minimal runtime excludes PyTorch3D and Open3D.

## Use the provided environment

The environment path is supplied privately with the authorized handover. Run from the extracted repository root:

```bash
module load cuda/11.8.0
export THESIS_ENV=/path/to/provided/unicorn-me-py38
export PATH="$THESIS_ENV/bin:$PATH"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MAX_JOBS=1
export TORCH_EXTENSIONS_DIR="${THESIS_TORCH_CACHE:-$HOME/.cache/scalable_attribute/torch_extensions}"
python -c 'import torch, MinkowskiEngine, torchac; print(torch.__version__, MinkowskiEngine.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The first torchac import may compile its small C++ extension. GCC/G++ 13.3.0 and Ninja 1.13.0 are available on the checked FAU frontend; the handover preparation checks an isolated extension cache. This does not rebuild MinkowskiEngine.

Inference intended for GPU comparison must run inside a GPU allocation. For the verified FAU partition, an interactive shell can be requested with:

```bash
srun --immediate=5 --nodes=1 --ntasks=1 --partition=rtx3080 \
  --gres=gpu:1 --cpus-per-task=4 --time=00:05:00 --pty bash -l
```

Run the environment setup above inside that shell. Request exactly one task; FAU assigns GPU-job memory automatically, so omit `--mem`. The immediate request exits if resources are busy. A login-node run may fall back to CPU and is not a replacement for GPU measurements.

## Supply a model and run

The seven files are `ours_512.pth`, `ours_1k.pth`, `ours_2k.pth`, `ours_4k.pth`, `ours_8k.pth`, `ours_16k.pth`, and `ours_32k.pth`. They total 952,005,347 bytes. Authorized local supply on FAU is the current delivery method; public model URLs remain unset pending permission and acceptance. Put one model under `checkpoints/` and verify it:

```bash
python -m scalable_attribute.download --point 8k --output checkpoints
python -m scalable_attribute.evaluate --checkpoint checkpoints/ours_8k.pth \
  --input examples/example.ply --output outputs/example \
  --gpcc third_party/tmc3_v21 --pc-error third_party/pc_error_d
```

The downloader verifies an existing local file against [the manifest](../configs/checkpoints.json); no network download is needed. The output directory must be new. The command writes `base.ply`, `full.ply`, payload containers and `rates.json`. Each checkpoint supplies its complete Prefix/Base/EL state; no historical or external Unicorn checkpoint is loaded.

The example is synthetic. CPU/GPU reconstructions and Enhancement bytes are not bit-exact interchangeable. Use one backend for encoding and decoding, and retain the original GPU protocol for thesis comparisons. See [FINAL_AUDIT.md](FINAL_AUDIT.md).

## Codec tools

Use the included Linux x86-64 `third_party/tmc3_v21` and `third_party/pc_error_d` by default. Their file identities are in [tools.json](../third_party/tools.json), and their notices are in [THIRD_PARTY.md](../THIRD_PARTY.md). Both run on the checked FAU system using its standard C++/C runtime; `pc_error_d` also uses the OpenMP runtime. The legacy tmc3 help command returns 1 after printing usage; that alone is not a codec failure.

The pinned upstream build commands below are checked separately on FAU (CMake 3.28.3, GCC/G++ 13.3.0). Build outside the handover tree and keep the supplied tools unchanged:

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

The resulting executables are `$THESIS_BUILD_ROOT/tmc13-build/tmc3/tmc3` and `$THESIS_BUILD_ROOT/dmetric/test/pc_error`. Pass them with `--gpcc` and `--pc-error` respectively. The source refs resolve to `f2f1c8e7215f70dd43c99db7df6075b1da0abea6` and `f54b6ca30507c8fa594fe79ed9bf1cf0fed4985c`. Rebuild provenance and the limited quick-test check are recorded with the tool metadata; the full formal matrix is not rerun for rebuilt tools.

## Installation scope

`requirements.txt` pins the observed Python runtime packages. Python, PyTorch/CUDA and compiled MinkowskiEngine are supplied by the tested FAU environment, not installed by that file. A completely fresh dependency solve and MinkowskiEngine rebuild have **not** been certified. The upstream [MinkowskiEngine v0.5.4 instructions](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.5.4) are a reference for a future independently validated rebuild, not an alternative tested installation recipe.

Optional tests in an authorized personal environment clone use `python -m pip install -r requirements-test.txt` followed by `python -m pytest -q`; do not modify the shared tested environment just to install test tools. BD-BR and SVG analysis require only the standard library; CTC preparation and aggregation additionally require NumPy. Results and protocol are indexed in [EVALUATION.md](EVALUATION.md).
