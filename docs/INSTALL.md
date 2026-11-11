# Installation

Use Linux, a C++ compiler, and a CUDA-capable GPU for the original evaluation environment. The retained `tmc3_v21` and `pc_error_d` executables are Linux x86-64 binaries from the original evaluation tree.

The existing evaluation environment contains Python 3.8.20, PyTorch 1.13.1+cu117, CUDA 11.7, MinkowskiEngine 0.5.4, torchac 0.9.3, NumPy 1.24.4, h5py 3.10.0, and pandas 1.5.3. Full codec validation status is recorded in [VALIDATION.md](VALIDATION.md). The following recipe reconstructs these dependency versions; a fresh installation has not been tested on every platform.

```bash
conda create -n scalable-attribute python=3.8 pip -y
conda activate scalable-attribute
python -m pip install 'setuptools==59.8.0'
python -m pip install 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Build MinkowskiEngine against this PyTorch and a matching CUDA toolkit. Install the system OpenBLAS development package and a CUDA-compatible compiler first. The original environment used GCC 11.2 and CUDA 11.7.

```bash
git clone --branch v0.5.4 --depth 1 https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine
(cd /tmp/MinkowskiEngine && python setup.py install --blas=openblas --force_cuda)
python -c 'import torch, MinkowskiEngine, torchac; print(torch.__version__, torch.cuda.is_available())'
```

MinkowskiEngine v0.5.4 is pinned to upstream commit `9f81ae66b33b883cd08ee4f64d08cf633608b118`. See its [installation documentation](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.5.4) for compiler and BLAS options. The final resnet path does not require PyTorch3D or Open3D.

From the repository root, place a supplied model under `checkpoints/`, or use the download command after release assets are published. The downloader also verifies a model already present at its expected path.

```bash
python -m scalable_attribute.download --point 8k --output checkpoints
python -m scalable_attribute.evaluate --checkpoint checkpoints/ours_8k.pth \
  --input examples/example.ply --output outputs/example \
  --gpcc third_party/tmc3_v21 --pc-error third_party/pc_error_d
```

Each model is self-contained. Base and Full require no external Unicorn checkpoint, training checkpoint, or machine-specific directory. The example is synthetic and demonstrates the interface; thesis measurements use the frozen sample identities and preprocessing in [EVALUATION.md](EVALUATION.md).

For checks and result analysis:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The BD-BR calculation and SVG plotting scripts use only the Python standard library. CTC chunk preparation and aggregation require NumPy. Codec execution additionally requires PyTorch, MinkowskiEngine, torchac, and the two codec/metric tools. To rebuild tools on another platform, use the pinned upstream versions and retain their notices from [THIRD_PARTY.md](../THIRD_PARTY.md).
