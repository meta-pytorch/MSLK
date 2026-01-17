# MSLK Library


MSLK (Meta Superintelligence Labs Kernels, formerly known as **[FBGEMM GenAI](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai)**)
is a collection of high-performance kernels and optimizations built on top of PyTorch
primitives for GenAI training and inference.

## **Installation**

```bash
# Install MSLK for CUDA
pip install mslk-cuda==1.0.0
# Install MSLK for ROCm
pip install mslk-rocm==1.0.0
# Install a nightly version
pip3 install --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Release Compatibility Table

MSLK is released in accordance to the PyTorch release schedule, and each
release has no guarantee to work in conjunction with PyTorch releases that are
older than the one that the MSLK release corresponds to.

| MSLK Release | Corresponding PyTorch Release | Supported Python Versions | Supported CUDA Versions | Supported CUDA Architectures | Supported ROCm Versions | Supported ROCm Architectures |
|---------|---------|---------|---------|----------|-------------|-------------|
| 1.0.0 | 2.10.x | 3.10, 3.11, 3.12, 3.13, 3.14 | 12.6, 12.8, 12.9, 13.0 | 8.0, 9.0a, 10.0a, 12.0a | 7.0, 7.1 | gfx908, gfx90a, gfx942, gfx950 |

## **Running Benchmarks**
```bash
python bench/gemm/gemm_bench.py
python bench/quantize/quantize_bench.py
```

## **Running Tests**
```bash
python test/gemm/gemm_test.py
python test/gemm/quantize_test.py
```

## **Build From Source**
We only support building on Linux. See the release compatibility table above for supported versions of Python, CUDA, ROCm.
```bash
# Clone repo
git clone https://github.com/meta-pytorch/MSLK
cd MSLK
git submodule sync
git submodule update --init --recursive
# Build and install
# The script will create a conda environment and install the required dependencies.
# The conda environment will look something like: build-py3.14-torchnightly-cuda12.9.1
./ci/integration/mslk_oss_build.bash
# After the initial environment setup, you can activate the environment and iterate faster:
conda activate build-py3.14-torchnightly-cuda12.9.1
python setup.py install
```

## Join the MSLK community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/meta-pytorch/MSLK/issues)

For contributions, please see the [`CONTRIBUTING`](./CONTRIBUTING.md) file for
ways to help out.

## License

MSLK is BSD licensed, as found in the [`LICENSE`](./LICENSE) file.
