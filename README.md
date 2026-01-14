# MSLK Library


MSLK (Meta Superintelligence Labs Kernels, formerly known as **[FBGEMM GenAI](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai)**)
is a collection of high-performance kernels and optimizations built on top of PyTorch
primitives for GenAI training and inference.

## **Installation**

```bash
# Full MSLK library
pip install mslk-cuda==1.0.0
pip install mslk==1.0.0 --index-url https://download.pytorch.org/whl/cu128
```

### Releases Compatibility Table

MSLK is released in accordance to the PyTorch release schedule, and each
release has no guarantee to work in conjunction with PyTorch releases that are
older than the one that the MSLK release corresponds to.

| MSLK Release | Corresponding PyTorch Release | Supported Python Versions | Supported CUDA Versions | Supported CUDA Architectures | Supported ROCm Versions | Supported ROCm Architectures |
|---------|---------|---------|---------|----------|-------------|-------------|
| 1.0.0 | 2.10.x | 3.10, 3.11, 3.12 3.13, 3.14 | 12.6, 12.8, 12.9, 13.0 | 8.0, 9.0a, 10.0a, 12.0a | 7.0, 7.1 | gfx908, gfx90a, gfx942, gfx950 |

## Join the MSLK community

For questions, support, news updates, or feature requests, please feel free to:

* File a ticket in [GitHub Issues](https://github.com/meta-pytorch/MSLK/issues)

For contributions, please see the [`CONTRIBUTING`](./CONTRIBUTING.md) file for
ways to help out.

## License

MSLK is BSD licensed, as found in the [`LICENSE`](./LICENSE) file.
