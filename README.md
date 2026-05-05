# FP16 GEMM in Triton

COSC 4397 final project. Mixed-precision FP16 matmul in OpenAI Triton with a switchable accumulator (FP16 vs FP32), benchmarked against cuBLAS and rocBLAS on three GPUs.

## install

```
pip install -r requirements.txt
```

Needs Triton 3.0+ (3.2 on Blackwell). On NVIDIA this comes with a recent PyTorch. On AMD use a ROCm-compatible PyTorch build.

## run

```
python tests/test_correctness.py            # correctness vs torch.matmul
python benchmarks/run_all.py                # full pipeline
python benchmarks/ablation.py               # V1-V5 ablation
python benchmarks/bench_gemm_sweep.py       # fixed-tile sweep
python benchmarks/bench_autotune_sweep.py   # 192-config autotune sweep
python benchmarks/profile_ncu.py            # nsight compute
python benchmarks/generate_plots.py         # figures from result csvs
```

Sizes: N in {512, 1024, 2048, 4096, 8192}. Each point is 10 independent processes, mean +/- std.

## use the kernel

```python
from kernels import triton_matmul, triton_matmul_autotune, make_test_case

a, b = make_test_case(1024, 1024, 1024)
c = triton_matmul(a, b, use_fp32_acc=True)   # default
c = triton_matmul_autotune(a, b)              # 192-config autotune
```

## results

Autotuned kernel vs vendor BLAS at N=8192:

- RTX 5060 Ti (Blackwell): 104% of cuBLAS
- RTX 3080 (Ampere): 98% of cuBLAS
- RX 7900 XTX (RDNA3): ~97% of rocBLAS

Fixed 64x64x32 kernel drops to 46-69% at the same size.

## hardware

- RTX 5060 Ti 16GB, CUDA 12.8, Triton 3.2.0
- RTX 3080 10GB, CUDA 12.6, Triton 3.0.0
- RX 7900 XTX 24GB, ROCm 6.2, Triton 3.0.0

## debug

```
TRITON_INTERPRET=1 python tests/test_correctness.py
compute-sanitizer python tests/test_correctness.py
```
