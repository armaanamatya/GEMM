"""
Nsight Compute profiling helper.

Designed to be called via ncu, not directly:

  /opt/nvidia/nsight-compute/2026.1.0/ncu --set full \
      --target-processes all \
      -o benchmarks/results/profile_triton_fp32 \
      python benchmarks/profile_ncu.py --mode triton_fp32

  /opt/nvidia/nsight-compute/2026.1.0/ncu --set full \
      --target-processes all \
      -o benchmarks/results/profile_triton_fp16 \
      python benchmarks/profile_ncu.py --mode triton_fp16

  /opt/nvidia/nsight-compute/2026.1.0/ncu --set full \
      --target-processes all \
      -o benchmarks/results/profile_cublas \
      python benchmarks/profile_ncu.py --mode cublas

Modes:
  triton_fp32  — Triton GEMM with FP32 accumulation
  triton_fp16  — Triton GEMM with FP16 accumulation
  cublas       — torch.matmul (cuBLAS baseline)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from kernels import make_test_case, pytorch_matmul, triton_matmul

DEFAULT_SIZE = 4096


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single GEMM for ncu profiling")
    p.add_argument("--mode", choices=["triton_fp32", "triton_fp16", "cublas"], required=True)
    p.add_argument("--n", type=int, default=DEFAULT_SIZE, help="Square matrix size (M=N=K)")
    args = p.parse_args()

    n = args.n
    a, b = make_test_case(n, n, n, device="cuda", dtype=torch.float16, seed=0)

    # Warmup (not profiled if ncu uses --target-processes all with launch skip)
    if args.mode == "triton_fp32":
        triton_matmul(a, b, use_fp32_acc=True)
    elif args.mode == "triton_fp16":
        triton_matmul(a, b, use_fp32_acc=False)
    else:
        pytorch_matmul(a, b)

    torch.cuda.synchronize()

    # Profiled run
    if args.mode == "triton_fp32":
        c = triton_matmul(a, b, use_fp32_acc=True)
    elif args.mode == "triton_fp16":
        c = triton_matmul(a, b, use_fp32_acc=False)
    else:
        c = pytorch_matmul(a, b)

    torch.cuda.synchronize()
    print(f"Done: mode={args.mode}, n={n}, output_shape={c.shape}")


if __name__ == "__main__":
    main()
