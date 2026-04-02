"""
Square GEMM performance sweep: Triton (FP32 / FP16 accumulation) vs cuBLAS (torch.matmul).

Uses triton.testing.do_bench for warmup + median timing. Run from repo root:

  python benchmarks/bench_gemm_sweep.py
  python benchmarks/bench_gemm_sweep.py --csv benchmarks/results/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from triton.testing import do_bench

from kernels import make_test_case, pytorch_matmul, triton_matmul

SQUARE_SIZES: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


def flops_matmul(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def tflops_from_seconds(m: int, n: int, k: int, seconds: float) -> float:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return flops_matmul(m, n, k) / seconds / 1e12


def bench_median_ms(fn, warmup: int, rep: int) -> float:
    """Median wall time in ms for one call to ``fn`` (CUDA sync included)."""
    return float(do_bench(fn, warmup=warmup, rep=rep))


def run_row(n: int, warmup: int, rep: int) -> dict[str, float]:
    a, b = make_test_case(n, n, n, device="cuda", dtype=torch.float16, seed=0)

    ms_triton_fp32 = bench_median_ms(
        lambda: triton_matmul(a, b, use_fp32_acc=True),
        warmup=warmup,
        rep=rep,
    )
    ms_triton_fp16 = bench_median_ms(
        lambda: triton_matmul(a, b, use_fp32_acc=False),
        warmup=warmup,
        rep=rep,
    )
    ms_cublas = bench_median_ms(
        lambda: pytorch_matmul(a, b),
        warmup=warmup,
        rep=rep,
    )

    s32 = ms_triton_fp32 / 1000.0
    s16 = ms_triton_fp16 / 1000.0
    s_blas = ms_cublas / 1000.0

    tf32 = tflops_from_seconds(n, n, n, s32)
    tf16 = tflops_from_seconds(n, n, n, s16)
    tf_blas = tflops_from_seconds(n, n, n, s_blas)

    return {
        "n": float(n),
        "ms_triton_fp32_acc": ms_triton_fp32,
        "ms_triton_fp16_acc": ms_triton_fp16,
        "ms_cublas": ms_cublas,
        "tflops_triton_fp32_acc": tf32,
        "tflops_triton_fp16_acc": tf16,
        "tflops_cublas": tf_blas,
        "pct_peak_vs_cublas_fp32": (tf32 / tf_blas) * 100.0 if tf_blas > 0 else 0.0,
        "pct_peak_vs_cublas_fp16": (tf16 / tf_blas) * 100.0 if tf_blas > 0 else 0.0,
    }


def write_csv(path: str, rows: list[dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="GEMM benchmark sweep (Triton vs torch.matmul / cuBLAS)")
    p.add_argument("--warmup", type=int, default=25, help="do_bench warmup iterations")
    p.add_argument("--rep", type=int, default=100, help="do_bench timed repetitions")
    p.add_argument("--csv", type=str, default="", help="optional path to write CSV results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    print(
        "device=",
        torch.cuda.get_device_name(0),
        "torch=",
        torch.__version__,
        "warmup=",
        args.warmup,
        "rep=",
        args.rep,
    )
    print()

    rows: list[dict[str, float]] = []
    for n in SQUARE_SIZES:
        row = run_row(n=n, warmup=args.warmup, rep=args.rep)
        rows.append(row)
        print(
            f"n={int(row['n']):5d}  "
            f"ms[triton fp32 acc]={row['ms_triton_fp32_acc']:.4f}  "
            f"ms[triton fp16 acc]={row['ms_triton_fp16_acc']:.4f}  "
            f"ms[cublas]={row['ms_cublas']:.4f}"
        )
        print(
            f"          TFLOPS triton_fp32={row['tflops_triton_fp32_acc']:.3f}  "
            f"triton_fp16={row['tflops_triton_fp16_acc']:.3f}  "
            f"cublas={row['tflops_cublas']:.3f}  "
            f"%cublas(fp32)={row['pct_peak_vs_cublas_fp32']:.1f}%  "
            f"%cublas(fp16)={row['pct_peak_vs_cublas_fp16']:.1f}%"
        )

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
