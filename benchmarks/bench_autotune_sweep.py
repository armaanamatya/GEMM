# autotuned vs fixed-tile vs cublas
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from triton.testing import do_bench

from kernels import (
    make_test_case,
    pytorch_matmul,
    triton_matmul,
    triton_matmul_autotune,
    get_autotune_best_config,
)

SQUARE_SIZES: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def bench_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    return float(do_bench(fn, warmup=warmup, rep=rep))


def run_row(n: int, warmup: int, rep: int) -> dict:
    a, b = make_test_case(n, n, n, device="cuda", dtype=torch.float16, seed=0)

    # auto fp32
    ms_auto_fp32 = bench_ms(lambda: triton_matmul_autotune(a, b, use_fp32_acc=True), warmup, rep)
    best_fp32 = get_autotune_best_config(a, b, use_fp32_acc=True)

    # auto fp16
    ms_auto_fp16 = bench_ms(lambda: triton_matmul_autotune(a, b, use_fp32_acc=False), warmup, rep)
    best_fp16 = get_autotune_best_config(a, b, use_fp32_acc=False)

    # fixed 64x64x32
    ms_fixed_fp32 = bench_ms(lambda: triton_matmul(a, b, use_fp32_acc=True), warmup, rep)
    ms_fixed_fp16 = bench_ms(lambda: triton_matmul(a, b, use_fp32_acc=False), warmup, rep)

    # cublas
    ms_cublas = bench_ms(lambda: pytorch_matmul(a, b), warmup, rep)

    tf_auto_fp32 = tflops(n, n, n, ms_auto_fp32)
    tf_auto_fp16 = tflops(n, n, n, ms_auto_fp16)
    tf_fixed_fp32 = tflops(n, n, n, ms_fixed_fp32)
    tf_fixed_fp16 = tflops(n, n, n, ms_fixed_fp16)
    tf_cublas = tflops(n, n, n, ms_cublas)

    return {
        "n": n,
        "tflops_auto_fp32": tf_auto_fp32,
        "tflops_auto_fp16": tf_auto_fp16,
        "tflops_fixed_fp32": tf_fixed_fp32,
        "tflops_fixed_fp16": tf_fixed_fp16,
        "tflops_cublas": tf_cublas,
        "pct_cublas_auto_fp32": tf_auto_fp32 / tf_cublas * 100,
        "pct_cublas_auto_fp16": tf_auto_fp16 / tf_cublas * 100,
        "speedup_auto_vs_fixed_fp32": tf_auto_fp32 / tf_fixed_fp32,
        "speedup_auto_vs_fixed_fp16": tf_auto_fp16 / tf_fixed_fp16,
        "best_fp32_config": json.dumps(best_fp32),
        "best_fp16_config": json.dumps(best_fp16),
    }


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Autotuned GEMM benchmark sweep")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  |  torch={torch.__version__}  |  warmup={args.warmup}  rep={args.rep}")
    print()

    header = (
        f"{'Size':>6} | {'Auto FP32':>10} | {'Auto FP16':>10} | {'Fixed FP32':>11} | "
        f"{'Fixed FP16':>11} | {'cuBLAS':>8} | {'Auto/Fix32':>10} | {'Auto/Fix16':>10} | "
        f"{'%cuBLAS32':>9} | Best FP32 Config"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n in SQUARE_SIZES:
        print(f"  Autotuning n={n}... ", end="", flush=True)
        row = run_row(n, args.warmup, args.rep)
        rows.append(row)
        print("done")
        print(
            f"{row['n']:>6} | "
            f"{row['tflops_auto_fp32']:>8.2f} T | "
            f"{row['tflops_auto_fp16']:>8.2f} T | "
            f"{row['tflops_fixed_fp32']:>9.2f} T | "
            f"{row['tflops_fixed_fp16']:>9.2f} T | "
            f"{row['tflops_cublas']:>6.2f} T | "
            f"{row['speedup_auto_vs_fixed_fp32']:>9.2f}x | "
            f"{row['speedup_auto_vs_fixed_fp16']:>9.2f}x | "
            f"{row['pct_cublas_auto_fp32']:>7.1f} % | "
            f"{row['best_fp32_config']}"
        )

    # summary
    print("\n" + "=" * 60)
    print("  Autotuning Summary")
    print("=" * 60)
    for row in rows:
        cfg = json.loads(row["best_fp32_config"])
        print(
            f"  n={row['n']:>5}: best tile {cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}  "
            f"warps={cfg['num_warps']}  stages={cfg['num_stages']}  group_m={cfg['GROUP_M']}  "
            f"→ {row['speedup_auto_vs_fixed_fp32']:.2f}x speedup over fixed 64x64x32"
        )

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
