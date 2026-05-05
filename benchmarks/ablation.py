#!/usr/bin/env python3
# ablation: fixed -> autotuned, one knob at a time
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

from kernels import triton_matmul, triton_matmul_autotune, make_test_case


# swizzled kernel, explicit config
@triton.jit
def _matmul_swizzled(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _call_swizzled(a, b, bm, bn, bk, gm, num_warps, num_stages):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _matmul_swizzled[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# bench helpers
WARMUP = 25
REP = 100


def bench_ms(fn, warmup=WARMUP, rep=REP):
    return float(do_bench(fn, warmup=warmup, rep=rep))


def tflops(n, ms):
    return 2.0 * n**3 / (ms * 1e-3) / 1e12


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# main
SIZES = [512, 1024, 2048, 4096, 8192]

VARIANTS = [
    ("V1  naive_64x64   ", "BLOCK=64×64,  no swizzle, stages=2"),
    ("V2  tiles_128x256 ", "BLOCK=128×256, no swizzle, stages=2"),
    ("V3  +swizzle      ", "BLOCK=128×256, GROUP_M=4,  stages=2"),
    ("V4  +pipeline     ", "BLOCK=128×256, GROUP_M=4,  stages=3"),
    ("V5  full_auto     ", "Triton autotuner"),
    ("cuBLAS            ", "torch.matmul fp16"),
]


def run_variant(label, n, a16, b16, warmup, rep):
    if "naive_64x64" in label:
        fn = lambda: triton_matmul(
            a16, b16, use_fp32_acc=True, block_m=64, block_n=64, block_k=32
        )
    elif "tiles_128x256" in label:
        # bigger tiles
        fn = lambda: triton_matmul(
            a16, b16, use_fp32_acc=True, block_m=128, block_n=256, block_k=32
        )
    elif "+swizzle" in label:
        fn = lambda: _call_swizzled(
            a16, b16, 128, 256, 32, 4, num_warps=4, num_stages=2
        )
    elif "+pipeline" in label:
        fn = lambda: _call_swizzled(
            a16, b16, 128, 256, 32, 4, num_warps=8, num_stages=3
        )
    elif "full_auto" in label:
        fn = lambda: triton_matmul_autotune(a16, b16, use_fp32_acc=True)
    else:  # cublas
        fn = lambda: torch.matmul(a16, b16)
    return bench_ms(fn, warmup, rep)


def main():
    p = argparse.ArgumentParser(description="GEMM ablation study")
    p.add_argument("--gpu-name", default="", help="GPU label, e.g. '5060ti'")
    p.add_argument("--run-id", default="01", help="Run number string, e.g. '01'")
    p.add_argument("--output-dir", default="", help="Directory for this run's outputs")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    args = p.parse_args()

    out = None
    log = None
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "run.log"
        log = logging.getLogger("ablation")
        log.setLevel(logging.INFO)
        log.propagate = False
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)

        meta = {
            "gpu_name": args.gpu_name,
            "run_id": args.run_id,
            "cuda_device": torch.cuda.get_device_name(0),
            "vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            ),
            "cuda_version": torch.version.cuda or "N/A",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "warmup": args.warmup,
            "rep": args.rep,
            "sizes": list(SIZES),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        log.info("=" * 60)
        log.info(f"  Ablation Study  |  GPU: {args.gpu_name}  |  Run: {args.run_id}")
        log.info("=" * 60)
        for k, v in meta.items():
            log.info(f"  {k}: {v}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found")
        sys.exit(1)

    gpu = args.gpu_name or torch.cuda.get_device_name(0).split()[-1]
    print(f"\nAblation Study — {torch.cuda.get_device_name(0)}")
    print(f"Warmup={args.warmup}  Rep={args.rep}  FP32 accumulator\n")

    # legend
    for label, desc in VARIANTS:
        print(f"  {label.strip():<22} {desc}")
    print()

    # header
    hdr = f"{'Variant':<22}" + "".join(f"  n={n:>5}" for n in SIZES)
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    cublas_tflops = {}  # %cublas col

    results = {}
    for label, _ in VARIANTS:
        row = f"{label.strip():<22}"
        for n in SIZES:
            a16, b16 = make_test_case(
                n, n, n, device="cuda", dtype=torch.float16, seed=42
            )
            ms = run_variant(label, n, a16, b16, args.warmup, args.rep)
            tf = tflops(n, ms)
            results[(label, n)] = tf
            if "cuBLAS" in label:
                cublas_tflops[n] = tf
            row += f"  {tf:>7.2f}"
            if log:
                log.info(
                    f"    {label.strip():<22} n={n:5d}: {ms:9.4f} ms  {tf:8.3f} TFLOPS"
                )
        print(row)

    # %cublas rows
    print(sep)
    print(f"{'% of cuBLAS':<22}" + "".join(f"  {'n=' + str(n):>7}" for n in SIZES))
    print(sep)
    for label, _ in VARIANTS:
        row = f"{label.strip():<22}"
        for n in SIZES:
            tf = results[(label, n)]
            pct = 100.0 * tf / cublas_tflops[n] if cublas_tflops.get(n) else 0
            row += f"  {pct:>6.1f}%"
        print(row)

    # incremental gains
    print(sep)
    print("\nIncremental TFLOPS gain at each step (n=4096 and n=8192):\n")
    prev_label = None
    for label, _ in VARIANTS:
        if prev_label is not None:
            for n in [4096, 8192]:
                gain = results[(label, n)] - results[(prev_label, n)]
                pct = 100.0 * gain / results[(prev_label, n)]
                print(
                    f"  {prev_label.strip()} → {label.strip():<22}  n={n}: {gain:+.2f} TFLOPS ({pct:+.1f}%)"
                )
        prev_label = label

    print()

    if out and log and meta:
        csv_rows = []
        for label, _ in VARIANTS:
            for n in SIZES:
                tf = results.get((label, n))
                csv_rows.append(
                    {
                        "variant": label.strip(),
                        "n": n,
                        "tflops": tf if tf is not None else "",
                    }
                )
        write_csv(out / "results.csv", csv_rows)
        log.info(f"Results  -> {out / 'results.csv'}")

        meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info(f"Metadata -> {out / 'meta.json'}")
        log.info(f"Log      -> {log_path}")
        log.info("=== Run complete ===")


if __name__ == "__main__":
    main()
