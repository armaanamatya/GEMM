#!/usr/bin/env python3
# unified gemm runner
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from triton.testing import do_bench

from kernels import (
    get_autotune_best_config,
    make_test_case,
    triton_matmul,
    triton_matmul_autotune,
)

SQUARE_SIZES: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


# helpers
def tflops(n: int, ms: float) -> float:
    return 2.0 * n ** 3 / (ms * 1e-3) / 1e12


def do_bench_ms(fn, warmup: int, rep: int) -> float:
    return float(do_bench(fn, warmup=warmup, rep=rep))


def _log_result(log: logging.Logger, label: str, n: int,
                ms: float | None, tf: float | None) -> None:
    if ms is not None and tf is not None:
        log.info(f"    {label:<24} n={n:5d} : {ms:9.4f} ms  {tf:8.3f} TFLOPS")
    else:
        log.info(f"    {label:<24} n={n:5d} : N/A")


# per-backend
def _bench_cublas_fp16(a16: torch.Tensor, b16: torch.Tensor,
                       warmup: int, rep: int) -> float:
    # cublas/rocblas hgemm
    return do_bench_ms(lambda: torch.matmul(a16, b16), warmup, rep)


def _bench_pytorch_fp32(a32: torch.Tensor, b32: torch.Tensor,
                        warmup: int, rep: int) -> float:
    # tf32 / sgemm
    return do_bench_ms(lambda: torch.matmul(a32, b32), warmup, rep)


def _bench_triton_fixed(a16, b16, warmup, rep, fp32_acc: bool) -> float:
    return do_bench_ms(
        lambda: triton_matmul(a16, b16, use_fp32_acc=fp32_acc), warmup, rep
    )


def _bench_triton_auto(a16, b16, warmup, rep, fp32_acc: bool, log) -> tuple[float, dict]:
    acc_label = "fp32" if fp32_acc else "fp16"
    log.info(f"      [autotuning triton acc={acc_label} ...]")
    ms = do_bench_ms(
        lambda: triton_matmul_autotune(a16, b16, use_fp32_acc=fp32_acc), warmup, rep
    )
    cfg = get_autotune_best_config(a16, b16, use_fp32_acc=fp32_acc)
    return ms, cfg


def _bench_cutlass(a16, b16, warmup, rep, log) -> float | None:
    # cutlass fp16 gemm, 3.x/4.x api
    cutlass = None
    for _import in ("cutlass", "nvidia.cutlass"):
        try:
            import importlib
            cutlass = importlib.import_module(_import)
            break
        except ImportError:
            continue
    if cutlass is None:
        log.warning("      [cutlass package not installed — skipping CUTLASS]")
        log.warning("      [  fix: pip install nvidia-cutlass  ]")
        return None

    version = getattr(cutlass, "__version__", "unknown")
    C = torch.zeros(a16.shape[0], b16.shape[1], device=a16.device, dtype=a16.dtype)
    D = torch.empty_like(C)
    fn = None
    last_exc = None

    # 4.x api
    if fn is None:
        for ctor_kwargs in (
            {"element": torch.float16, "layout": cutlass.LayoutType.RowMajor, "element_accumulator": torch.float32},
            {"element": torch.float16, "layout": cutlass.LayoutType.RowMajor},
        ):
            for run_args, run_kwargs in (
                ((a16, b16, C, D), {"alpha": 1.0, "beta": 0.0}),
                ((a16, b16, C, D), {}),
                ((a16, b16, C),    {}),  # 4.x may return D instead of taking it
            ):
                try:
                    plan = cutlass.op.Gemm(**ctor_kwargs)
                    result = plan.run(*run_args, **run_kwargs)
                    if len(run_args) == 3:
                        fn = lambda: plan.run(a16, b16, C, **run_kwargs)  # noqa: B023
                    else:
                        kw = run_kwargs
                        fn = lambda: plan.run(a16, b16, C, D, **kw)  # noqa: B023
                    break
                except Exception as e:
                    last_exc = e
                    continue
            if fn is not None:
                break

    # 3.x api
    if fn is None:
        try:
            plan = cutlass.Gemm(A=a16, B=b16, C=C, D=D, alpha=1.0, beta=0.0)
            plan.run()
            fn = lambda: plan.run()
        except Exception as e:
            last_exc = e

    if fn is None:
        log.warning(f"      [CUTLASS {version} installed but no compatible API found — skipping]")
        log.warning(f"      [  last error: {last_exc!r}  ]")
        return None

    try:
        return do_bench_ms(fn, warmup, rep)
    except Exception as exc:
        log.warning(f"      [CUTLASS bench failed: {exc!r}]")
        return None
        return None


# per-size
def run_size(n: int, warmup: int, rep: int,
             log: logging.Logger, is_amd: bool, skip_cutlass: bool = False) -> dict:
    a16, b16 = make_test_case(n, n, n, device="cuda", dtype=torch.float16, seed=42)
    a32, b32 = make_test_case(n, n, n, device="cuda", dtype=torch.float32, seed=42)
    row: dict = {"n": n}

    def record(key_ms, key_tf, ms):
        row[key_ms] = ms
        row[key_tf] = tflops(n, ms) if ms is not None else None

    # cublas/rocblas fp16
    ms = _bench_cublas_fp16(a16, b16, warmup, rep)
    record("ms_cublas_fp16", "tflops_cublas_fp16", ms)
    _log_result(log, "cublas_fp16", n, ms, row["tflops_cublas_fp16"])

    # pytorch fp32
    ms = _bench_pytorch_fp32(a32, b32, warmup, rep)
    record("ms_pytorch_fp32", "tflops_pytorch_fp32", ms)
    _log_result(log, "pytorch_fp32", n, ms, row["tflops_pytorch_fp32"])

    # triton fixed fp32
    ms = _bench_triton_fixed(a16, b16, warmup, rep, fp32_acc=True)
    record("ms_triton_fixed_fp32", "tflops_triton_fixed_fp32", ms)
    _log_result(log, "triton_fixed_fp32", n, ms, row["tflops_triton_fixed_fp32"])

    # triton fixed fp16
    ms = _bench_triton_fixed(a16, b16, warmup, rep, fp32_acc=False)
    record("ms_triton_fixed_fp16", "tflops_triton_fixed_fp16", ms)
    _log_result(log, "triton_fixed_fp16", n, ms, row["tflops_triton_fixed_fp16"])

    # triton auto fp32
    ms, cfg_fp32 = _bench_triton_auto(a16, b16, warmup, rep, fp32_acc=True, log=log)
    record("ms_triton_auto_fp32", "tflops_triton_auto_fp32", ms)
    row["best_config_fp32"] = json.dumps(cfg_fp32)
    _log_result(log, "triton_auto_fp32", n, ms, row["tflops_triton_auto_fp32"])
    log.info(f"      best_config_fp32 = {cfg_fp32}")

    # triton auto fp16
    ms, cfg_fp16 = _bench_triton_auto(a16, b16, warmup, rep, fp32_acc=False, log=log)
    record("ms_triton_auto_fp16", "tflops_triton_auto_fp16", ms)
    row["best_config_fp16"] = json.dumps(cfg_fp16)
    _log_result(log, "triton_auto_fp16", n, ms, row["tflops_triton_auto_fp16"])
    log.info(f"      best_config_fp16 = {cfg_fp16}")

    # cutlass
    if is_amd or skip_cutlass:
        row["ms_cutlass"] = None
        row["tflops_cutlass"] = None
        reason = "AMD/ROCm" if is_amd else "disabled via --no-cutlass"
        log.info(f"    {'cutlass':<24} n={n:5d} : N/A ({reason})")
    else:
        ms = _bench_cutlass(a16, b16, warmup, rep, log)
        record("ms_cutlass", "tflops_cutlass", ms)
        _log_result(log, "cutlass", n, ms, row["tflops_cutlass"])

    return row


# csv
def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# summary
_BACKENDS = [
    ("cublas_fp16",       "ms_cublas_fp16",       "tflops_cublas_fp16"),
    ("pytorch_fp32",      "ms_pytorch_fp32",       "tflops_pytorch_fp32"),
    ("triton_fixed_fp32", "ms_triton_fixed_fp32",  "tflops_triton_fixed_fp32"),
    ("triton_fixed_fp16", "ms_triton_fixed_fp16",  "tflops_triton_fixed_fp16"),
    ("triton_auto_fp32",  "ms_triton_auto_fp32",   "tflops_triton_auto_fp32"),
    ("triton_auto_fp16",  "ms_triton_auto_fp16",   "tflops_triton_auto_fp16"),
    ("cutlass",           "ms_cutlass",            "tflops_cutlass"),
]


def print_summary(log: logging.Logger, rows: list[dict]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("  Summary — TFLOPS per backend")
    log.info("=" * 72)
    header = f"{'Backend':<24}" + "".join(f"{n:>9}" for n in SQUARE_SIZES)
    log.info(header)
    log.info("-" * 72)
    for label, _, tf_key in _BACKENDS:
        vals = ""
        for row in rows:
            v = row.get(tf_key)
            vals += f"{v:>9.2f}" if v is not None else f"{'N/A':>9}"
        log.info(f"{label:<24}{vals}")
    log.info("=" * 72)


# main
def main() -> None:
    p = argparse.ArgumentParser(description="Unified GEMM benchmark runner")
    p.add_argument("--gpu-name",    required=True, help="GPU label, e.g. '3080'")
    p.add_argument("--run-id",      required=True, help="Run number string, e.g. '01'")
    p.add_argument("--output-dir",  required=True, help="Directory for this run's outputs")
    p.add_argument("--warmup", type=int, default=25,  help="do_bench warmup iters")
    p.add_argument("--rep",    type=int, default=100, help="do_bench timed iters")
    p.add_argument("--amd",        action="store_true", help="ROCm mode: skip CUTLASS")
    p.add_argument("--no-cutlass", action="store_true", help="skip CUTLASS benchmark")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_path = out / "run.log"
    log = logging.getLogger("run_all")
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)

    if not torch.cuda.is_available():
        log.error("No CUDA/ROCm device found — aborting")
        sys.exit(1)

    cuda_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    meta = {
        "gpu_name":       args.gpu_name,
        "run_id":         args.run_id,
        "cuda_device":    cuda_name,
        "vram_gb":        round(props.total_memory / 1024**3, 2),
        "cuda_version":   torch.version.cuda or "N/A",
        "torch_version":  torch.__version__,
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "warmup":         args.warmup,
        "rep":            args.rep,
        "amd_mode":       args.amd,
        "sizes":          list(SQUARE_SIZES),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log.info("=" * 60)
    log.info(f"  GEMM Benchmark  |  GPU: {args.gpu_name}  |  Run: {args.run_id}")
    log.info("=" * 60)
    for k, v in meta.items():
        log.info(f"  {k}: {v}")
    log.info("")

    rows: list[dict] = []
    for n in SQUARE_SIZES:
        log.info(f"--- n = {n} " + "-" * 30)
        try:
            row = run_size(n, args.warmup, args.rep, log, is_amd=args.amd,
                           skip_cutlass=args.no_cutlass)
            rows.append(row)
        except Exception as exc:
            log.error(f"  FAILED n={n}: {exc}")
            log.error(traceback.format_exc())

    if rows:
        print_summary(log, rows)
        csv_path = out / "results.csv"
        write_csv(csv_path, rows)
        log.info(f"\nResults  → {csv_path}")

    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"Metadata → {out / 'meta.json'}")
    log.info(f"Log      → {log_path}")
    log.info("=== Run complete ===\n")


if __name__ == "__main__":
    main()
