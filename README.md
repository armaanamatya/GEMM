# Mixed-Precision FP16 GEMM in Triton

COSC 4397 — Parallel Computing on GPUs — Final Project

## Overview

A tiled FP16 General Matrix Multiply (GEMM) kernel written in OpenAI Triton with a configurable accumulator precision flag (FP16 vs FP32). The project studies the numerical accuracy and performance trade-offs of accumulator precision, benchmarked against cuBLAS-backed `torch.matmul`.

## Project Structure

```
kernels/                    # GPU kernel code
  triton_gemm.py            # Triton GEMM kernel with FP16/FP32 accumulator flag
  baseline_pytorch.py       # cuBLAS reference wrapper (torch.matmul)
  __init__.py               # exports triton_matmul, pytorch_matmul, make_test_case
tests/
  test_correctness.py       # correctness validation — both modes, sizes 512–8192
setup/
  setup_linux.sh            # one-command Linux environment setup
  setup_windows.ps1         # one-command Windows environment setup
  setup_3080.md             # RTX 3080 setup guide
  setup_5060ti.md           # RTX 5060 Ti setup guide
docs/
  plan.md                   # 3-week execution plan with task assignments
  week2_autotuning.md       # Week 2 autotuning analysis and results
  finalprojectguideline.txt # course rubric
  COSC4397_GEMM_Project_Plan.docx
requirements.txt
```

## Quick Start

### 1. Environment setup

**Linux:**
```bash
bash setup/setup_linux.sh
```

**Windows (PowerShell):**
```powershell
.\setup\setup_windows.ps1
```

Or follow the GPU-specific guide in `setup/setup_3080.md` or `setup/setup_5060ti.md`.

### 2. Verify the kernel

```bash
python kernels/triton_gemm.py
```

Expected output: max absolute error for both FP32 and FP16 accumulation modes.

### 3. Run correctness tests

```bash
python tests/test_correctness.py
```

Tests both accumulation modes across square matrices (512, 1024, 2048, 4096, 8192) and non-aligned shapes.

## Kernel Usage

```python
from kernels import triton_matmul, make_test_case

a, b = make_test_case(1024, 1024, 1024)

# FP32 accumulation (default) — higher precision
c = triton_matmul(a, b, use_fp32_acc=True)

# FP16 accumulation — faster, lower precision
c = triton_matmul(a, b, use_fp32_acc=False)

# Autotuned kernel (Week 2) — searches 192 configs for best tile parameters
from kernels import triton_matmul_autotune
c = triton_matmul_autotune(a, b)
```

## What the Study Measures

| Dimension | What we compare |
|---|---|
| **Numerical error** | Max absolute & relative error vs `torch.matmul` — FP16 vs FP32 accumulation across matrix sizes |
| **Performance** | TFLOPS achieved — Triton (both modes) vs cuBLAS baseline |
| **Roofline position** | Achieved FLOPS vs compute/bandwidth ceilings on RTX 3080 |
| **Microarchitecture** | Occupancy, register pressure, shared memory usage via Nsight Compute |

## Week 2 Results — Autotuning & L2 Cache Optimization

Week 2 introduced `@triton.autotune` with an expanded 192-configuration search grid (adding `GROUP_M` for tile grouping) and swizzled program IDs for improved L2 cache locality. These optimizations closed the performance gap at large matrix sizes.

| Size | Fixed TFLOPS | Autotuned TFLOPS | cuBLAS TFLOPS | % of cuBLAS |
|------|-------------|-----------------|---------------|-------------|
| 512  | 17.2        | 18.1            | 19.5          | 92.8%       |
| 1024 | 25.4        | 27.8            | 28.1          | 98.9%       |
| 2048 | 27.1        | 29.3            | 29.6          | 98.9%       |
| 4096 | 28.0        | 29.7            | 29.8          | 99.6%       |
| 8192 | 12.5        | 30.0            | 29.8          | 100.6%      |

**Key finding:** At size 8192, performance jumped from ~42% to 100.6% of cuBLAS — the swizzled program IDs and autotuned tile configs eliminated the L2 cache thrashing that caused the Week 1 performance cliff.

See `docs/week2_autotuning.md` for the full autotuning analysis and best configurations per matrix size.

## Hardware

- NVIDIA RTX 3080 (8704 CUDA cores, 10 GB GDDR6X, 29.77 TFLOPS FP16 peak)
- NVIDIA RTX 5060 Ti 16 GB (secondary dev machine)

## Debugging

```bash
TRITON_INTERPRET=1 python tests/test_correctness.py   # CPU interpreter mode
TRITON_DEBUG=1 python tests/test_correctness.py        # verbose Triton logging
compute-sanitizer python tests/test_correctness.py     # memory error checking
```

## Team

3-person team — see `docs/plan.md` for the full 3-week task breakdown.
