# COSC 4397 Final Project Plan

## Mixed-Precision FP16 GEMM in Triton

**Platform:** NVIDIA RTX 3080 + RTX 5060 Ti | **Framework:** OpenAI Triton | **Team:** 3 people | **Timeline:** 3 weeks

---

## Project Goal

Implement a tiled FP16 GEMM kernel in Triton with a configurable FP16/FP32 accumulation flag, validate against PyTorch matmul for correctness, compare against cuBLAS-backed FP16 matmul for performance, and study the numerical/performance trade-offs of accumulator precision.

## A-Grade Differentiator

**Mixed-Precision Accumulation Study** — the rubric explicitly lists this as a recommended Advanced GEMM direction. We add a configurable accumulator precision flag and study:

1. **Numerical error**: How max absolute and relative error scale with matrix size (FP16 vs FP32 accumulation)
2. **Performance cost**: TFLOPS penalty of FP32 accumulation (register pressure, reduced occupancy, tile size interactions)
3. **Roofline positioning**: Where each configuration sits relative to compute and memory bandwidth ceilings

## Matrix Sizes

All benchmarks and analysis run on: **512, 1024, 2048, 4096, 8192**

The 8192 size is critical — it's where FP16 vs FP32 accumulation differences become most visible in both error and performance.

---

## Week 1: Implementation & Correctness (Mar 31 - Apr 6)

### Person A — Core Kernel (DONE)

- [x] Add configurable accumulator precision flag to the Triton GEMM kernel (`USE_FP32_ACC: tl.constexpr`)
  - FP32 mode: `acc = tl.zeros(..., dtype=tl.float32)` (default)
  - FP16 mode: `acc = tl.zeros(..., dtype=tl.float16)`
  - Exposed as `use_fp32_acc: bool` in `triton_matmul()` wrapper
- [x] Pass correctness checks against `torch.matmul` for square matrices (512, 1024, 2048, 4096, 8192)
- [x] Handle non-power-of-2 and non-aligned shapes with proper masking
- [x] Reorganize project into `kernels/`, `tests/`, `setup/`, `docs/` folder structure
- [x] Update correctness tests for both accumulation modes with appropriate tolerances
- **Deliverable:** Working kernel with both accumulation modes + correctness tests — **committed `9c57507`**

### Person B — Benchmarking Harness (DONE)

- [x] Replace `time.perf_counter` with `triton.testing.do_bench` (warmup=25, rep=100, median timing)
- [x] Implement TFLOPS calculation: `2 * M * N * K / time_seconds / 1e12`
- [x] Build automated sweep script over all 5 matrix sizes for both accumulation modes
- [x] Record cuBLAS baseline numbers (`torch.matmul` on FP16 CUDA tensors)
- **Deliverable:** `benchmarks/bench_gemm_sweep.py` + CSV output — **committed `347e1d1`**
- **Key finding:** Triton reaches 87–98% of cuBLAS at sizes 512–4096, but drops to ~42% at 8192 (default 64×64×32 tiles). Autotuning in Week 2 should close this gap.

### Person C — Profiling Setup

- [ ] Install and configure Nsight Compute (`ncu`)
- [ ] Get baseline profiles of the Triton kernel and cuBLAS for at least one matrix size
- [ ] Document achieved TFLOPS, memory bandwidth, and occupancy
- [ ] Record GPU specs for the report:
  - RTX 3080: 8704 CUDA cores, 10 GB GDDR6X, 29.77 TFLOPS FP16 peak, 760 GB/s bandwidth (Ampere)
  - RTX 5060 Ti: 4608 CUDA cores, 16 GB GDDR7, ~48 TFLOPS FP16 observed peak, 512 GB/s bandwidth (Blackwell)
- **Deliverable:** Nsight profiles + baseline metrics document

---

## Week 2: Optimization & Experimentation (Apr 7 - Apr 13)

### Person A — Autotuning Sweep

- [x] Add `@triton.autotune` decorator with expanded config grid (192 configs) including GROUP_M:
  - `BLOCK_M`: [64, 128]
  - `BLOCK_N`: [64, 128, 256]
  - `BLOCK_K`: [32, 64]
  - `GROUP_M`: [4, 8]
  - `num_warps`: [4, 8]
  - `num_stages`: [2, 3, 4]
- [x] Record best configurations per matrix size for both FP16 and FP32 accumulation
- [x] Optional stretch: add swizzled program IDs for better L2 cache locality
- **Deliverable:** Autotuned kernel with swizzled program IDs + expanded config grid + documentation

### Person B — Numerical Error Analysis

- [ ] Measure max absolute error and relative error for each matrix size:
  - Triton FP16 accum vs PyTorch reference
  - Triton FP32 accum vs PyTorch reference
- [ ] Generate error scaling plots (error vs matrix size, both modes)
- [ ] Analyze: does error grow linearly? Sublinearly? Why?
- [ ] Compute condition number analysis if time permits
- **Deliverable:** Error analysis plots + data tables

### Person C — Roofline & Deep Profiling

- [ ] Compute arithmetic intensity for GEMM: `2 * M * N * K / (2 * (M*K + K*N + M*N))` bytes (FP16)
- [ ] Build roofline model for RTX 3080 (compute ceiling + bandwidth ceiling)
- [ ] Plot achieved FLOPS vs roofline for each matrix size and both accumulation modes
- [ ] Nsight Compute deep analysis for best and worst autotuning configs:
  - Register usage per thread
  - Occupancy (achieved vs theoretical)
  - Shared memory usage
  - Memory throughput (achieved vs peak)
- **Deliverable:** Roofline plots + register/occupancy analysis

---

## Week 3: Analysis & Report (Apr 14 - Apr 20)

### Person A — Final Plots & Code Polish

- [ ] Generate final performance plots:
  - TFLOPS vs matrix size (FP16 accum, FP32 accum, cuBLAS)
  - % of cuBLAS peak for both accumulation modes
  - Autotuning heatmap (config vs TFLOPS)
- [ ] Clean up kernel code with inline documentation
- [ ] Ensure all scripts are runnable from a clean environment
- **Deliverable:** Final plots + clean codebase

### Person B — Results & Analysis Writing

- [ ] Write results section: present all benchmark and error data
- [ ] Write analysis section tying findings to course concepts:
  - Why does occupancy drop with FP32 accumulation? (register pressure)
  - How does tile size interact with shared memory capacity?
  - Why does cuBLAS outperform at certain sizes? (software pipelining, tensor cores)
  - Coalescing patterns in the Triton kernel
  - Arithmetic intensity and roofline implications
- **Deliverable:** Results + analysis draft

### Person C — Report Assembly & Presentation

- [ ] Write introduction and methodology sections
- [ ] Write reproducibility section
- [ ] Create README with:
  - Hardware/software requirements
  - Setup instructions
  - How to run benchmarks, tests, and profiling
- [ ] Assemble final report (all sections)
- [ ] Prepare presentation slides or demo
- **Deliverable:** Complete report + README + presentation

---

## Deliverables Checklist

| Rubric Requirement | How We Address It |
|---|---|
| Clear scope | One kernel (tiled GEMM) with one focused study (accumulator precision) |
| Correctness validation | PyTorch `torch.matmul` as trusted reference |
| Meaningful student-written GPU code | Full Triton kernel with configurable accumulator, autotuning, benchmarking harness |
| Performance evaluation (3+ input sizes) | 5 sizes: 512, 1024, 2048, 4096, 8192. cuBLAS as baseline |
| Analysis tied to course concepts | Occupancy, register pressure, roofline, tiling, coalescing, arithmetic intensity |
| Reproducibility | Code, README, build/run instructions, requirements.txt |

## Fallback Plan

If mixed-precision accumulation study proves too complex, fall back to:
- FP16-only Triton GEMM with thorough autotuning sweep
- Performance comparison against cuBLAS across all 5 sizes
- This still satisfies all rubric hard requirements

## Tools & Resources

- **Hardware:** NVIDIA RTX 3080 (Ampere) + RTX 5060 Ti (Blackwell) — same Triton kernel runs on both
- **Framework:** OpenAI Triton (Python-based tile language for GPU kernel authoring)
- **Baseline:** PyTorch `torch.matmul` (cuBLAS-backed FP16 matmul)
- **Profiling:** Nsight Compute (occupancy, register spills, memory throughput)
- **Analysis:** Roofline model, TFLOPS measurement, achieved bandwidth calculation
