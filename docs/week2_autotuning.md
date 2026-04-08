# Week 2: Autotuning and Swizzled Scheduling

**COSC 4397 — Parallel Computing on GPUs**
**Date:** April 2026

---

## 1. What is Autotuning?

Triton's `@triton.autotune` decorator benchmarks multiple kernel configurations at runtime and caches the fastest one for each unique problem shape `(M, N, K)`. This eliminates the need for manual tuning — the compiler tries every combination in a config grid and picks the winner.

### V1 Config Grid (72 configurations)

```python
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
            num_warps=w, num_stages=s,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [32, 64]
        for w  in [4, 8]
        for s  in [2, 3, 4]
    ],
    key=['M', 'N', 'K'],
)
```

This produces `2 × 3 × 2 × 2 × 3 = 72` configs. For every new `(M, N, K)` triple the autotuner sees, it benchmarks all 72 and picks the fastest.

### V2 Expanded Config Grid (~192 configurations)

```python
def _autotune_configs():
    configs = []
    for bm in [64, 128, 256]:
        for bn in [64, 128, 256]:
            if bm == 256 and bn == 256:
                continue          # skip — too much register pressure
            for bk in [32, 64]:
                for w in [4, 8]:
                    for s in [2, 3, 4]:
                        for gm in [4, 8]:
                            configs.append(triton.Config(
                                {'BLOCK_M': bm, 'BLOCK_N': bn,
                                 'BLOCK_K': bk, 'GROUP_M': gm},
                                num_warps=w, num_stages=s,
                            ))
    return configs   # ~192 configs
```

Key additions in V2:
- `BLOCK_M` and `BLOCK_N` now include **256** (the 256×256 combo is excluded to stay within register limits).
- A new parameter `GROUP_M` with values **[4, 8]** controls L2 cache locality (explained in Section 2).

### What Each Parameter Controls

| Parameter | What It Does | Trade-off |
|-----------|-------------|-----------|
| `BLOCK_M` | Height of the output tile each thread block computes. | Larger → more work per block, but higher register pressure. |
| `BLOCK_N` | Width of the output tile each thread block computes. | Same trade-off as `BLOCK_M`, along the N dimension. |
| `BLOCK_K` | Chunk size along the inner (reduction) dimension per loop iteration. | Larger → more data reuse per load, but more shared memory consumed. |
| `num_warps` | Number of 32-thread warps executing within one block. | More warps → better latency hiding (can cover memory stalls), but each warp needs its own registers. |
| `num_stages` | Software pipelining depth — how many loop iterations are in flight at once. | More stages → loads overlap with compute (hides memory latency), but uses more shared memory for buffering. |
| `GROUP_M` | **(New in V2)** Number of adjacent M-tiles grouped together for scheduling. | Controls L2 cache locality — groups of blocks share the same B tiles. |

---

## 2. Swizzled Program IDs (L2 Cache Locality)

### The Problem: Naive 2D Grid

In V1 the kernel uses a straightforward 2D grid:

```python
# V1: Naive 2D grid
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
# grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
```

For an 8192×8192 matrix with 128×128 tiles there are `64 × 64 = 4096` blocks. The GPU schedules them in row-major order, so block `(0, 0)` reads `B[:, 0:128]` and block `(0, 1)` reads `B[:, 128:256]`. These are completely different memory regions. As block (0, 1) loads its B slice, it evicts the B slice that block (0, 0) just loaded. The result is **L2 cache thrashing** — every block misses on B.

### The Solution: Swizzled 1D Grid

V2 collapses the 2D grid into a single dimension and remaps program IDs so that a *group* of M-tiles sweeps across N together:

```python
# V2: Swizzled 1D grid with GROUP_M for L2 locality
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
# grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
```

With `GROUP_M = 8`, the first `8 × num_pid_n` blocks all read from the same 8 row-tiles of A. As they sweep across the columns of B, they share B column-tiles in L2 cache. By the time a block in the group needs a B tile, a nearby block has likely already loaded it.

### Block Scheduling Comparison

```
Naive (row-major):             Swizzled (GROUP_M = 4):
 [ 0][ 1][ 2][ 3]             [ 0][ 4][ 8][12]
 [ 4][ 5][ 6][ 7]             [ 1][ 5][ 9][13]
 [ 8][ 9][10][11]             [ 2][ 6][10][14]
 [12][13][14][15]             [ 3][ 7][11][15]

 Block 0 → 1: different B     Block 0 → 1: same B column,
 column → L2 thrashing          different A row → B stays in L2
```

In the naive layout, consecutively-scheduled blocks step *across* B columns, constantly evicting cached data. In the swizzled layout, consecutive blocks step *down* A rows while reusing the **same** B column tiles. This dramatically reduces L2 miss rate for large matrices where the full B tile set exceeds L2 capacity.

---

## 3. V1 Autotuning Results (Before Swizzle)

Hardware: **RTX 5060 Ti** (Blackwell, 16 GB GDDR7, 512 GB/s)

| Size | Auto FP32 (TFLOPS) | Auto FP16 (TFLOPS) | Fixed FP32 (TFLOPS) | cuBLAS (TFLOPS) | % cuBLAS | Speedup vs Fixed | Best Config |
|------|---------------------|---------------------|----------------------|------------------|----------|------------------|-------------|
| 512  | 19.25 | 19.13 | 16.10 | 19.10 | 100.8% | 1.20× | 128×64×64, warps=8, stages=4 |
| 1024 | 35.62 | 35.99 | 35.83 | 35.23 | 101.1% | 0.99× | 128×64×64, warps=8, stages=3 |
| 2048 | 46.42 | 43.41 | 46.85 | 43.16 | 107.6% | 0.99× | 64×64×32, warps=4, stages=4 |
| 4096 | 46.71 | 45.81 | 43.97 | 49.68 | 94.0%  | 1.06× | 128×64×64, warps=4, stages=2 |
| 8192 | 48.95 | 44.26 | 19.68 | 48.66 | 100.6% | 2.49× | 128×128×64, warps=4, stages=2 |

### Observations

- **8192 saw a 2.49× speedup** over fixed tiles. The autotuner found 128×128×64 tiles, which keep each thread block busy with enough work to saturate compute, whereas the default 64×64×32 tiles left performance on the table.
- **2048 was already optimal** with the original 64×64×32 config — the autotuner independently confirmed this.
- **4096 at 94% cuBLAS** is the main gap. This is exactly where L2 thrashing starts to bite (B tiles at 4096 exceed what naive scheduling can keep cached), making it the prime target for swizzled scheduling.

---

## 4. V2 Results (After Swizzle + Expanded Grid)

Hardware: **RTX 5060 Ti** (Blackwell, 16 GB GDDR7, 512 GB/s)

| Size | Auto FP32 (TFLOPS) | Auto FP16 (TFLOPS) | Fixed FP32 (TFLOPS) | cuBLAS (TFLOPS) | % cuBLAS | Speedup vs Fixed | Best Config |
|------|---------------------|---------------------|----------------------|------------------|----------|------------------|-------------|
| *(to be filled after benchmark run)* | | | | | | | |

---

## 5. Key Findings

1. **Autotuning matters most for large matrices.** At 8192×8192 the autotuner delivered a 2.49× improvement over the hand-picked default, proving that the "best" tile size is highly shape-dependent.

2. **Small matrices are already well-served.** At 512 and 1024 the autotuned and fixed configs perform within a few percent of each other — the kernel is compute-bound at these sizes and tile choice has limited impact.

3. **Larger matrices need larger tiles.** The autotuner consistently selects bigger `BLOCK_M × BLOCK_N` tiles as the problem grows. Larger tiles amortize pointer arithmetic and launch overhead, and they give each block a bigger share of the output to fill.

4. **4096 is the swizzle sweet spot.** At 94% of cuBLAS this is where naive scheduling hurts the most. With 128×128 tiles on a 4096 matrix there are 32×32 = 1024 blocks. In row-major order each row of 32 blocks reads 32 different B column-tiles — far more than L2 can hold simultaneously. Swizzling should close this gap.

5. **Swizzle expected to help at 4096+.** For smaller sizes the entire B matrix fits in L2 (~48 MB on the 5060 Ti), so scheduling order is irrelevant. At 4096 the B matrix is 4096 × 4096 × 2 bytes = 32 MB in FP16, still fitting in L2 as a whole but not when accessed with bad locality. At 8192 the B matrix is 128 MB — swizzling is essential to keep the working set in cache.

---

## 6. Hardware Context

| Spec | RTX 5060 Ti (ours) | RTX 3080 (teammate) |
|------|-------------------|---------------------|
| Architecture | Blackwell | Ampere |
| CUDA Cores | 4608 | 8704 |
| VRAM | 16 GB GDDR7 | 10 GB GDDR6X |
| Memory Bandwidth | 512 GB/s | 760 GB/s |
| Observed FP16 Peak | ~48 TFLOPS | ~61 TFLOPS |
| L2 Cache | ~48 MB | ~5 MB |

### Why Hardware Matters for This Optimization

- **L2 cache size is the key differentiator for swizzling.** The 5060 Ti's ~48 MB L2 can hold a significant fraction of B tiles, but only if the scheduling pattern gives it a chance. Swizzling ensures the active working set stays within L2 budget.
- **Cross-GPU comparison:** On the teammate's RTX 3080 with its 5 MB L2, the fixed-tile kernel reached 40.84 TFLOPS at 8192. Our autotuned 5060 Ti hits 48.95 TFLOPS despite having fewer CUDA cores and lower bandwidth — the massive L2 advantage of Blackwell makes up for it.
- **Bandwidth vs compute:** The 3080 has 48% more memory bandwidth (760 vs 512 GB/s) but only 27% higher peak FP16 throughput. The 5060 Ti compensates with its dramatically larger L2, reducing the number of DRAM round-trips needed.

---

## 7. Running the Benchmarks

```bash
# Run autotuned benchmark sweep
# (first run is slow — the autotuner benchmarks all ~192 configs per shape)
python benchmarks/bench_autotune_sweep.py --csv benchmarks/results/autotune_sweep.csv

# Run correctness tests (includes autotuned kernel validation)
python tests/test_correctness.py

# Run original fixed-tile sweep for comparison
python benchmarks/bench_gemm_sweep.py --csv benchmarks/results/sweep.csv
```

**Note:** The first invocation of the autotuned kernel for a new `(M, N, K)` shape will be slow as Triton benchmarks every configuration. Subsequent runs use the cached winner and are fast. The cache is stored in `~/.triton/cache/` and persists across runs.
