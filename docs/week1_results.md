# Week 1 Experimental Results

## Environment

- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **PyTorch**: 2.11.0+cu128
- **Triton**: 3.6.0 (triton-windows)
- **Python**: 3.11.9
- **OS**: Windows 11

## Kernel Configuration

- Tile sizes: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 (defaults)
- Input dtype: FP16
- Output dtype: FP16
- Accumulator: configurable FP32 or FP16

## Correctness Results — FP32 Accumulation

| Matrix Size (M=N=K) | Max Absolute Error | Pass |
|---|---|---|
| 512 | 0.000000 | Yes |
| 1024 | 0.000000 | Yes |
| 2048 | 0.000000 | Yes |
| 4096 | 0.000000 | Yes |
| 8192 | 0.000000 | Yes |
| 127x91 @ 91x193 | 0.031250 | Yes |
| 255x257 @ 257x129 | 0.015625 | Yes |

FP32 accumulation produces **zero error** on all square sizes compared to cuBLAS (`torch.matmul`). Non-aligned shapes show small errors due to masking/boundary handling, but well within tolerance.

## Correctness Results — FP16 Accumulation

| Matrix Size (M=N=K) | Max Absolute Error | Pass |
|---|---|---|
| 512 | 0.125000 | Yes |
| 1024 | 0.312500 | Yes |
| 2048 | 0.750000 | Yes |
| 4096 | 1.250000 | Yes |
| 8192 | 3.750000 | Yes |
| 127x91 @ 91x193 | 0.031250 | Yes |
| 255x257 @ 257x129 | 0.062500 | Yes |

## Key Observation: Error Scaling

FP16 accumulation error grows with matrix size:

```
Size   | Error  | Ratio to previous
-------|--------|------------------
512    | 0.125  | —
1024   | 0.3125 | 2.5x
2048   | 0.75   | 2.4x
4096   | 1.25   | 1.7x
8192   | 3.75   | 3.0x
```

Error grows roughly **2-3x per doubling of matrix size**. This is expected: with FP16 accumulation, each `tl.dot` partial sum is truncated to FP16 before being added to the accumulator. As K increases, more rounding errors compound. FP32 accumulation avoids this entirely by keeping full precision throughout the reduction.

This scaling behavior is the core finding that Week 2's numerical error analysis (Person B) will quantify in detail with relative error, error distributions, and plots.

## Non-Aligned Shapes

Both accumulation modes handle non-power-of-2 shapes correctly via boundary masking. The small errors on non-aligned shapes (0.015–0.0625) are comparable across both modes, suggesting they come from floating-point ordering differences rather than accumulator precision.

## What This Validates

1. The `USE_FP32_ACC` flag correctly switches accumulator precision at the kernel level
2. FP32 accumulation matches cuBLAS exactly on square sizes
3. FP16 accumulation introduces measurable, size-dependent error — confirming the study's hypothesis
4. Masking logic handles arbitrary shapes in both modes
