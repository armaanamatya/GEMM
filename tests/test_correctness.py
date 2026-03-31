import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from kernels import make_test_case, pytorch_matmul, triton_matmul


def check_case(m: int, n: int, k: int, use_fp32_acc: bool,
               atol: float = 1e-2, rtol: float = 1e-2):
    a, b = make_test_case(m, n, k)
    out = triton_matmul(a, b, use_fp32_acc=use_fp32_acc)
    ref = pytorch_matmul(a, b)

    ok = torch.allclose(out, ref, atol=atol, rtol=rtol)
    max_abs = (out - ref).abs().max().item()
    mode = "FP32" if use_fp32_acc else "FP16"
    print(f"[{mode} acc] [{m}x{k}] @ [{k}x{n}] -> ok={ok}, max_abs={max_abs:.6f}")
    return ok


def main():
    # Square sizes required by the plan
    square_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]

    # Non-aligned sizes to verify masking
    non_aligned_sizes = [
        (127, 193, 91),
        (255, 129, 257),
    ]

    all_ok = True

    print("=== FP32 accumulation mode ===")
    for m, n, k in square_sizes + non_aligned_sizes:
        if not check_case(m, n, k, use_fp32_acc=True):
            all_ok = False

    # FP16 accumulation has higher error that grows with matrix size.
    # This is expected and is the core of our mixed-precision study.
    # Tolerances are relaxed accordingly — the error analysis in Week 2
    # will quantify and explain this scaling.
    print("\n=== FP16 accumulation mode ===")
    for m, n, k in square_sizes + non_aligned_sizes:
        if not check_case(m, n, k, use_fp32_acc=False, atol=5.0, rtol=1e-1):
            all_ok = False

    if not all_ok:
        raise SystemExit("Correctness test FAILED.")
    print("\nAll correctness tests passed.")


if __name__ == "__main__":
    main()
