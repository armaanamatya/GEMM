"""Quick smoke test: verifies CUTLASS 3.5.1 can run FP16 GEMM on this GPU."""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cutlass

print(f"cutlass : {cutlass.__version__}")
print(f"torch   : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"CUDA    : {torch.version.cuda}")
print()

sizes = [512, 1024, 2048, 4096, 8192]
for n in sizes:
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    C = torch.zeros(n, n, dtype=torch.float16, device="cuda")
    D = torch.empty_like(C)
    plan = cutlass.op.Gemm(element=torch.float16, layout=cutlass.LayoutType.RowMajor,
                          element_accumulator=torch.float32)
    plan.run(a, b, C, D, alpha=1.0, beta=0.0)
    ref = torch.matmul(a, b)
    err = (D - ref).abs().max().item()
    # FP32 accumulation should give near-zero error vs cuBLAS reference
    print(f"  n={n:5d}  max_abs_err={err:.4f}  {'OK' if err < 0.1 else 'HIGH (FP16 acc?)'}")

print()
print("CUTLASS smoke test complete.")
