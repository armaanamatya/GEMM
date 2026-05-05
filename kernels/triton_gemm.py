import torch
import triton
import triton.language as tl


# autotune grid
def _autotune_configs():
    configs = []
    for BLOCK_M in [64, 128, 256]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        for GROUP_M in [4, 8]:
                            if BLOCK_M == 256 and BLOCK_N == 256:
                                continue
                            configs.append(
                                triton.Config(
                                    {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                                     "BLOCK_K": BLOCK_K, "GROUP_M": GROUP_M},
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                ))
    return configs


@triton.autotune(
    configs=_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel_autotune(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    USE_FP32_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # swizzle for L2
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

    if USE_FP32_ACC:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    for _ in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        if USE_FP32_ACC:
            acc += tl.dot(a, b)
        else:
            acc += tl.dot(a, b).to(tl.float16)

        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# fixed-tile kernel
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    USE_FP32_ACC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    if USE_FP32_ACC:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    for _ in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        if USE_FP32_ACC:
            acc += tl.dot(a, b)
        else:
            acc += tl.dot(a, b).to(tl.float16)

        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor,
                  use_fp32_acc: bool = True,
                  block_m: int = 64,
                  block_n: int = 64,
                  block_k: int = 32) -> torch.Tensor:
    # tiled fp16 gemm, MxK @ KxN
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("A and B must be CUDA tensors.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Inputs must be FP16 tensors.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Inputs must be contiguous tensors.")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        USE_FP32_ACC=use_fp32_acc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


def triton_matmul_autotune(a: torch.Tensor, b: torch.Tensor,
                           use_fp32_acc: bool = True) -> torch.Tensor:
    # autotuned fp16 gemm
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("A and B must be CUDA tensors.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("Inputs must be FP16 tensors.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Inputs must be contiguous tensors.")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _matmul_kernel_autotune[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        USE_FP32_ACC=use_fp32_acc,
    )
    return c


def get_autotune_best_config(a: torch.Tensor, b: torch.Tensor,
                              use_fp32_acc: bool = True) -> dict:
    # best autotune config
    triton_matmul_autotune(a, b, use_fp32_acc=use_fp32_acc)
    best = _matmul_kernel_autotune.best_config
    return {
        "BLOCK_M": best.kwargs["BLOCK_M"],
        "BLOCK_N": best.kwargs["BLOCK_N"],
        "BLOCK_K": best.kwargs["BLOCK_K"],
        "GROUP_M": best.kwargs["GROUP_M"],
        "num_warps": best.num_warps,
        "num_stages": best.num_stages,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b)

    c_fp32 = triton_matmul(a, b, use_fp32_acc=True)
    c_fp16 = triton_matmul(a, b, use_fp32_acc=False)

    print("FP32 accum — max abs error:", (c_fp32 - ref).abs().max().item())
    print("FP16 accum — max abs error:", (c_fp16 - ref).abs().max().item())
