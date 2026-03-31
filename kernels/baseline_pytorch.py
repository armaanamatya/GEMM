import torch


def pytorch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")
    return torch.matmul(a, b)


def make_test_case(m: int, n: int, k: int,
                   device: str = "cuda",
                   dtype: torch.dtype = torch.float16,
                   seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    return a, b


if __name__ == "__main__":
    a, b = make_test_case(128, 128, 128)
    c = pytorch_matmul(a, b)
    print(c.shape, c.dtype, c.device)
