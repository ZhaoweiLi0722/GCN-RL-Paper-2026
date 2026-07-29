"""Fail fast unless PyTorch can execute work on an NVIDIA CUDA device."""

from __future__ import annotations

import platform
import time

import torch


def main() -> None:
    print(f"python_platform={platform.platform()}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Check the NVIDIA driver and install the "
            "CUDA-enabled PyTorch wheel."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"device_name={properties.name}")
    print(
        "device_memory_gib="
        f"{properties.total_memory / (1024 ** 3):.2f}"
    )
    print(
        "compute_capability="
        f"{properties.major}.{properties.minor}"
    )

    torch.manual_seed(0)
    left = torch.randn((2048, 2048), device=device)
    right = torch.randn((2048, 2048), device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = left @ right
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"matmul_seconds={elapsed:.4f}")
    print(f"matmul_checksum={float(output[0, 0].cpu()):.6f}")


if __name__ == "__main__":
    main()
