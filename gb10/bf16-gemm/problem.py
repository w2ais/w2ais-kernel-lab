"""BF16 allocations and reference checks shared by the worklog runners.

C[M,N] = A[M,K] @ B[N,K].T, with K-major inputs and N-major output.
The two allocation policies preserve the original checkpoint and TMA inputs;
neither changes the data between correctness checks and timed launches.
"""

from dataclasses import dataclass

import torch
from cutlass.cute.runtime import from_dlpack


@dataclass
class Problem:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor

    @property
    def matrices(self):
        return tuple(t[:, :, 0] if t.ndim == 3 else t for t in (self.a, self.b, self.c))

    def cute_tensors(self):
        views = []
        for tensor in (self.a, self.b, self.c):
            view = from_dlpack(tensor, assumed_align=16)
            if tensor.ndim == 3:
                view = view.mark_layout_dynamic(leading_dim=1)
                view = view.mark_compact_shape_dynamic(
                    mode=1,
                    stride_order=(2, 0, 1),
                    divisibility=8,
                )
            views.append(view)
        return tuple(views)

    def reference(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        a, b, _ = self.matrices
        return a.float() @ b.float().T

    def check(self, launch, *, repeats=1, atol=0.01, rtol=0.004):
        expected = self.reference()
        errors = []
        for _ in range(repeats):
            self.c.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            actual = self.matrices[2].float()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            difference = (actual - expected).abs()
            errors.append(float(difference.max()))
        return {
            "checked": True,
            "passed": True,
            "atol": atol,
            "rtol": rtol,
            "repeats": repeats,
            "max_absolute_error": max(errors),
            "max_absolute_errors": errors,
            "mean_absolute_error": float(difference.mean()),
        }


def make_problem(m, n, k, *, seed=2026, input="normal", allocation="checkpoint"):
    """Allocate BF16 inputs; ``tma`` keeps the original CPU RNG and batch axis.

    The checkpoint policy generates normal values directly on the GPU. The
    TMA policy generates FP32 normals (or small integers) on the CPU, then
    converts to BF16 and transfers to the GPU. Keeping this distinction makes
    old seeded checks reproducible, including the initialized TMA output.
    """
    if min(m, n, k) <= 0:
        raise ValueError("M, N and K must be positive")
    if input not in ("normal", "integer"):
        raise ValueError(f"Unknown input distribution: {input}")
    torch.manual_seed(seed)
    if allocation == "tma":

        def allocate(rows, columns, k_major=True):
            shape = (1, rows, columns) if k_major else (1, columns, rows)
            if input == "integer":
                tensor = torch.empty(shape, dtype=torch.int32).random_(-2, 2)
            else:
                tensor = torch.randn(shape, dtype=torch.float32)
            tensor = tensor.to(torch.bfloat16)
            return (
                tensor.permute(1, 2, 0).cuda()
                if k_major
                else tensor.permute(2, 1, 0).cuda()
            )

        return Problem(allocate(m, k), allocate(n, k), allocate(m, n))
    if allocation != "checkpoint":
        raise ValueError(f"Unknown allocation policy: {allocation}")

    def allocate(rows, columns):
        if input == "normal":
            return torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16)
        return torch.randint(-2, 2, (rows, columns), device="cuda").to(torch.bfloat16)

    return Problem(
        allocate(m, k),
        allocate(n, k),
        torch.empty((m, n), device="cuda", dtype=torch.bfloat16),
    )
