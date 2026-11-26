"""Packed NVFP4 test data. All values are exact; quantization is outside timing."""

from typing import Any
import torch
import cutlass
import cutlass.torch as cutlass_torch

SF_VECTOR_SIZE = 16


def pack_nvfp4_scales(logical: torch.Tensor) -> torch.Tensor:
    """Pack logical ``(MN, K/16, L)`` scales into SM120 basic chunks."""
    mn, sf_k, batch = logical.shape
    rest_mn = mn // 128
    rest_k = sf_k // 4

    # Contiguous physical order is (L,restMN,restK,32,4,4).  The final view has
    # the strides expected by the MMA-facing (32,4,restMN,4,restK,L) tensor.
    physical_base = (
        logical.reshape(rest_mn, 4, 32, rest_k, 4, batch)
        .permute(5, 0, 3, 2, 1, 4)
        .contiguous()
    )
    return physical_base.permute(3, 4, 1, 5, 2, 0)


def make_problem(m: int, n: int, k: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Cover every finite E2M1 magnitude and both signs. Dyadic scales keep
    # these test dot products exactly representable during FP32 accumulation.
    levels = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        device="cuda",
    )
    a_reference = levels[torch.randint(0, len(levels), (m, k, 1), device="cuda")]
    b_reference = levels[torch.randint(0, len(levels), (n, k, 1), device="cuda")]

    scale_choices = torch.tensor([0.5, 1.0, 2.0], device="cuda", dtype=torch.float32)
    sfa_reference = scale_choices[
        torch.randint(0, 3, (m, k // SF_VECTOR_SIZE, 1), device="cuda")
    ]
    sfb_reference = scale_choices[
        torch.randint(0, 3, (n, k // SF_VECTOR_SIZE, 1), device="cuda")
    ]
    sfa_packed_reference = pack_nvfp4_scales(sfa_reference)
    sfb_packed_reference = pack_nvfp4_scales(sfb_reference)

    c_reference_storage = torch.empty((m, n, 1), device="cuda", dtype=torch.bfloat16)

    a_tensor, a_storage = cutlass_torch.cute_tensor_like(
        a_reference,
        cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, b_storage = cutlass_torch.cute_tensor_like(
        b_reference,
        cutlass.Float4E2M1FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor, c_storage = cutlass_torch.cute_tensor_like(
        c_reference_storage,
        cutlass.BFloat16,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    sfa_tensor, sfa_storage = cutlass_torch.cute_tensor_like(
        sfa_packed_reference,
        cutlass.Float8E4M3FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    sfb_tensor, sfb_storage = cutlass_torch.cute_tensor_like(
        sfb_packed_reference,
        cutlass.Float8E4M3FN,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    # NVFP4 packs two E2M1 values per byte.  The divisibility promise ensures that
    # dynamic K extents preserve 128-bit vector alignment and full K=128 tiles.
    a_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=128,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=128,
    )
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)

    return {
        "a_reference": a_reference,
        "b_reference": b_reference,
        "sfa_reference": sfa_reference,
        "sfb_reference": sfb_reference,
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
        "sfa_tensor": sfa_tensor,
        "sfb_tensor": sfb_tensor,
        "c_tensor": c_tensor,
        "c_storage": c_storage,
        # Retain every backing allocation for the lifetime of its CuTe view.
        "live_storage": (
            a_storage,
            b_storage,
            sfa_storage,
            sfb_storage,
            c_storage,
        ),
        "stream": cutlass_torch.current_stream(),
    }


def launch(compiled: Any, problem: dict[str, Any]) -> None:
    compiled(
        problem["a_tensor"],
        problem["b_tensor"],
        problem["sfa_tensor"],
        problem["sfb_tensor"],
        problem["c_tensor"],
        problem["stream"],
    )
