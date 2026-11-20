import importlib

import pytest
import torch

from shared.attention import launch, make_inputs, reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("hardware", ["gb10", "b300"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,heads,dim,page_size,lengths", [
    (4, 2, 64, 16, [0, 1, 31, 65]),
    (3, 3, 80, 7, [7, 32, 97]),
    (2, 1, 8, 16, [0, 0]),
    (1, 4, 128, 32, [513]),
    (0, 2, 64, 16, []),
    (2, 0, 64, 16, [1, 33]),
])
def test_attention(hardware, dtype, batch, heads, dim, page_size, lengths):
    solve = importlib.import_module(f"{hardware}.flash_attention_decode.kernel").solve
    inputs = make_inputs(batch, heads, dim, max(lengths, default=0), page_size, dtype, lengths)
    expected = reference(*inputs)
    launch(solve, inputs)
    torch.testing.assert_close(inputs[-2], expected[0], atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(inputs[-1], expected[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("hardware", ["gb10", "b300"])
def test_without_lse(hardware):
    solve = importlib.import_module(f"{hardware}.flash_attention_decode.kernel").solve
    inputs = make_inputs(batch=1, heads=2, context=33)
    q, k, v, table, lengths, output, _ = inputs
    solve(q, k, v, table, lengths, output,
          q.shape[0], q.shape[1], q.shape[2], k.shape[1], table.shape[1])
    torch.testing.assert_close(output, reference(*inputs)[0], atol=2e-3, rtol=2e-2)
