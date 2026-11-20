"""Inputs and independent PyTorch reference shared by both hardware targets."""
import torch


def make_inputs(batch=4, heads=8, dim=64, context=1024, page_size=16,
                dtype=torch.float16, lengths=None):
    generator = torch.Generator(device="cuda").manual_seed(42)
    blocks = max(1, (context + page_size - 1) // page_size)
    pages = batch * blocks
    q = torch.randn((batch, heads, dim), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((pages, page_size, heads, dim), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=dtype, generator=generator)
    table = torch.randperm(pages, device="cuda", generator=generator).to(torch.int32).reshape(batch, blocks)
    lens = torch.tensor(lengths if lengths is not None else [context] * batch,
                        device="cuda", dtype=torch.int32)
    output = torch.full_like(q, float("nan"))
    lse = torch.full((batch, heads), float("nan"), device="cuda", dtype=torch.float32)
    return q, k, v, table, lens, output, lse


def reference(q, k, v, table, lengths, output, lse):
    result = torch.zeros_like(output)
    expected_lse = torch.full_like(lse, float("-inf"))
    for batch in range(q.shape[0]):
        length = int(lengths[batch].item())
        if length == 0:
            continue
        tokens = torch.arange(length, device=q.device)
        pages = table[batch, tokens // k.shape[1]].long()
        slots = tokens % k.shape[1]
        keys, values = k[pages, slots].float(), v[pages, slots].float()
        scores = torch.einsum("hd,thd->ht", q[batch].float(), keys) * q.shape[-1] ** -0.5
        expected_lse[batch] = torch.logsumexp(scores, dim=-1)
        result[batch] = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
    return result, expected_lse


def launch(solve, inputs):
    q, k, v, table, lengths, output, lse = inputs
    solve(q, k, v, table, lengths, output,
          q.shape[0], q.shape[1], q.shape[2], k.shape[1], table.shape[1], lse=lse)
