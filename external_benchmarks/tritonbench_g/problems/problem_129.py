import torch
import torch.nn as nn
import triton
import triton.language as tl

# Self-reference Model: defines the same triton kernel as ModelNew
# so baseline differential tests (ref vs orig) trivially pass and
# stress tests can still detect non-determinism via repeated_run.
# Inputs were mined from the original TritonBench-G test_* block
# in apply_penalty.py.

import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, presence_penalty, freqency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len, 
    stride_logit_b, stride_logit_s,
    BLOCK_P: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0)
    
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset<cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset<cur_batch_end_index)

    return

@torch.no_grad()
def apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
    assert Logits.is_contiguous()
    BLOCK = triton.next_power_of_2(p_max_len_in_batch)
    if BLOCK <= 512:
        BLOCK = 512
    elif BLOCK <= 1024:
        BLOCK = 1024
    num_warps = 8
    _fwd_kernel_apply_penalty[(Logits.shape[0], )](
        Logits, presence_penalty, freqency_penalty, repetition_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    return






class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
        return apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch)


def get_inputs():
    results = {}
    batch_size = 2
    seq_len = 10
    vocab_size = 50
    Logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device='cuda').contiguous()
    presence_penalty = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda')
    freqency_penalty = torch.tensor([0.1, 0.2], dtype=torch.float32, device='cuda')
    repetition_penalty = torch.tensor([1.2, 1.3], dtype=torch.float32, device='cuda')
    p_token_ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.int32, device='cuda')
    p_token_counts = torch.randint(1, 5, (seq_len,), dtype=torch.int32, device='cuda')
    p_cumsum_seq_len = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
    p_max_len_in_batch = seq_len
    return [Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch]

def get_init_inputs():
    return []
