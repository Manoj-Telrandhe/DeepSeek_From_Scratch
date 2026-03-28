import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
  def __init__(self, d_in, d_out, num_heads, num_groups, dropout=0.0, max_seq_len: int = 1024):
    super().__init__()
    assert d_out % num_heads == 0, "d_out must be divide by num_heads"
    assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.num_groups = num_groups
    self.heads_per_group = num_heads // num_groups

    self.W_query = nn.Linear(d_in, d_out, bias=False)
    self.W_key = nn.Linear(d_in, self.num_groups * self.head_dim, bias=False)   # Grouped projection for K
    self.W_value = nn.Linear(d_in, self.num_groups * self.head_dim, bias=False) # Grouped projection for K

    self.out_proj = nn.Linear(d_out, d_out)

    self.dropout = nn.Dropout(dropout)
    self.register_mask_buffer(max_seq_len)

  def register_mask_buffer(self, max_seq_len):
    if max_seq_len > 0:
      mask = torch.triu(torch.ones(1, 1, max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
      self.register_buffer("causal_mask", mask, persistent=False)
    else:
      self.causal_mask = None

  def _get_causal_mask(self, seq_len, device):
    if self.causal_mask is not None and self.causal_mask.size(-1) >= seq_len:
      return self.causal_mask[:, :, :seq_len, :seq_len]
    # Dynamically create mask if needed
    return torch.triu(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

  def forward(self, x):
    b, num_tokens, d_in = x.shape

    # Query
    queries = self.W_query(x)   # ( b, num_tokens, d_out )
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # (b, num_tokens, num_heads, head_dim)
    queries = queries.transpose(1,  2)  #(b, num_heads, num_tokens, head_dim)

    # Key & Value
    keys = self.W_key(x)  # (b, num_tokens, num_groups * head_dim)
    keys = keys.view(b, num_tokens, self.num_groups, self.head_dim)  # (b, num_tokens, num_groups, head_dim)
    values = self.W_value(x)  # (b, num_tokens, num_groups * head_dim)
    values = values.view(b, num_tokens, self.num_groups, self.head_dim)  # (b, num_tokens, num_groups, head_dim)

    keys = keys.transpose(1, 2)                                      # (b, num_groups, num_tokens, head_dim)
    values = values.transpose(1, 2)                                      # (b, num_groups, num_tokens, head_dim)

    heads_per_group = self.num_heads // self.num_groups

    # Reapeat K and V
    keys = keys.repeat_interleave(self.heads_per_group, dim=1)   # (b, num_head, num_tokens, head_dim)
    values = values.repeat_interleave(self.heads_per_group, dim=1)   # (b, num_head, num_tokens, head_dim)

    # Attn scores
    attn_scores = queries @ keys.transpose(2, 3)

    # Apply causal mask
    causal_mask = self._get_causal_mask(num_tokens, x.device)
    attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

    # Attn weights
    attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)

    # dropout
    attn_weights = self.dropout(attn_weights)

    # context vectors
    context_vector = attn_weights @ values  # (batch_size, num_heads, num_tokens, head_dim)

    # (batch_size, num_tokens, num_heads, head_dim)
    context_vector = context_vector.transpose(1, 2)

    # Combine heads, where self.d_out = self.num_heads * self.head_dim
    # (batch_size, num_tokens, num_heads, head_dim) ---> # (batch_size, num_tokens, d_out)
    context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

    context_vector = self.out_proj(context_vector)

    return context_vector