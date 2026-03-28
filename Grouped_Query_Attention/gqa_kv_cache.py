import torch
import torch.nn as nn

class GroupedQueryAttentionKV(nn.Module):
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
    self.W_value = nn.Linear(d_in, self.num_groups * self.head_dim, bias=False) # Grouped projection for V

    self.out_proj = nn.Linear(d_out, d_out)

    self.dropout = nn.Dropout(dropout)

    ###
    self.register_buffer('cache_k', None, persistent=False)
    self.register_buffer('cache_v', None, persistent=False)
    self.ptr_current_pos = 0
    ####


  def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    # Query
    queries = self.W_query(x)   # ( b, num_tokens, d_out )  # d_out = num_heads * head_dim
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # (b, num_tokens, num_heads, head_dim)
    queries = queries.transpose(1,  2)  #(b, num_heads, num_tokens, head_dim)

    # Key & Value
    keys = self.W_key(x)  # (b, num_tokens, num_groups * head_dim)
    keys = keys.view(b, num_tokens, self.num_groups, self.head_dim)  # (b, num_tokens, num_groups, head_dim)
    values = self.W_value(x)  # (b, num_tokens, num_groups * head_dim)
    values = values.view(b, num_tokens, self.num_groups, self.head_dim)  # (b, num_tokens, num_groups, head_dim)

    keys_new = keys.transpose(1, 2)                                      # (b, num_groups, num_tokens, head_dim)
    values_new = values.transpose(1, 2)                                  # (b, num_groups, num_tokens, head_dim)

    # K-V Cache
    if use_cache:
      if self.cache_k is None:
        self.cache_k, self.cache_v = keys_new, values_new
      else:
        self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)  # dim = 2 we are concatenating alog num_tokens 
        self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
      keys, values = self.cache_k, self.cache_v
    else:
      keys, values = keys_new, values_new
    ############

    heads_per_group = self.num_heads // self.num_groups

    # add number of heads in each group K and V to match the query
    keys = keys.repeat_interleave(self.heads_per_group, dim=1)   # (b, num_head, num_tokens, head_dim)
    values = values.repeat_interleave(self.heads_per_group, dim=1)   # (b, num_head, num_tokens, head_dim)

    # Attn scores
    attn_scores = queries @ keys.transpose(2, 3)

    # Apply causal mask
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    device = queries.device
    if use_cache:
      q_positions = torch.arange(self.ptr_current_pos, self.ptr_current_pos + num_tokens_Q, device=device, dtype=torch.long)
      self.ptr_current_pos += num_tokens_Q
    else:
      q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
      self.ptr_current_pos = 0
    k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
    mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

    # Use the mask to fill attention scores
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)

    # Attn weights
    attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)
    assert keys.shape[-1] == self.head_dim

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
    
  def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0