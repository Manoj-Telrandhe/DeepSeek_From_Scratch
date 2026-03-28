import torch
import torch.nn as nn

class MultiQueryAttentionKV(nn.Module):
  def __init__(self, d_in, d_out, num_heads, dropout=0.0):
    super().__init__()
    assert d_out % num_heads == 0, "d_model must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads

    self.W_query = nn.Linear(d_in, d_out, bias=False)
    self.W_key = nn.Linear(d_in, self.head_dim , bias=False)    # Single projection for k
    self.W_value = nn.Linear(d_in, self.head_dim , bias=False)  # Single projection for V

    self.out_proj = nn.Linear(d_out, d_out)

    self.dropout = nn.Dropout(dropout)

    self.register_buffer("cache_k", None, persistent=False)
    self.register_buffer("cache_v", None, persistent=False)
    self.ptr_current_pos = 0
    

  def forward(self, x, use_cache=False):
    batch_size, num_tokens, d_in =  x.shape

    # Query
    queries = self.W_query(x)    # (batch_size, num_tokens, d_out)
    
    # Unroll last dim : (batch_size, num_tokens, d_out)  ---> (batch_size, num_tokens, num_heads, head_dim)
    queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
    # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    queries = queries.transpose(1, 2)
    
    # Key & value
    keys = self.W_key(x)      # (batch_size, num_tokens, head_dim)
    values = self.W_value(x)  # (batch_size, num_tokens, head_dim)

    # Unroll last dim : (batch_size, num_tokens, head_dim)  ---> (batch_size, num_tokens, 1, head_dim)
    keys = keys.view(batch_size, num_tokens, 1, self.head_dim)     # only 1 head
    values = values.view(batch_size, num_tokens, 1, self.head_dim)  # only 1 head

    # Transpose: (batch_size, num_tokens, 1, head_dim) -> (batch_size, 1, num_tokens, head_dim)
    keys_new = keys.transpose(1, 2)
    values_new = values.transpose(1, 2)
    
    # K-V cache
    if use_cache:
      if self.cache_k is None:
        self.cache_k, self.cache_v = keys_new, values_new
      else:
        self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)   # dim = 2, concatenate across num tokens
        self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
      keys_base, values_base = self.cache_k, self.cache_v
    else:
      keys_base, values_base = keys_new, values_new


    # Now Repeat K and V to match the query head
    keys = keys.repeat(1, self.num_heads, 1, 1)  # (batch_size, num_heads, num_tokens, head_dim)
    values = values.repeat(1, self.num_heads, 1, 1)  # (batch_size, num_heads, num_tokens, head_dim)

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
    context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.d_out)

    context_vector = self.out_proj(context_vector)

    return context_vector
    
  def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
