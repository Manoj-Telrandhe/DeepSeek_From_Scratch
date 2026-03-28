import torch     
import torch.nn as nn      

class MultiQueryAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
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

    # Using a fixed size mask for demonstration. A dynamic one is better in practice.
    self.register_buffer("mask", torch.triu(torch.ones(1, 1, 1024, 1024), diagonal=1))

  def forward(self, x):
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
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    # Now Repeat K and V to match the query head
    keys = keys.repeat(1, self.num_heads, 1, 1)  # (batch_size, num_heads, num_tokens, head_dim)
    values = values.repeat(1, self.num_heads, 1, 1)  # (batch_size, num_heads, num_tokens, head_dim)

    # Attn scores
    attn_scores = queries @ keys.transpose(2, 3)

    # Apply causal mask
    # Original mask truncated to the number of tokens and converted to boolean
    mask_bool = self.mask.bool()[:,:,:num_tokens,:num_tokens]
    # Use the mask to fill attention scores
    attn_scores = attn_scores.masked_fill_(mask_bool, -torch.inf)

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
