import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
  def __init__(self, d_model, context_length, num_heads, d_latent, dropout=0.0):
    super().__init__()
    # d_in = d_out = d_model
    assert d_model % num_heads == 0,  "d_model must be divide by num_heads"

    self.d_model = d_model
    self.num_heads = num_heads 
    self.head_dim = d_model // num_heads
    self.d_latent = d_latent 

    # Query
    self.W_q = nn.Linear(d_model, d_model, bias=False)

    # KV Down-Projector  # compress
    self.W_dkv = nn.Linear(d_model, d_latent, bias=False)

    # The new Key and Value Up-Projectors. # decompress
    self.W_uk = nn.Linear(d_latent, d_model, bias=False)
    self.W_uv = nn.Linear(d_latent, d_model, bias=False)

    # The Final output projection
    self.W_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)
    # Causal mask to prevent attending to future tokens.
    self.register_buffer('mask', torch.triu(torch.ones(1, 1, context_length, context_length), diagonal=1).bool())
    
    self.register_buffer("c_kv_cache", None , persistent=False)
    self.ptr_current_pos = 0
    

  def forward(self, x, kv_cache=False):
    b, num_tokens, d_model = x.shape

    q = self.W_q(x) # (b, num_tokens, d_model)
    q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads num_tokens, head_dim)


    c_kv = self.W_dkv(x)  # (b, num_tokens, d_latent)
    
    # caching logic
    if kv_cache:
        if self.c_kv_cache is None:
            updated_c_kv = c_kv
        else:
            updated_c_kv = torch.cat([self.c_kv_cache, c_kv], dim=1)
        self.c_kv_cache = updated_c_kv
    else:
        updated_c_kv = c_kv
            

    k = self.W_uk(updated_c_kv).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)#(b, num_heads num_tokens, head_dim)
    v = self.W_uv(updated_c_kv).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)#(b, num_heads num_tokens, head_dim)
    
    # attn scores
    attn_scores = q @ k.transpose(2, 3)
    
    # causal mask
    num_tokens_Q = q.shape[-2]
    num_tokens_K = k.shape[-2]
    device = q.device
    if kv_cache:
        q_positions = torch.arange(
            self.ptr_current_pos,
            self.ptr_current_pos + num_tokens_Q,
            device=device,
            dtype=torch.long,
        )
        self.ptr_current_pos += num_tokens_Q
    else:
        q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
        self.ptr_current_pos = 0
    k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
    mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
    
    attn_scores = attn_scores.masked_fill(mask_bool, float('-inf'))
    
    # attn weights
    
    attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)

    context_vector = (attn_weights @ v).transpose(1, 2).contiguous().view(b, num_tokens, self.d_model)

    output = self.W_o(context_vector)
    return output
  
  
  def reset_cache(self):
      self.c_kv_cache = None
      self.ptr_current_pos = 0


