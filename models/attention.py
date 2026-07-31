import math
import torch
import torch.nn as nn
from models.hook import HookPoint


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    def __init__(self, dim, n_heads, use_layer_norm, use_rope_2d):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads
        self.use_layer_norm = use_layer_norm
        self.use_rope_2d = use_rope_2d

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # LayerNorm
        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        
        # HookPoints for Q, K, V
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        
        # HookPoints for LayerNorm
        self.hook_q_norm = HookPoint()  # LayerNorm applied to Q
        self.hook_k_norm = HookPoint()  # LayerNorm applied to K
        
        # HookPoints by Heads
        self.hook_q_by_head = HookPoint()
        self.hook_k_by_head = HookPoint()
        self.hook_v_by_head = HookPoint()
        
        # HookPoints for Rope
        self.hook_rope_q = HookPoint()    # Rotary embedding applied to Q
        self.hook_rope_k = HookPoint()    # Rotary embedding applied to K
        
        self.hook_A_raw = HookPoint()
        self.hook_A_softmax = HookPoint()
        self.hook_Av = HookPoint()
        self.hook_output = HookPoint()

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out    

    @staticmethod
    def apply_rotary_emb_individually(x, freqs_cis, dtype):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis_x = Attention.reshape_for_broadcast(freqs_cis, x_)
        x_out = torch.view_as_real(x_ * freqs_cis_x).flatten(3)
        x_out = x_out.to(dtype)
        return x_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        
        # Q, K, V projections with hooks
        xq = self.hook_q(self.wq(x))
        xk = self.hook_k(self.wk(x))
        xv = self.hook_v(self.wv(x))
        dtype = xq.dtype

        # LayerNorm Hook
        if self.use_layer_norm:
            xq = self.hook_q_norm(self.q_norm(xq))
            xk = self.hook_k_norm(self.k_norm(xk))
        else:
            xq = self.hook_q_norm(xq)
            xk = self.hook_k_norm(xk)

        xq = self.hook_q_by_head(xq.view(bsz, seqlen, self.n_heads, self.head_dim))
        xk = self.hook_k_by_head(xk.view(bsz, seqlen, self.n_heads, self.head_dim))
        xv = self.hook_v_by_head(xv.view(bsz, seqlen, self.n_heads, self.head_dim))

        # Rope for Q and K with hooks
        if self.use_rope_2d:
            xq = self.hook_rope_q(self.apply_rotary_emb_individually(xq, freqs_cis, dtype))
            xk = self.hook_rope_k(self.apply_rotary_emb_individually(xk, freqs_cis, dtype))
        else:
            xq = self.hook_rope_q(xq)
            xk = self.hook_rope_k(xk)
        
        xq_permuted = xq.permute(0, 2, 1, 3)    # (B,H,S,D)
        xk_permuted = xk.permute(0, 2, 1, 3)
        xv_permuted = xv.permute(0, 2, 1, 3)        
        
        # Scaled dot-product attention from pytorch
        L, S = xq_permuted.size(-2), xk_permuted.size(-2)
        scale_factor = 1 / math.sqrt(xq_permuted.size(-1))
        
        # Raw attention weights with hook   
        attn_weight = self.hook_A_raw(
            xq_permuted @ xk_permuted.transpose(-2, -1) * scale_factor  # (B, H, S, S)
        )
        
        attn_bias = torch.zeros(L, S, dtype=xq_permuted.dtype, device=xq_permuted.device)        
        attn_weight = attn_weight + attn_bias   
        
        # Softmaxed attention weights with hook
        attn_weight = self.hook_A_softmax(torch.softmax(attn_weight, dim=-1))
        
        # Attention output with hook
        Av = self.hook_Av(attn_weight @ xv_permuted)    # (B, H, S, D)
        Av = Av.permute(0, 2, 1, 3)
        Av = Av.flatten(-2)     # (B, S, H*D)

        # Final output projection with hook
        output = self.hook_output(self.wo(Av))
        
        return output