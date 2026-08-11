import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hook import HookPoint
from models.attention import Attention, TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward_GeLU(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_multiplier=1):
    # def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.hidden_dim = ffn_multiplier * hidden_dim
        
        self.w_in = nn.Linear(dim, hidden_dim, bias=False)
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)
        self.b_out = nn.Parameter(torch.zeros(dim))

        self.hook_pre_activation = HookPoint()
        self.hook_post_activation = HookPoint() # Also used for the neutralization
        
    def forward(self, x):
        x = self.hook_pre_activation(self.w_in(x) + self.b_in)
        x = F.gelu(x)
        x = self.hook_post_activation(x)
        x = self.w_out(x) + self.b_out
        return x


class FeedForward_SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, ffn_multiplier=1):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = ffn_multiplier * ((hidden_dim + ffn_multiplier - 1) // ffn_multiplier)
        
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        self.hook_silu_gating = HookPoint()
        self.hook_out = HookPoint()

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        x = self.hook_silu_gating(self._forward_silu_gating(self.w1(x), self.w3(x)))
        x = self.hook_out(self.w2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        norm_eps,
        ffn_type,
        use_ffn_sandwich,
        ffn_dim,
        ffn_dim_multiplier,
        use_layer_norm,
        use_rope_2d,
        use_ffn_residual,
    ):
        super().__init__()
        self.dim = dim
        self.attention = Attention(dim, n_heads, use_layer_norm, use_rope_2d)     
        self.layer_id = layer_id
        self.use_layer_norm = use_layer_norm            
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_type = ffn_type
        self.use_ffn_sandwich = use_ffn_sandwich
        if ffn_type == "gelu":
            self.feed_forward_in = FeedForward_GeLU(dim, ffn_dim, ffn_dim_multiplier) if use_ffn_sandwich else None
            self.feed_forward_out = FeedForward_GeLU(dim, ffn_dim, ffn_dim_multiplier)
        elif ffn_type == "swiglu":
            self.feed_forward_in = FeedForward_SwiGLU(dim, ffn_dim, ffn_dim_multiplier) if use_ffn_sandwich else None
            self.feed_forward_out = FeedForward_SwiGLU(dim, ffn_dim, ffn_dim_multiplier)
        else:
            raise ValueError(f"Unexpected ffn_type : {ffn_type}")
        self.ffn_in_norm = nn.LayerNorm(dim, eps=norm_eps) if use_ffn_sandwich else None
        self.ffn_out_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.use_ffn_residual = use_ffn_residual

        # adaLN_modulation output size: 6*dim for full block, 3*dim for only attention
        self.num_adaln_chunks = 9 if use_ffn_sandwich else 6
        # self.adaln_output_dim = self.num_adaln_chunks * dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), self.num_adaln_chunks * dim, bias=True),
        )
        
        # Hookpoints for Prepended FFN in FFN-Sandwich Case
        self.hook_ffn_in_norm = HookPoint() if use_ffn_sandwich else None
        self.hook_ffn_in_adaLN = HookPoint() if use_ffn_sandwich else None
        self.hook_ffn_in = HookPoint() if use_ffn_sandwich else None
        self.hook_ffn_in_gating = HookPoint() if use_ffn_sandwich else None
                
        # HookPoints for each block
        self.hook_attn_norm = HookPoint()
        self.hook_attn_adaLN = HookPoint()  
        self.hook_attn = HookPoint()
        self.hook_attn_gating = HookPoint()
        
        # Regular FFN
        self.hook_ffn_out_norm = HookPoint()
        self.hook_ffn_out_adaLN = HookPoint()
        self.hook_ffn_out = HookPoint()
        self.hook_ffn_out_gating = HookPoint()

    def forward(self, x, freqs_cis, adaln_input):
        if self.use_ffn_sandwich:
            shift_msa, scale_msa, gate_msa, \
                shift_mlp_in, scale_mlp_in, gate_mlp_in,  \
                    shift_mlp_out, scale_mlp_out, gate_mlp_out  = (
                        self.adaLN_modulation(adaln_input).chunk(self.num_adaln_chunks, dim=1)
                    )
        else:
            shift_mlp_in, scale_mlp_in, gate_mlp_in = None, None, None
            shift_msa, scale_msa, gate_msa, \
                shift_mlp_out, scale_mlp_out, gate_mlp_out = (
                    self.adaLN_modulation(adaln_input).chunk(self.num_adaln_chunks, dim=1)
                )
        
        # Hooked FFN-IN
        if self.use_ffn_sandwich:
            if self.use_layer_norm:
                x = self.hook_ffn_in_norm(self.ffn_in_norm(x))
            else:
                x = self.hook_ffn_in_norm(x)
            ffn_in_adaLN = self.hook_ffn_in_adaLN(modulate(x, shift_mlp_in, scale_mlp_in))
            ffn_in = self.hook_ffn_in(self.feed_forward_in(ffn_in_adaLN))
            if self.use_ffn_residual:
                x = self.hook_ffn_in_gating(x + gate_mlp_in.unsqueeze(1) * ffn_in)
            else:
                x = self.hook_ffn_in_gating(gate_mlp_in.unsqueeze(1) * ffn_in)
        
        # Hooked attention block
        if self.use_layer_norm:
            x = self.hook_attn_norm(self.attention_norm(x))
        else:
            x = self.hook_attn_norm(x)
        x_adaLN = self.hook_attn_adaLN(modulate(x, shift_msa, scale_msa))    # Embedding : x + t (adaLN)
        attn = self.hook_attn(self.attention(x_adaLN, freqs_cis))
        x = self.hook_attn_gating(x + gate_msa.unsqueeze(1) * attn)
        
        # Hooked FFN-OUT
        if self.use_layer_norm:
            x = self.hook_ffn_out_norm(self.ffn_out_norm(x))
        else:
            x = self.hook_ffn_out_norm(x)
        ffn_out_adaLN = self.hook_ffn_out_adaLN(modulate(x, shift_mlp_out, scale_mlp_out))
        ffn_out = self.hook_ffn_out(self.feed_forward_out(ffn_out_adaLN))
        
        # FFN Residual Connection
        if self.use_ffn_residual:
            x = self.hook_ffn_out_gating(x + gate_mlp_out.unsqueeze(1) * ffn_out)
        else:
            x = self.hook_ffn_out_gating(gate_mlp_out.unsqueeze(1) * ffn_out)

        return x

    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        ffn_type="gelu",
        use_ffn_sandwich=False,
        ffn_dim=512,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        use_layer_norm=False,
        use_rope_2d=False,
        use_ffn_residual=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.x_embedder = nn.Linear(patch_size * patch_size * in_channels, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    norm_eps,
                    ffn_type,
                    use_ffn_sandwich,
                    ffn_dim,
                    ffn_dim_multiplier,
                    use_layer_norm=use_layer_norm,
                    use_rope_2d=use_rope_2d,
                    use_ffn_residual=use_ffn_residual,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self.dim = dim
        self.n_heads = n_heads
        self.freqs_cis = None  # Will be computed dynamically based on input size
        
        # Give name to all HookPoints in its descendant modules
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = self.input_size // p
        w = (self.input_size * 3) // p  # 3x width after concatenation
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)   # (B, H//p * W//p, C*p*p)
        return x

    def forward(self, x, t, y):
        input_x = x.clone()

        src_channels = x.size(1)
        src_height = x.size(2)
        src_width = x.size(3)

        # Concatenate along W dimension (dim=3)
        x = torch.cat([x, y], dim=3)  # Now shape is (B, C, H, 3*W)
        
        x = self.patchify(x)    # (B, H//p*W//p, C*p*p)
        
        # Compute 2D positional embeddings for the current spatial size
        B, num_patches, _ = x.shape
        H_patches = src_height // self.patch_size
        W_patches = (src_width * 3) // self.patch_size  # 3x width after concatenation
        
        if self.freqs_cis is None or self.freqs_cis.size(0) != num_patches:
            self.freqs_cis = DiT_Llama.precompute_freqs_cis_2d(
                self.dim // self.n_heads, 
                H_patches, 
                W_patches
            ).to(x.device)
        
        self.freqs_cis = self.freqs_cis.to(x.device)
        
        x = self.x_embedder(x)                  # (B, H//p*W//p, dim)
        t_embedded = self.t_embedder(t)         # (N, D)
        adaln_input = t_embedded.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis, adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, 3*W)
        
        network_out_x = x[:, :src_channels, :src_height, :src_width]        
        return network_out_x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        pred, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_pred, uncond_pred = torch.split(pred, len(pred) // 2, dim=0)
        half_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        pred = torch.cat([half_pred, half_pred], dim=0)
        return torch.cat([pred, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis_2d(dim, height, width, theta=10000.0):
        """
        Precompute 2D rotary positional embeddings for height and width.
        dim: dimension to split between height and width
        height: number of patches in height dimension
        width: number of patches in width dimension (3x original width)
        """
        dim_h = dim // 2  # half for height
        dim_w = dim - dim_h  # half for width
        
        # Height frequencies
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        t_h = torch.arange(height)
        freqs_h = torch.outer(t_h, freqs_h).float()
        freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
        
        # Width frequencies
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        t_w = torch.arange(width)
        freqs_w = torch.outer(t_w, freqs_w).float()
        freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w)
        
        # Create 2D grid
        freqs_cis_h = freqs_cis_h.unsqueeze(1).repeat(1, width, 1)  # (H, W, dim_h/2)
        freqs_cis_w = freqs_cis_w.unsqueeze(0).repeat(height, 1, 1)  # (H, W, dim_w/2)
        
        # Concatenate and flatten to (H*W, dim/2)
        freqs_cis_2d = torch.cat([freqs_cis_h, freqs_cis_w], dim=-1)
        freqs_cis_2d = freqs_cis_2d.reshape(-1, dim // 2)
        
        return freqs_cis_2d
    
    
    # Methods for HookPoint management
    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
        
    def hook_points(self):
        return [module for name, module in self.named_modules() if ('hook' in name and ('moments' not in name or 'inject' not in name))]
    
    def moments_hook_points(self):
        return [module for name, module in self.named_modules() if 'moments' in name]
    
    def neutralize_frequency_hook_points(self):
        return [module for name, module in self.named_modules() if 'neutralize' in name]
    
    def inject_hook_points(self):
        return [module for name, module in self.named_modules() if 'inject' in name]
    
    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            if 'moments' in name:
                cache[name] = tensor
            elif 'inject' in name:
                pass
            else:
                cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
                
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')
        
        for mhp in self.moments_hook_points():
            mhp.add_hook(save_hook, 'moments')
            
    def enable_frequency_ablation(self, vectors):
        """
        vectors: A list of vectors u_k, v_k that will be neutralized.
        """
        for name, module in self.named_modules():
            if "hook_post_activation" in name:
                module.neutralize_target_vectors = vectors
                module.add_hook(None, dir='neutralize')

    def disable_frequency_ablation(self):
        for name, module in self.named_modules():
            if "hook_post_activation" in name:
                module.neutralize_target_vectors = None
                module.remove_hooks('fwd')


