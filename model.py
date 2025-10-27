import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_float = input.half()
        variance = input_float.pow(2).mean(dim=1, keepdim=True)
        input_norm = input_float * torch.rsqrt(variance + self.eps)
        return (input_norm * self.weight.unsqueeze(0).unsqueeze(-1)).type_as(input)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).half() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :].to(dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoformerLayer(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        num_heads: int = 8,
        max_seq_len: int = 10000,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        rope_base: int = 10000
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len, base=rope_base)
        self.dropout = dropout

        self.input_norm = RMSNorm(feature_dim)
        self.qkv_proj = nn.Linear(feature_dim, feature_dim * 3, bias=False)
        self.output_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp_norm = RMSNorm(feature_dim)
        self.mlp_up = nn.Linear(feature_dim, mlp_hidden_dim * 2, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        x_residual = x
        x_norm = self.input_norm(x).transpose(1, 2)

        qkv = self.qkv_proj(x_norm)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(Q, seq_len=T)
        Q = (Q * cos) + (rotate_half(Q) * sin)
        K = (K * cos) + (rotate_half(K) * sin)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, N)
        attn_output = self.output_proj(attn_output).transpose(1, 2)

        x = x_residual + attn_output

        x_residual = x
        x_norm = self.mlp_norm(x).transpose(1, 2)

        mlp_out = self.mlp_up(x_norm)
        gate, values = mlp_out.chunk(2, dim=-1)
        mlp_out = F.silu(gate) * values
        mlp_out = self.mlp_down(mlp_out)

        output = x_residual + mlp_out.transpose(1, 2)

        return output

class Roformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000, 
                 input_drop=0., attention_drop=0., causal=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta
        self.window = window
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)
        
        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size*self.num_head*3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size*self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(RMSNorm(self.input_size),
                                 nn.Conv1d(self.input_size, self.input_size*8, 1, bias=False),
                                 nn.SiLU()
                                )
        self.MLP_output = nn.Conv1d(self.input_size*4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        freq = 1. / (self.theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))
        freq = freq.reshape(1, -1)
        pos = torch.arange(0, self.window).reshape(-1, 1)
        cos_freq = torch.cos(pos*freq)
        sin_freq = torch.sin(pos*freq)
        cos_freq = torch.stack([cos_freq]*2, -1).reshape(self.window, self.hidden_size)
        sin_freq = torch.stack([sin_freq]*2, -1).reshape(self.window, self.hidden_size)

        return cos_freq, sin_freq
    
    def _add_rotary_emb(self, feature, pos):
        N = feature.shape[-1]

        feature_reshape = feature.reshape(-1, N)
        pos = min(pos, self.window-1)
        cos_freq = self.cos_freq[pos]
        sin_freq = self.sin_freq[pos]
        reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
        feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
    
        return feature_rope.reshape(feature.shape)

    def _add_rotary_sequence(self, feature):
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        reverse_sign = torch.from_numpy(np.asarray([-1, 1])).to(feature.device).type(feature.dtype)
        feature_reshape_neg = (torch.flip(feature_reshape.reshape(-1, N//2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(-1, T, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)
    
        return feature_rope.reshape(feature.shape)
    
    def forward(self, input):
        B, _, T = input.shape

        weight = self.weight(self.input_drop(self.input_norm(input))).reshape(B, self.num_head, self.hidden_size*3, T).transpose(-2,-1)
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)
        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V.contiguous(), dropout_p=self.attention_drop, is_causal=self.causal)  # B, num_head, T, N
        attention_output = attention_output.transpose(-2,-1).reshape(B, -1, T)
        output = self.output(attention_output) + input

        gate, z = self.MLP(output).chunk(2, dim=1)
        output = output + self.MLP_output(F.silu(gate) * z)

        return output

class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, expansion: int = 4):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels
        )
        self.norm = RMSNorm(channels)
        self.pwconv1 = nn.Conv1d(channels, channels * expansion, 1)
        self.act = nn.GLU(dim=1)
        self.pwconv2 = nn.Conv1d(channels * expansion // 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class ICB(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.block1 = ConvBlock(channels, kernel_size, 1, )
        self.block2 = ConvBlock(channels, kernel_size, dilation)
        self.block3 = ConvBlock(channels, kernel_size, 1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((channels)), requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x * self.gamma.unsqueeze(0).unsqueeze(-1) + residual


class BSNet(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        kernel_size: int, 
        dilation_rate: int, 
        num_heads: int,
        max_bands: int = 512,
        band_rope_base: int = 10000,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        self.band_net = Roformer(feature_dim, feature_dim, num_head=num_heads, window=max_bands, causal=False)

        self.seq_net = ICB(
            feature_dim, 
            kernel_size=kernel_size, 
            dilation=dilation_rate, 
            layer_scale_init_value=layer_scale_init_value
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, nband, N, T = input.shape
        
        band_input = input.permute(0, 3, 2, 1).reshape(B * T, N, nband)
        band_output = self.band_net(band_input)
        band_output = band_output.view(B, T, N, nband).permute(0, 3, 2, 1)
        
        seq_input = band_output.reshape(B * nband, N, T)
        seq_output = self.seq_net(seq_input)
        output = seq_output.view(B, nband, N, T)
        
        return output

class Renaissance(nn.Module):
    def __init__(
        self,
        n_freqs: int = 2049,
        feature_dim: int = 128,
        layer: int = 9,
        sample_rate: int = 48000,
        dilation_start_layer: int = 3,
        n_bands: int = 80,
        num_heads: int = 16,
        max_seq_len: int = 10000,
        band_rope_base: int = 10000,
        temporal_rope_base: int = 10000
    ):
        super().__init__()
        self.enc_dim = n_freqs
        self.feature_dim = feature_dim
        self.eps = 1e-7
        self.dilation_start_layer = dilation_start_layer
        self.n_bands = n_bands
        self.sr = sample_rate
        self.max_seq_len = max_seq_len
        self.band_rope_base = band_rope_base
        self.temporal_rope_base = temporal_rope_base

        self.band_width = self._generate_mel_bandwidths()
        self.nband = len(self.band_width)
        assert self.enc_dim == sum(self.band_width), "Mel band splitting failed to cover all frequencies."

        self._build_feature_extractor()
        self._build_main_network(layer, num_heads)
        self._build_output_synthesis()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _generate_mel_bandwidths(self) -> List[int]:
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

        min_freq, max_freq = 0.1, self.sr / 2
        min_mel, max_mel = hz_to_mel(min_freq), hz_to_mel(max_freq)

        mel_points = np.linspace(min_mel, max_mel, self.n_bands + 1)
        hz_points = mel_to_hz(mel_points)
        
        bin_width = self.sr / 2 / self.enc_dim
        bw = np.round(np.diff(hz_points) / bin_width).astype(int)
        
        bw = np.maximum(1, bw)
        
        remainder = self.enc_dim - np.sum(bw)
        if remainder != 0:
            sorted_indices = np.argsort(bw)
            op = 1 if remainder > 0 else -1
            indices_to_adjust = sorted_indices if op == 1 else sorted_indices[::-1]
            
            for i in range(abs(remainder)):
                idx = indices_to_adjust[i % len(indices_to_adjust)]
                if bw[idx] + op > 0:
                    bw[idx] += op
        
        if np.sum(bw) != self.enc_dim:
            bw[-1] += self.enc_dim - np.sum(bw)
            
        return bw.tolist()

    def _build_feature_extractor(self):
        self.feature_extractor_layers = nn.ModuleList([
            nn.Sequential(RMSNorm(bw * 2 + 1), nn.Conv1d(bw * 2 + 1, self.feature_dim, 1))
            for bw in self.band_width
        ])
    
    def _build_main_network(self, num_layers, num_heads):
        self.net = nn.ModuleList()
        max_bands = max(512, self.nband * 2)

        layer_scale_init = 1e-6
        
        for i in range(num_layers):
            dilation = min(2 ** max(0, i - self.dilation_start_layer + 1), 4)
            self.net.append(BSNet(
                self.feature_dim, 
                kernel_size=7, 
                dilation_rate=dilation, 
                num_heads=num_heads, 
                max_bands=max_bands, 
                band_rope_base=self.band_rope_base,
                layer_scale_init_value=layer_scale_init
            ))

    def _build_output_synthesis(self):
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(self.feature_dim),
                nn.Conv1d(self.feature_dim, self.feature_dim * 2, 1),
                nn.SiLU(),
                nn.Conv1d(self.feature_dim * 2, bw * 4, kernel_size=1),
                nn.GLU(dim=1),
            ) for bw in self.band_width
        ])

    def spec_band_split(self, spec: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        subband_spec_ri = []
        subband_power = []
        band_idx = 0
        for width in self.band_width:
            this_spec_ri = spec[:, band_idx : band_idx + width, :, :]
            subband_spec_ri.append(this_spec_ri)
            
            power = (this_spec_ri.pow(2).sum(dim=-1)).sum(dim=1, keepdim=True).add(self.eps).sqrt()
            subband_power.append(power)
            band_idx += width
            
        subband_power = torch.cat(subband_power, 1)
        return subband_spec_ri, subband_power

    def feature_extraction(self, input_spec: torch.Tensor) -> torch.Tensor:
        subband_spec_ri, subband_power = self.spec_band_split(input_spec)
        features = []
        for i in range(self.nband):
            power_for_norm = subband_power[:, i:i+1, :].unsqueeze(1)
            norm_spec_ri = subband_spec_ri[i] / (power_for_norm.transpose(2,3) + self.eps)
            B, F_band, T, _ = norm_spec_ri.shape
            norm_spec_flat = norm_spec_ri.permute(0, 3, 1, 2).reshape(B, F_band*2, T)

            log_power_feature = torch.log(power_for_norm.squeeze(1) + self.eps)
            feature_input = torch.cat([norm_spec_flat, log_power_feature], dim=1)
            
            features.append(self.feature_extractor_layers[i](feature_input))
            
        return torch.stack(features, 1)

    def forward(self, input_spec: torch.Tensor) -> torch.Tensor:
        B, F, T, _ = input_spec.shape

        features = self.feature_extraction(input_spec)

        residual_features = features 
        processed = features
        for layer in self.net:
            processed = layer(processed)
        processed = processed + residual_features

        est_spec_bands = []
        for i in range(self.nband):
            band_output = self.output_layers[i](processed[:, i])
            bw = self.band_width[i]
            est_spec_band = band_output.view(B, bw, 2, T).permute(0, 1, 3, 2)
            est_spec_bands.append(est_spec_band)
        est_spec_full = torch.cat(est_spec_bands, dim=1)

        return est_spec_full
