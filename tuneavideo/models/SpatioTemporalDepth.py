import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SpatioTemporalAttention(nn.Module):

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.s_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        # self.norm = nn.LayerNorm(channels)
        # self.norm = nn.LayerNorm([channels, 1, 1])
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        B, F, C, H, W = x.shape

        # --- 时间注意力 (沿帧维度) ---
        t_tokens = rearrange(x, 'b f c h w -> (b h w) f c')
        t_out = self.t_attn(t_tokens, t_tokens, t_tokens)[0]
        # print(t_out.shape)
        t_out = rearrange(t_out, '(b h w) f c -> b f c h w', b=B, h=H, w=W)

        # --- 空间注意力 (沿像素维度) ---
        s_tokens = rearrange(x, 'b f c h w -> (b f) (h w) c')
        s_out = self.s_attn(s_tokens, s_tokens, s_tokens)[0]
        s_out = rearrange(s_out, '(b f) (h w) c -> b f c h w', b=B, f=F, h=H, w=W)

        combined = t_out + s_out
        combined = rearrange(combined, 'b f c h w -> (b f) c h w')
        combined = self.norm(combined)
        return rearrange(combined, '(b f) c h w -> b f c h w', b=B, f=F)

        # return self.norm((t_out + s_out).flatten(0, 1)).view(B, F, C, H, W)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, use_attn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=ksize // 2)
        self.norm = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize // 2)
        self.attn = SpatioTemporalAttention(out_channels) if use_attn else None

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.GroupNorm(32, out_channels)
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        if self.attn is not None:
            BxF, C, H, W = x.shape
            # print(x.shape)
            x = x.view(-1, BxF, C, H, W)
            x = rearrange(self.conv1(rearrange(x, 'b f c h w -> (b f) c h w')), '(b f) c h w -> b f c h w', b=1)
            x = self.attn(x)
            x = rearrange(x, 'b f c h w -> (b f) c h w')
        else:
            x = F.relu(self.norm(self.conv1(x)))

        x = F.relu(self.norm(self.conv2(x)))
        return x + residual


class DepthFeatureExtractor(nn.Module):

    def __init__(self, cin=1, channels=[64, 128, 256, 512], attn_positions=[1, 2]):
        super().__init__()
        self.initial_conv = nn.Sequential(
            Rearrange('b f c h w -> b c f h w'),
            nn.Conv3d(cin, channels[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, channels[0]),
            nn.ReLU(),
            Rearrange('b c f h w -> (b f) c h w')
        )

        self.layers = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            use_attn = i in attn_positions
            self.layers.append(
                nn.Sequential(
                    ResidualBlock(in_ch, out_ch, stride=2, use_attn=use_attn),
                    ResidualBlock(out_ch, out_ch, use_attn=use_attn)
                )
            )
            in_ch = out_ch

    def forward(self, x):

        B, F = x.shape[:2]
        x = self.initial_conv(x)

        features = [rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)]
        for layer in self.layers:
            x = layer(x)
            features.append(rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F))

        return features