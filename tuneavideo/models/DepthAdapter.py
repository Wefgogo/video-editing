# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops.layers.torch import Rearrange
#
#
# class DepthAdapter(nn.Module):
#     def __init__(self,
#                  cin=1,  # 输入通道数（深度图为1）
#                  channels=[320, 640, 1280, 1280],  # 输出通道列表
#                  nums_rb=2,  # 每层残差块数量
#                  ksize=1,  # 卷积核大小
#                  sk=True,  # 是否使用Selective Kernel
#                  use_conv=False):
#         super().__init__()
#
#         # 时间维度融合：将帧序列 [B,F,C,H,W] -> [B*F,C,H,W]
#         self.time_pool = nn.Sequential(
#             Rearrange('b f c h w -> b c f h w'),
#             nn.Conv3d(cin, cin, kernel_size=(3, 1, 1), padding=(1, 0, 0)),  # 时间维3D卷积
#             nn.ReLU(),
#             Rearrange('b c f h w -> (b f) c h w'),
#             # nn.Flatten(1, 2)  # 合并 B 和 F 维度
#         )
#
#         # 空间特征提取（与骨架适配器结构对齐）
#         self.spatial_layers = nn.ModuleList()
#         in_channels = cin
#         for out_channels in channels:
#             layer = []
#             for _ in range(nums_rb):
#                 layer.append(ResidualBlock(in_channels, out_channels, ksize, sk))
#                 in_channels = out_channels
#             self.spatial_layers.append(nn.Sequential(*layer))
#
#         # 可选最终卷积
#         self.use_conv = use_conv
#         if use_conv:
#             self.final_conv = nn.Conv2d(in_channels, channels[-1], kernel_size=3, padding=1)
#
#     def forward(self, x):
#         """输入: [B,F,C,H,W], 输出: 多尺度特征列表"""
#         B, F, C, H, W = x.shape
#
#         # 1. 时间维度处理
#         x = self.time_pool(x)  # [B,F,C,H,W] -> [B*F,C,H,W]
#
#         # 2. 空间多尺度特征提取
#         features = []
#         for layer in self.spatial_layers:
#             x = layer(x)
#             features.append(x)  # 保存各层输出
#
#         # 3. 恢复时间维度 [B*F,C,H,W] -> [B,F,C,H,W]
#         features = [f.view(B, F, *f.shape[1:]) for f in features]
#
#         # 4. 可选最终卷积
#         if self.use_conv:
#             features[-1] = self.final_conv(features[-1])
#
#         return features  # 返回多尺度特征列表
#
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, ksize=1, sk=False):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=ksize // 2)
#         self.norm = nn.GroupNorm(32, out_channels)  # 组归一化
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=ksize // 2)
#         self.sk = sk
#
#         # Selective Kernel模块（可选）
#         if sk:
#             self.sk_conv = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
#                 nn.Sigmoid()
#             )
#
#         # 输入输出通道不匹配时的调整
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         else:
#             self.shortcut = nn.Identity()
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = F.relu(self.norm(self.conv1(x)))
#         x = self.conv2(x)
#
#         if self.sk:
#             attn = self.sk_conv(x)
#             x = x * attn
#
#         return F.relu(x + residual)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DepthAdapter(nn.Module):
    def __init__(self,
                 cin=1,  # 输入通道数（深度图为1）
                 channels=[320, 640, 1280, 1280],  # 输出通道列表
                 nums_rb=2,  # 每层残差块数量
                 ksize=3,  # 卷积核大小（改为3以获得更好的空间感知）
                 sk=True,  # 是否使用Selective Kernel
                 use_conv=False):
        super().__init__()

        # 时间维度处理（保持不变）
        self.time_pool = nn.Sequential(
            Rearrange('b f c h w -> b c f h w'),
            nn.Conv3d(cin, cin, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            Rearrange('b c f h w -> (b f) c h w'),
        )

        # 空间特征提取（多尺度）
        self.spatial_layers = nn.ModuleList()
        in_channels = cin

        # 定义每个阶段的下采样率
        downsample_steps = [1, 2, 2, 2]  # 1表示不下采样，2表示下采样2倍

        for i, (out_channels, downsample) in enumerate(zip(channels, downsample_steps)):
            layer = []
            for _ in range(nums_rb):
                # 第一个残差块可能需要下采样
                stride = downsample if _ == 0 and downsample != 1 else 1
                layer.append(ResidualBlock(in_channels, out_channels, ksize, sk, stride=stride))
                in_channels = out_channels
            self.spatial_layers.append(nn.Sequential(*layer))

        # 可选最终卷积
        self.use_conv = use_conv
        if use_conv:
            self.final_conv = nn.Conv2d(in_channels, channels[-1], kernel_size=3, padding=1)

    def forward(self, x):
        """输入: [B,F,C,H,W], 输出: 多尺度特征列表"""
        B, F, C, H, W = x.shape

        # 1. 时间维度处理
        x = self.time_pool(x)  # [B,F,C,H,W] -> [B*F,C,H,W]

        # 2. 空间多尺度特征提取
        features = []
        current_res = x
        for layer in self.spatial_layers:
            current_res = layer(current_res)
            features.append(current_res)

        # 3. 恢复时间维度 [B*F,C,H,W] -> [B,F,C,H,W]
        features = [f.view(B, F, *f.shape[1:]) for f in features]

        # 4. 可选最终卷积
        if self.use_conv:
            features[-1] = self.final_conv(features[-1])

        return features


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, sk=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                               stride=stride, padding=ksize // 2)
        self.norm = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize,
                               padding=ksize // 2)
        self.sk = sk
        self.stride = stride

        # Selective Kernel模块（可选）
        if sk:
            self.sk_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # 输入输出通道不匹配或需要下采样时的调整
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(32, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.norm(self.conv1(x)))
        x = self.conv2(x)

        if self.sk:
            attn = self.sk_conv(x)
            x = x * attn

        return F.relu(x + residual)