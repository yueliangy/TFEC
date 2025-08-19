from torch import nn
import torch.nn.functional as F

class SamePadConv(nn.Module):   #自适应填充，确保在使用膨胀卷积时输入和输出的序列长度保持一致
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out

class ConvBlock(nn.Module):    #包含两个膨胀卷积层和一个残差连接,可以在扩大感受野的同时避免梯度消失问题
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):    #通过堆叠多个 ConvBlock，使用指数增长的膨胀率在不同时间尺度上提取特征。它的设计适用于时间序列特征提取，能够在捕捉长时间依赖的同时保持较低的参数量
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)

# import torch
# from torch import nn
# import torch.nn.functional as F
#
# class CausalConv1d(nn.Module):
#     """
#     1D 因果卷积层，确保输出不依赖未来的信息。
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
#         super(CausalConv1d, self).__init__()
#         self.padding = (kernel_size - 1) * dilation
#         self.conv = nn.Conv1d(
#             in_channels, out_channels, kernel_size,
#             padding=self.padding, dilation=dilation
#         )
#
#     def forward(self, x):
#         # Apply convolution and remove extra padding to ensure causality
#         out = self.conv(x)
#         if self.padding > 0:
#             out = out[:, :, :-self.padding]  # 移除未来时间步的填充部分
#         return out
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
#         super().__init__()
#         # 使用因果卷积代替标准卷积
#         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
#         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
#         self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
#
#     def forward(self, x):
#         residual = x if self.projector is None else self.projector(x)
#         x = F.gelu(x)
#         x = self.conv1(x)
#         x = F.gelu(x)
#         x = self.conv2(x)
#         return x + residual
#
# class DilatedConvEncoder(nn.Module):
#     def __init__(self, in_channels, channels, kernel_size):
#         super().__init__()
#         dilation_rates = [1, 2, 4]  # 使用较小的膨胀率
#         self.net = nn.Sequential(*[
#             ConvBlock(
#                 channels[i - 1] if i > 0 else in_channels,
#                 channels[i],
#                 kernel_size=kernel_size,
#                 dilation=dilation_rates[i % len(dilation_rates)],
#                 final=(i == len(channels) - 1)
#             )
#             for i in range(len(channels))
#         ])
#
#     def forward(self, x):
#         return self.net(x)

