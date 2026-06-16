import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels=256):
        super().__init__()

        c3_ch, c4_ch, c5_ch = in_channels

        # Lateral 1×1 convolutions: reduce channel dimension
        self.lateral_c5 = nn.Conv2d(c5_ch, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(c4_ch, out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(c3_ch, out_channels, kernel_size=1)

        # Smooth 3×3 convolutions: reduce aliasing after addition
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   padding=1)

        # Extra levels: P6 from C5, P7 from P6
        # P6 uses C5's lateral output (before smooth) with stride 2
        self.conv_p6 = nn.Conv2d(c5_ch, out_channels, kernel_size=3,
                                 stride=2, padding=1)
        self.conv_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                 stride=2, padding=1)

        self.out_channels = out_channels

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        # P5: lateral only (top of the pyramid)
        p5_lateral = self.lateral_c5(c5)                     # (B, 256, H/32, W/32)

        # P4: lateral + upsampled P5
        p4_lateral = self.lateral_c4(c4)                     # (B, 256, H/16, W/16)
        p5_up = F.interpolate(p5_lateral, size=p4_lateral.shape[2:],
                              mode='nearest')
        p4_fused = p4_lateral + p5_up

        # P3: lateral + upsampled P4
        p3_lateral = self.lateral_c3(c3)                     # (B, 256, H/8, W/8)
        p4_up = F.interpolate(p4_fused, size=p3_lateral.shape[2:],
                              mode='nearest')
        p3_fused = p3_lateral + p4_up

        p3 = self.smooth_p3(p3_fused)
        p4 = self.smooth_p4(p4_fused)
        p5 = self.smooth_p5(p5_lateral)

        p6 = self.conv_p6(c5)                                # (B, 256, H/64, W/64)
        p7 = self.conv_p7(F.relu(p6))                        # (B, 256, H/128, W/128)

        return [p3, p4, p5, p6, p7]