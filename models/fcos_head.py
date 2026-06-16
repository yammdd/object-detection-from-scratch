import math
import torch
import torch.nn as nn


class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=5, num_convs=4,
                 num_levels=5, prior_prob=0.01):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels

        # Classification tower
        cls_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                          bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Regression tower
        reg_tower = []
        for i in range(num_convs):
            reg_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                          bias=False)
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Prediction heads
        # Classification: predict class logits
        self.cls_logits = nn.Conv2d(in_channels, num_classes,
                                    kernel_size=3, padding=1)

        # Regression: predict (l, t, r, b) distances
        self.bbox_pred = nn.Conv2d(in_channels, 4,
                                   kernel_size=3, padding=1)

        # Centerness: predict center-ness score
        self.centerness = nn.Conv2d(in_channels, 1,
                                    kernel_size=3, padding=1)

        self.scales = nn.Parameter(torch.ones(num_levels, dtype=torch.float32))

        # Weight initialization
        self._init_weights(prior_prob)

    def _init_weights(self, prior_prob):
        # Tower convolutions
        for module in [self.cls_tower, self.reg_tower]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                            nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Prediction heads
        for layer in [self.cls_logits, self.bbox_pred, self.centerness]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                    nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_init)

    def forward(self, features):
        cls_outputs = []
        reg_outputs = []
        ctr_outputs = []

        for level_idx, feature in enumerate(features):
            # Shared towers
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)

            # Classification branch
            cls_logits = self.cls_logits(cls_feat)
            cls_outputs.append(cls_logits)

            # Regression branch with per-level scaling
            bbox_raw = self.bbox_pred(reg_feat)
            bbox_pred = torch.exp(
                torch.clamp(self.scales[level_idx] * bbox_raw, max=8.0, min=-8.0)
            )
            reg_outputs.append(bbox_pred)

            # Centerness branch (shared with regression tower)
            ctr_logits = self.centerness(reg_feat)
            ctr_outputs.append(ctr_logits)

        return cls_outputs, reg_outputs, ctr_outputs