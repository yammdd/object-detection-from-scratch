import torch
import torch.nn as nn

from models.backbone import ResNet50Backbone
from models.fpn import FPN
from models.fcos_head import FCOSHead


FPN_STRIDES = [8, 16, 32, 64, 128]
FPN_SCALE_RANGES = [
    (0, 64),       # P3 - small objects
    (64, 128),     # P4
    (128, 256),    # P5
    (256, 512),    # P6
    (512, 1e6),    # P7 - very large objects
]


class FCOSDetector(nn.Module):
    def __init__(self, num_classes=5, pretrained_backbone=True):
        super().__init__()

        self.num_classes = num_classes
        self.strides = FPN_STRIDES
        self.scale_ranges = FPN_SCALE_RANGES

        # Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        # FPN Neck
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,  # (512, 1024, 2048)
            out_channels=256
        )

        # FCOS Head
        self.head = FCOSHead(
            in_channels=256,
            num_classes=num_classes,
            num_convs=4,
            num_levels=len(FPN_STRIDES)
        )

    def forward(self, images):
        # Extract multi-scale features
        c3, c4, c5 = self.backbone(images)

        # Build feature pyramid
        features = self.fpn(c3, c4, c5)  # [P3, P4, P5, P6, P7]

        # Predict
        cls_outputs, reg_outputs, ctr_outputs = self.head(features)

        return cls_outputs, reg_outputs, ctr_outputs

    def freeze_backbone_stem(self):
        self.backbone.freeze_stem()

    def unfreeze_backbone(self):
        self.backbone.unfreeze_all()

    def get_param_groups(self, base_lr, weight_decay=1e-4):
        # Backbone groups (lower lr)
        backbone_groups = self.backbone.get_param_groups(base_lr)

        # FPN group
        fpn_group = {
            'params': list(self.fpn.parameters()),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'fpn'
        }

        # Head group
        head_params = []
        head_scale_params = []
        for name, param in self.head.named_parameters():
            if 'scales' in name:
                head_scale_params.append(param)
            else:
                head_params.append(param)

        head_group = {
            'params': head_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'head'
        }

        head_scale_group = {
            'params': head_scale_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'name': 'head_scales'
        }

        return backbone_groups + [fpn_group, head_group, head_scale_group]

    def compute_grid_points(self, feature_shapes, device):
        all_points = []
        all_strides = []
        level_start_indices = []
        total = 0

        for level_idx, (h, w) in enumerate(feature_shapes):
            stride = self.strides[level_idx]

            shifts_x = (torch.arange(0, w, dtype=torch.float32,
                                     device=device) + 0.5) * stride
            shifts_y = (torch.arange(0, h, dtype=torch.float32,
                                     device=device) + 0.5) * stride

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x,
                                               indexing='ij')
            points = torch.stack([shift_x.reshape(-1),
                                  shift_y.reshape(-1)], dim=1)  # (H*W, 2)

            strides = torch.full((points.shape[0],), stride,
                                 dtype=torch.float32, device=device)

            level_start_indices.append(total)
            total += points.shape[0]

            all_points.append(points)
            all_strides.append(strides)

        all_points = torch.cat(all_points, dim=0)    # (N_total, 2)
        all_strides = torch.cat(all_strides, dim=0)  # (N_total,)

        return all_points, all_strides, level_start_indices

    def count_parameters(self):
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        fpn_params = sum(p.numel() for p in self.fpn.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total = backbone_params + fpn_params + head_params
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)

        print(f"\n{'='*50}")
        print(f"  FCOS Detector - Parameter Count")
        print(f"{'='*50}")
        print(f"  Backbone (ResNet-50): {backbone_params:>12,}")
        print(f"  FPN Neck:            {fpn_params:>12,}")
        print(f"  FCOS Head:           {head_params:>12,}")
        print(f"  {'-'*36}")
        print(f"  Total:               {total:>12,}")
        print(f"  Trainable:           {trainable:>12,}")
        print(f"{'='*50}\n")

        return total, trainable