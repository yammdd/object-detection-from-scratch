import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Stem: conv1 -> bn1 -> relu -> maxpool (stride 4)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Stage 1: stride 4, 256 channels
        self.layer1 = resnet.layer1

        # Stage 2: stride 8, 512 channels -> C3
        self.layer2 = resnet.layer2

        # Stage 3: stride 16, 1024 channels -> C4
        self.layer3 = resnet.layer3

        # Stage 4: stride 32, 2048 channels -> C5
        self.layer4 = resnet.layer4

        # Channel counts for downstream modules
        self.out_channels = [512, 1024, 2048]

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

    def freeze_stem(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self, base_lr):
        return [
            {
                'params': list(self.stem.parameters()) +
                          list(self.layer1.parameters()),
                'lr': base_lr * 0.01,
                'name': 'backbone_stem'
            },
            {
                'params': list(self.layer2.parameters()) +
                          list(self.layer3.parameters()) +
                          list(self.layer4.parameters()),
                'lr': base_lr * 0.1,
                'name': 'backbone_body'
            },
        ]