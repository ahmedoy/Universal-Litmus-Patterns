import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTSmall14(nn.Module):
    """
    Vision Transformer (small, patch 14×14) backbone with
    a new classification head for CIFAR-10 (10 classes).
    """
    architecture_name = "ViT"
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # create_model with num_classes=0 returns the feature backbone (no head)
        self.backbone = timm.create_model(
            'vit_small_patch14_dinov2.lvd142m',  # DINOv2-S/14
            pretrained=True,
            num_classes=0
        )
        in_features = self.backbone.num_features  # typically 384 or 512 depending on variant
        # new head for CIFAR-10
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # resize CIFAR-10 up to 518×518 just before the backbone
        x = F.interpolate(
            x,
            size=self.backbone.patch_embed.img_size,   # (518, 518)
            mode='bilinear',
            align_corners=False
        )
        x = self.backbone(x)
        return self.head(x)

def load_vit(device: torch.device = None) -> nn.Module:
    """
    Instantiates ViT-Small/14 pretrained backbone + CIFAR-10 head,
    moves it to `device`, and returns it.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTSmall14(num_classes=10)
    return model.to(device)
