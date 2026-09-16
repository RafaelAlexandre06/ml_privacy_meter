import timm
import torch.nn as nn


def create_vit_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 10,
    pretrained: bool = False,
    img_size: int = 224,
) -> nn.Module:
    """
    Create a Vision Transformer model using timm.

    Args:
        model_name (str): timm model name (default: 'vit_base_patch16_224')
        num_classes (int): Number of output classes (default: 10)
        pretrained (bool): Use pretrained weights (default: False)

    Returns:
        nn.Module: ViT model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )
    return model