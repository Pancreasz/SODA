"""Backbone factory: ResNet-50 and DINOv2 ViT (frozen + LoRA)."""
from __future__ import annotations

import timm
import torch.nn as nn

_TIMM_NAMES = {
    "resnet50": "resnet50",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}


def build_backbone(name, lora=False, lora_r=8, lora_alpha=16):
    if name not in _TIMM_NAMES:
        raise ValueError(f"Unknown backbone '{name}'. Expected {list(_TIMM_NAMES)}.")
    # num_classes=0 -> return pooled feature vector; img_size=224 for speed.
    kwargs = dict(pretrained=True, num_classes=0)
    if name.startswith("dinov2"):
        kwargs["img_size"] = 224
    model = timm.create_model(_TIMM_NAMES[name], **kwargs)
    feat_dim = model.num_features

    if lora:
        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad = False
        # timm ViT attention linear layers are named 'qkv' and 'proj'
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                         bias="none", target_modules=["qkv", "proj"])
        model = get_peft_model(model, cfg)
    return model, feat_dim
