"""Save/load only the trainable tensors (LoRA + heads + projection), Kaggle only.

The frozen DINOv2 backbone is re-instantiated from HF on load, so the checkpoint
is a few MB, not the full backbone.
"""
from __future__ import annotations

import torch


def _trainable_keys(net):
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    return names


def save_trainable(net, path, config):
    keys = _trainable_keys(net)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items() if k in keys}
    torch.save({"state": state, "config": config}, path)


def load_trainable(net, path):
    ckpt = torch.load(path, map_location="cpu")
    net.load_state_dict(ckpt["state"], strict=False)
    return ckpt.get("config", {})
