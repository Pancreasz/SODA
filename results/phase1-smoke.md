# Phase 1 — Kaggle Smoke Test Results (Tasks 3–6)

Run the snippets in `kaggle/phase1_runbook.md` §2. Fill in below.

## Task 3 — data loaders
- xb.shape: `[16, 3, 224, 224]` (expect [16,3,224,224])
- labels sample: `[2, 0, 2, 3, 1, 2, 2, 0]` (expect ints 0..4)
- distinct domain idxs in a train batch: `[0, 1, 3, 4]`
- train/val/test sizes (target=APTOS): `6132` / `1563` / **3662 expected for test**

## Task 4 — backbones + LoRA
| backbone | feat_dim (expect) | out shape | trainable% (LoRA <5%) |
|---|---|---|---|
| resnet50 | 2048 | (2, 2048) | 100.0 (n/a, full FT) |
| dinov2_vits14 | 384 | (2, 384) | 1.05 |
| dinov2_vitb14 | 768 | (2, 768) | 0.53 |

- LoRA-targeted modules (endswith qkv/proj): `['base_model.model.patch_embed.proj', 'base_model.model.blocks.0.attn.qkv', ...]`
- Does `patch_embed.proj` appear? [x] yes → decide whether to tighten target_modules  [ ] no

## Task 5 — heads/loss/decode
- softmax loss / corn loss: `1.6028` / `0.7764` (finite positive)
- CORN class-prob rowsums: `[1. 1. 1.]` (~1.0)
- decoded grades: `[0 0 0 0 0 1 0 0]` (ints 0..4)
