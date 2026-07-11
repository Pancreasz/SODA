# Phase 1 — Kaggle Smoke Test Results (Tasks 3–6)

Run the snippets in `kaggle/phase1_runbook.md` §2. Fill in below.

## Task 3 — data loaders
- xb.shape: _____ (expect [16,3,224,224])
- labels sample: _____ (expect ints 0..4)
- distinct domain idxs in a train batch: _____
- train/val/test sizes (target=APTOS): _____ / _____ / **3662 expected for test**

## Task 4 — backbones + LoRA
| backbone | feat_dim (expect) | out shape | trainable% (LoRA <5%) |
|---|---|---|---|
| resnet50 | 2048 | | (n/a, full FT) |
| dinov2_vits14 | 384 | | |
| dinov2_vitb14 | 768 | | |

- LoRA-targeted modules (endswith qkv/proj): _____
- Does `patch_embed.proj` appear? [ ] yes → decide whether to tighten target_modules  [ ] no

## Task 5 — heads/loss/decode
- softmax loss / corn loss: _____ / _____ (finite positive)
- CORN class-prob rowsums: _____ (~1.0)
- decoded grades: _____ (ints 0..4)
