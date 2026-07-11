# Phase 1 — First New Numbers (Task 8)

DINOv2 ViT-B/14 (frozen + LoRA), 50 epochs, AdamW, lr 1e-4, seed 0. DG leave-one-domain-out.
See `kaggle/phase1_runbook.md` §4. This isolates two levers before any alignment loss:
backbone (ResNet→DINOv2) and ordinality (softmax→CORN).

## Per-target AUC
| target | ResNet-ERM | DINOv2-ERM (softmax) | DINOv2+CORN |
|---|---|---|---|
| APTOS | (Task 7) | | |
| DEEPDR | | | |
| FGADR | | | |
| IDRID | | | |
| MESSIDOR | | | |
| RLDR | | | |
| **avg** | | | |

## Per-target QWK (the ordinal payoff — expect CORN ≥ softmax)
| target | DINOv2-ERM (softmax) | DINOv2+CORN |
|---|---|---|
| APTOS | | |
| DEEPDR | | |
| FGADR | | |
| IDRID | | |
| MESSIDOR | | |
| RLDR | | |
| **avg** | | |

## Also record (from each JSON): catastrophic-error rate, MAE, val_best_auc.

## Bars to beat (DG avg AUC, for context): DECO 83.7 · GDRNet 82.6 · ERM 75.9.
## Reading: this is exploratory — a lower number is a real finding. Phase 2 adds the SODA alignment loss.
