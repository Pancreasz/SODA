# Phase 2 — SODA DG Ablation Ladder (Task 9)

DINOv2 ViT-B/14 (frozen + attention-only LoRA) + CORN, 50 ep, AdamW lr 1e-4, seed 42, λ*=(_,_) from
the dev gate. DG leave-one-domain-out. Four columns: the Phase-1 DINOv2+CORN baseline, then each
alignment term added in turn. See `kaggle/phase2_runbook.md §4`. JSONs: soda_{c,cp,full}_<T>.json.

## Per-target AUC
| target | DINOv2+CORN (base) | +L_consist | +L_proto | full SODA |
|---|---|---|---|---|
| APTOS | 0.8133 | | | |
| DEEPDR | 0.8427 | | | |
| FGADR | 0.7640 | | | |
| IDRID | 0.8522 | | | |
| MESSIDOR | 0.8119 | | | |
| RLDR | 0.8264 | | | |
| **avg** | **0.8184** | | | |

## Per-target QWK
| target | base | +L_consist | +L_proto | full SODA |
|---|---|---|---|---|
| APTOS | 0.8596 | | | |
| DEEPDR | 0.6946 | | | |
| FGADR | 0.5817 | | | |
| IDRID | 0.6117 | | | |
| MESSIDOR | 0.5813 | | | |
| RLDR | 0.5482 | | | |
| **avg** | **0.6462** | | | |

## Catastrophic-error rate (lower=better)
| target | base | +L_consist | +L_proto | full SODA |
|---|---|---|---|---|
| **avg** | **0.1164** | | | |

## Also record per run: MAE, val_best_auc. Persist soda_full_<T>.pt weights.

## Headline
- full-SODA 6-avg AUC vs bars: ERM 75.9 · GDRNet 82.6 · DECO 83.7 · DINOv2+CORN 81.84 (Phase 1) · full-SODA ____.
- Per-term marginal contribution (which term buys QWK vs catastrophic vs AUC): ____.
- A term that doesn't help is a real finding — report honestly. Next: Phase 3 (ESDG + synthetic domains + ResNet control + CIs).
