# Phase 1 — First New Numbers (Task 8)

DINOv2 ViT-B/14 (frozen + attention-only LoRA), 50 epochs, AdamW lr 1e-4, seed 42, best-val
checkpoint selection. DG leave-one-domain-out. Isolates two levers before any alignment loss:
backbone (ResNet→DINOv2) and ordinality (softmax→CORN). Source JSONs: D:\fundus\dino\{erm,corn}\.

## Per-target AUC (primary)
| target | ResNet-ERM | DINOv2-ERM (softmax) | DINOv2+CORN |
|---|---|---|---|
| APTOS | 0.7952 | 0.8026 | 0.8133 |
| DEEPDR | — | 0.8496 | 0.8427 |
| FGADR | — | 0.7885 | 0.7640 |
| IDRID | — | 0.8403 | 0.8522 |
| MESSIDOR | — | 0.7749 | 0.8119 |
| RLDR | — | 0.8130 | 0.8264 |
| **avg (6)** | (APTOS only) | **0.8115** | **0.8184** |

## Per-target QWK (the ordinal payoff)
| target | DINOv2-ERM (softmax) | DINOv2+CORN | Δ |
|---|---|---|---|
| APTOS | 0.8196 | 0.8596 | +0.040 |
| DEEPDR | 0.6766 | 0.6946 | +0.018 |
| FGADR | 0.5197 | 0.5817 | +0.062 |
| IDRID | 0.6545 | 0.6117 | −0.043 |
| MESSIDOR | 0.5700 | 0.5813 | +0.011 |
| RLDR | 0.5007 | 0.5482 | +0.048 |
| **avg (6)** | **0.6235** | **0.6462** | **+0.023** |

## Catastrophic-error rate (|pred−true|≥2, lower=better) — CORN wins on ALL 6
| target | softmax | CORN | Δ |
|---|---|---|---|
| APTOS | 0.0956 | 0.0431 | −0.053 |
| DEEPDR | 0.1815 | 0.1425 | −0.039 |
| FGADR | 0.1031 | 0.0787 | −0.024 |
| IDRID | 0.1686 | 0.1492 | −0.019 |
| MESSIDOR | 0.1187 | 0.1124 | −0.006 |
| RLDR | 0.1877 | 0.1726 | −0.015 |
| **avg (6)** | **0.1425** | **0.1164** | **−0.026** |

## Average summary
| model | AUC | QWK | catastrophic | MAE | acc |
|---|---|---|---|---|---|
| ResNet-ERM (APTOS only) | 0.7952 | 0.8154 | 0.0915 | 0.4050 | 0.6901 |
| DINOv2-ERM (softmax) | 0.8115 | 0.6235 | 0.1425 | 0.6318 | 0.5217 |
| DINOv2+CORN | **0.8184** | **0.6462** | **0.1164** | **0.6105** | 0.5120 |

## Conclusion
**Both levers behave as the SODA thesis predicts, with the ordinal head giving its clearest
gains exactly where it should:**

1. **Backbone lever (ResNet→DINOv2).** Only APTOS has a ResNet point (sweep deferred): 0.7952→0.8026
   AUC. DINOv2 6-target avg AUC = 0.8115. Foundation backbone is a solid, controllable base.

2. **Ordinality lever (softmax→CORN) — the headline.** Averaged over 6 DG targets, CORN improves
   every ordinal metric while barely moving AUC:
   - QWK +0.023 (better on 5/6 targets; only IDRID regresses).
   - **Catastrophic-error −0.026, reduced on ALL 6 targets** (0.1425→0.1164, ~18% relative). This is
     the clinically important number — CORN makes far fewer 2+ grade blunders (e.g. severe DR called
     mild), the exact failure mode ordinal structure should prevent.
   - MAE −0.021; AUC +0.007 (better on 4/6).
   - acc −0.010: a small exact-match dip — the expected ordinal-vs-nominal trade (CORN optimizes
     ordering, not exact-bin hits), and a good trade given the catastrophic-error reduction.

**Context vs bars (DG avg AUC): ERM 75.9 · GDRNet 82.6 · DECO 83.7.** DINOv2+CORN = **81.84** —
clearly above ERM, not yet at GDRNet/DECO. Expected: Phase 1 is only backbone + ordinal head; the
SODA **alignment loss** (L_proto/L_consist/L_mono, Phase 2) is where the DG gain is designed to come
from. Phase 1's job was to validate the two building-block levers and the harness — done. (Note: our
macro-OVO AUC may differ slightly from GDRBench's published scheme, so the trustworthy signal is the
internally-consistent softmax-vs-CORN comparison, not the absolute cross-paper number.)

**Caveats:** ResNet baseline is APTOS-only (full 6-target ResNet sweep deferred — cheap add-on for a
fair backbone-lever average). Weakest targets (FGADR, RLDR, MESSIDOR QWK ~0.5–0.58) are where Phase 2
alignment should help most; source-val AUC ~0.92 vs target-test ~0.76–0.85 quantifies the domain gap
Phase 2 targets.
