# Phase 2 — Dev Gate (Task 8)

Full SODA on APTOS, ViT-S/14, before the 18-run ViT-B sweep. See `kaggle/phase2_runbook.md §3`.

## 3a. Stability (full SODA, 30 ep, λ1=λ2=0.1)
- NaN/inf seen? [ ] no
- val_auc rises during warm-up (scale 0.00)? [ ] yes
- val_auc holds (no collapse) once scale → 1.00? [ ] yes
- Verdict: [ ] STABLE  [ ] UNSTABLE (debug loss wiring before sweep)

## 3b. λ sweep (ViT-S, APTOS) — baseline to beat: DINOv2+CORN APTOS QWK 0.8596 / cat 0.0431 / AUC ~0.81
| λ1 | λ2 | AUC | QWK | cat |
|---|---|---|---|---|
| 0.05 | 0.05 | | | |
| 0.05 | 0.1 | | | |
| 0.05 | 0.3 | | | |
| 0.1 | 0.05 | | | |
| 0.1 | 0.1 | | | |
| 0.1 | 0.3 | | | |
| 0.3 | 0.05 | | | |
| 0.3 | 0.1 | | | |
| 0.3 | 0.3 | | | |

- **Chosen (λ1*, λ2*):** _____
- **Gate:** [ ] PASS (a setting beats CORN QWK or catastrophic, no AUC collapse → proceed to Task 9)
             [ ] FAIL (STOP; re-check scale schedule / bank fill / mono sign)
