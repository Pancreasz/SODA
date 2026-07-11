# Phase 1 — Validation Gate (Task 7)

Fresh-harness ResNet-50 + softmax + CE, target=APTOS, full fine-tune, SGD+nesterov, lr 1e-3,
100 epochs (`kaggle/phase1_runbook.md` §3).

**GATE: test AUC in ~[0.73, 0.77]** (DGDR reference 0.7497; published 75.0).

| field | value |
|---|---|
| test AUC | _____ |
| val_best_auc | _____ |
| QWK | _____ |
| catastrophic (|pred-true|>=2) | _____ |
| MAE | _____ |
| acc | _____ |

- Result JSON: `erm_APTOS.json`
- **Gate verdict:** [ ] PASS (in band → proceed to Task 8)  [ ] FAIL (STOP; debug data pipeline vs DGDR)
- Notes: _____
