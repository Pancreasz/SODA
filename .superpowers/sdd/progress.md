# SODA Phase 0 — Progress Ledger

Plan: D:\fundus\docs\plans\2026-07-11-soda-phase0-harness-baselines.md

- Task 1: complete (commits 3ecb726..b912a6f, controller-reviewed; added pyproject for importability)
- Task 2: complete (commit 516670b, reviewed: spec OK, quality approved)
  - Minor (defer to final review): _to_rgb_array PIL/grayscale branches untested; regenerate_masks.py has no per-image try/except (a corrupt image could halt the Task 6 batch — watch for this).
- Task 3: complete (commit ac27947, controller-reviewed; trivial parsing, test passes, real-data run clean). Finding: all 16 splits resolve with 0 missing; datasets are root-relative (no images/ prefix) — Kaggle loader must place datasets under images/<DATASET>. masks/ absent (Task 6 creates). Class imbalance confirmed (PDR rare).
- Task 4: complete (commit 8c772f2, reviewed: spec OK, quality approved; QWK math verified vs sklearn oracle).
  - Minor (defer to final review): confusion matrix uses python zip loop (efficiency nit); n==0/den==0 guards untested; out-of-range labels (>=num_classes) raise unguarded IndexError — add a validation guard when wiring real predictions in Phase 1.
