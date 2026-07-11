# SODA Phase 0 — Progress Ledger

Plan: D:\fundus\docs\plans\2026-07-11-soda-phase0-harness-baselines.md

- Task 1: complete (commits 3ecb726..b912a6f, controller-reviewed; added pyproject for importability)
- Task 2: complete (commit 516670b, reviewed: spec OK, quality approved)
  - Minor (defer to final review): _to_rgb_array PIL/grayscale branches untested; regenerate_masks.py has no per-image try/except (a corrupt image could halt the Task 6 batch — watch for this).
- Task 3: complete (commit ac27947, controller-reviewed; trivial parsing, test passes, real-data run clean). Finding: all 16 splits resolve with 0 missing; datasets are root-relative (no images/ prefix) — Kaggle loader must place datasets under images/<DATASET>. masks/ absent (Task 6 creates). Class imbalance confirmed (PDR rare).
- Task 4: complete (commit 8c772f2, reviewed: spec OK, quality approved; QWK math verified vs sklearn oracle).
  - Minor (defer to final review): confusion matrix uses python zip loop (efficiency nit); n==0/den==0 guards untested; out-of-range labels (>=num_classes) raise unguarded IndexError — add a validation guard when wiring real predictions in Phase 1.
- Task 5: complete (commit b8a2891, reviewed: spec OK, quality approved; bootstrap percentile logic sound). Full suite 11/11 green.
  - Minor (Phase 1): bootstrap_ci wired to macro_ovo_auc can raise if a resample omits a class — add class-coverage handling in Phase 1 (already noted in plan).
- Task 7: complete (runbook kaggle/reproduce_baselines.md committed). Kaggle execution (actual ERM/GDRNet runs) is the user's step.
- Task 6: complete (commit adds hardened regenerate_masks.py + preview_mask.py; findings recorded). Mask run: 11,357 masks, 0 failures, all 512x512, coverage 0.69-0.79 (healthy). DDR/EYEPACS skipped per convention.

ALL PHASE 0 TASKS COMPLETE. Next: final whole-branch review, then finishing-a-development-branch.

---

# SODA Phase 1 — Progress Ledger

Plan: D:\fundus\docs\plans\2026-07-11-soda-phase1-dinov2-corn.md
Base commit at Phase 1 start: cfb6fbf (chore: phase 0 progress ledger)
Executing via subagent-driven-development on branch `main` (continues Phase 0 convention).

Scope note: Tasks 1-2 are local pure-numpy (pytest-verified here). Tasks 3-6 write
Kaggle-only torch code (committed here; Kaggle smoke tests deferred to the user — this
machine has no torch/GPU). Tasks 7-8 are Kaggle execution = the user's step (not run here).

- Task 1: complete (commits cfb6fbf..4a97dd7, review clean)
  - Minor (defer to final review): `dg_config.py:7` mutable default arg `domains=DG_DOMAINS` (read-only now, foot-gun if mutated); `domains=` override param path untested (matches brief tests).
- Task 2: complete (commits 4a97dd7..70a2547, 17/17 suite green)
  - **PLAN-VS-REVIEW (for user / final review):** reviewer rated Important — `ordinal_encoding.py:48` `np.clip(probs,0,1)` silently breaks sum-to-1 on a *non-monotone* `cum` row (counterexample cum=[[0.1,0.9]] -> clipped sum 1.8, no error/warning). Controller adjudication: DOWNGRADED to Minor in-context — the ONLY producer of `cum` is Task 5 `_corn_cumprobs` = cumprod(sigmoid(logits)), which is structurally monotone non-increasing, so the non-monotone row is unreachable and the clip only squashes ~1e-16 float noise (sum-to-1 holds to float precision). Line is VERBATIM from approved plan (plan line 172), so kept as-is; user decides whether to harden (assert monotonicity, or renormalize-after-clip) in a later pass.
  - Minor: no per-function docstrings stating the monotone-non-increasing precondition; no regression test for non-monotone/out-of-range `cum`.
- Task 3: complete (commits 70a2547..3be00c8, review clean; py_compile OK, __init__ empty, soda/__init__ untouched). Controller-verified: all 6 DG domains have train+crossval split files matching DG_DOMAINS; APTOS 2921+741=3662 test count matches plan expectation. Kaggle smoke test deferred to user.
  - Minor (final review): no `assert target not in sources` guard; `open()` has no explicit encoding; `_read_split` raises bare FileNotFoundError with no context on a missing split file.
- Task 4: pending
- Task 5: pending
- Task 6: pending
- Task 7 (Kaggle gate): user-run, not executed here
- Task 8 (Kaggle DG sweep): user-run, not executed here
