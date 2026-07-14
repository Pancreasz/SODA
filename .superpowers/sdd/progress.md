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
- Task 4: complete (commits 3be00c8..6818fb4, review Approved; py_compile OK, isolation intact). Kaggle smoke test deferred to user.
  - **PLAN-VS-REVIEW / KAGGLE-VERIFY (for user):** reviewer rated Important — `backbones.py:29-30` `target_modules=["qkv","proj"]` matches by endswith, so LoRA also attaches to timm's `patch_embed.proj` (Conv2d stem), not just attention projections — deviates from the plan's stated intent ("LoRA on attention projections"). Value is VERBATIM from approved plan (plan line 380); kept as-is. Practical impact minor (adds ~11k params, applies uniformly to DINOv2-ERM and DINOv2+CORN so does NOT confound the Phase-1 comparison), but for a clean "frozen+LoRA on attention" claim tighten to `["attn.qkv","attn.proj"]`. ACTION at Task-4 Kaggle smoke test: print `[n for n,_ in model.named_modules() if n.endswith(("qkv","proj"))]`; if `patch_embed.proj` appears, tighten target_modules. (See Kaggle handoff checklist.)
  - Minor (final review): unused `import torch.nn as nn` (F401, from plan); no guard against `build_backbone("resnet50", lora=True)` (would freeze all + attach 0 adapters silently).
- Task 5: complete (commits 6818fb4..ca6419d, review Approved; py_compile OK, isolation intact). Reviewer web-verified coral_pytorch `corn_loss(logits, labels, num_classes)` signature and cumprod(sigmoid)+0.5 decode parity — both correct. Confirms Task-2 clip adjudication (cumprod-of-sigmoid is the sole, structurally-monotone producer of `cum`). Kaggle smoke test deferred to user.
  - Minor (final review): unused `import numpy as np` (F401); no per-function docstrings; `corn_decode_grade(cum)` relies on implicit threshold=0.5 default (correct but coupling invisible at call site); terse `raise ValueError(head_kind)`.
- Task 6: complete (commits ca6419d..121a819 initial, +15e3673 hardening fix; py_compile OK, isolation intact). Initial review "Needs fixes" on 4 PLAN-MANDATED methodology gaps; USER APPROVED fixing all four; fix commit 15e3673 re-reviewed -> Spec OK, quality Approved, no regressions. RESOLVED.
  Fixes applied in 15e3673 (approved deviations from plan's literal code, matching its "best test AUC" intent): (1) best-val checkpoint snapshot+restore before test eval; (2) try/except ValueError->NaN guard around per-epoch macro_ovo_auc; (3) set_seed(random/numpy/torch/cuda) + `--seed` CLI flag; (4) `optim`+`seed` recorded in result JSON config; also explicit ValueError on unknown head_kind, and run.py docstring example fixed to the real SGD gate command.
  Residual Minor (final review): all-val-NaN edge case reports val_best_auc=-1.0 (cosmetic; unreachable with 5 combined source domains); full state_dict CPU clone per val-improvement (runtime cost note, negligible).
  Original findings (now resolved) were, all plan-mandated:
  - [Important, gate-relevant] No best-val checkpoint restore: loop tracks best_auc/`best` but `best` is dead code; final test eval uses the LAST-epoch model. Validation gate wording says "reproduce best test AUC ~0.75" (plan line 15) — intent-vs-code mismatch. DGDR 0.7497 reference may be best-val-checkpoint; last-epoch could miss ±0.02. FIX: snapshot state_dict on best val AUC, load before final test eval.
  - [Important] `macro_ovo_auc` in evaluate() has no try/except; raises ValueError if a val fold lacks a DR grade (roc_auc_score ovo, labels=range(5)) -> aborts a multi-hour Kaggle run; JSON only written at end. Risk low-moderate (val = 5 combined source crossval sets). Echoes Phase-0 Task-5 note. FIX: guard the per-epoch AUC call.
  - [Important] No seeding (no torch.manual_seed/np.random.seed) — run-to-run variance vs the tight ±0.02 gate tolerance. FIX: add --seed.
  - [Important] Output `config` JSON omits `optim` (plan line 637 tuple) — can't audit SGD-vs-AdamW from a result file post-hoc. FIX: add "optim" to the recorded tuple.
  - [Minor/context] run.py DOCSTRING example (lines 3-5) omits `--optim sgd --lr 1e-3`; NEUTRALIZED because plan's actual Task-7 gate command (plan line 668) includes both. Also engine.py SGD default `cfg.get("lr",1e-3)` is dead via CLI (CLI always passes lr). Doc nit.
  - [Minor] engine.py:54 non-"softmax" head_kind silently -> CORNHead (argparse restricts CLI, but train_and_eval is public). val_every/wd not CLI-exposed (matches brief).
- Task 7 (Kaggle gate): user-run, not executed here
- Task 8 (Kaggle DG sweep): user-run, not executed here

## Final whole-branch review (cfb6fbf..15e3673) — done inline (opus subagent hit session cap)
Verdict: READY TO HAND OFF. No Critical/Important defects in the branch.
- Architectural invariant verified: soda/train/ isolated (0-byte __init__), never imported by soda/__init__; pure-numpy core imports without torch; 17/17 local tests green; all 5 torch modules py_compile clean.
- Cross-module contracts confirmed at branch tip: engine.py <-> soda.metrics (macro_ovo_auc, QWK, catastrophic, MAE, bootstrap_ci->(point,lo,hi)) and soda.ordinal_encoding all match real signatures. Split semantics validated vs real data (APTOS 2921+741=3662; all 6 DG domains have both split files). CORN decode parity with coral_pytorch confirmed.
- MUST-FIX before Kaggle: none for Task 7 (ResNet gate, no LoRA). For Task 8 (DINOv2), decide the LoRA target_modules over-match at the Task-4 smoke test (tighten to ["attn.qkv","attn.proj"] if patch_embed.proj appears) — optional, doesn't confound the comparison.
- Deferred to Phase 2 cleanup: unused imports (backbones torch.nn, heads numpy); guards/docstrings; ordinal_encoding clip hardening (Minor, unreachable); data.py encoding/FileNotFound context.

## PHASE 1 STATUS: COMPLETE — Tasks 1-6 code + reviews here; Tasks 7-8 Kaggle DONE by user. Results in results/phase1-*.md. DINOv2+CORN DG-avg AUC 0.8184 / QWK 0.6462. LoRA tightened to attention-only (eeabb67). Pushed to origin @4bc9e5f+.

---

# SODA Phase 2 — Progress Ledger

Plan: D:\fundus\docs\plans\2026-07-14-soda-phase2-alignment-loss.md
Spec: D:\fundus\docs\specs\2026-07-14-soda-phase2-alignment-loss-design.md
Base commit at Phase 2 start: ca20faf. Branch main (continues Phase 0/1 convention).
Scope: DG-only full SODA (alignment loss). Tasks 1-2 local pure-numpy (pytest). Tasks 3-7 Kaggle-only
torch (py_compile + Kaggle smoke, deferred to user). Tasks 8-9 = Kaggle dev-gate + 18-run sweep = user's step.
Pre-flight scan clean (bank-sizing bug + tidy-ups already fixed in plan during write-plans self-review;
numpy-oracle/torch-mirror duplication is the intentional local-testability pattern, parity-checked on Kaggle).

- Task 1 (alignment loss numpy oracle): pending
- Task 2 (domain-balanced sampler): pending
- Task 3 (style-aug bank): pending
- Task 4 (torch alignment module): pending
- Task 5 (data: style view + sampler): pending
- Task 6 (checkpoint save/load): pending
- Task 7 (SODA engine + CLI): pending
- Task 8 (Kaggle dev gate): user-run, not executed here
- Task 9 (Kaggle 18-run ablation): user-run, not executed here
Next action (user): run Kaggle Tasks 7 (ResNet-50 ERM validation gate on APTOS, expect AUC ~0.75) then 8 (DINOv2-ERM + DINOv2+CORN DG sweep). Kaggle handoff checklist + runbook prepared. See soda/kaggle/reproduce_baselines.md and results/ (to be filled by the Kaggle runs).
