# Phase 2 — Kaggle Smoke Test Results (Tasks 3–6)

Run the snippets in `kaggle/phase2_runbook.md §2`. Fill in below.

## Task 3 — style-aug bank
- style view shape: _____ (expect (3,224,224)); min/max: _____

## Task 4 — torch alignment parity vs numpy oracle
- severity parity assert: [ ] pass
- EMA-bank parity assert: [ ] pass
- l_proto requires_grad: _____ (expect True)

## Task 5 — align loaders
- train x / x_style shape: _____ / _____ (expect [16,3,224,224] each)
- distinct domains in a train batch: _____ (expect 4)
- test dataset size (APTOS): _____ (expect 3662)

## Task 6 — checkpoint round-trip
- saved keys are trainable-only: [ ] yes
- round-trip weights match: _____ (expect True); cfg returned: _____
