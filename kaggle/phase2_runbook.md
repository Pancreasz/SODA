# SODA Phase 2 — Kaggle Runbook (alignment loss, DG)

Executes plan Tasks 3–9 on Kaggle: smoke-test the new torch modules, pass the **dev gate**
(stability + λ sweep on APTOS), then run the **18-run DG ablation ladder**. Builds on the Phase-1
harness. **Phase-2 SODA is DINOv2-only** (`--backbone dinov2_vitb14`); SODA-on-ResNet is Phase 3.

Prereqs: same data layout as Phase 1 (`kaggle/phase1_runbook.md §0` — datasets DIRECTLY under
`/kaggle/working/gd`, splits under `gd/splits/`). `REPO_ROOT` = the soda repo root on Kaggle.
Re-`git pull` the repo so it includes the Phase-2 commits.

## 1. Deps
```python
!pip -q install timm peft coral-pytorch     # torch/torchvision/sklearn already on Kaggle
import sys; sys.path.insert(0, REPO_ROOT)
```
If HF model download fails (the earlier xet incident), use the offline snapshot workaround from before
(`snapshot_download(...)` then run with `HF_HUB_OFFLINE=1`). It has no effect on training math.

## 2. Smoke tests (Tasks 3–6) — record in `results/phase2-smoke.md`
```python
# Task 3 — style-aug bank
from PIL import Image
from soda.train.style_aug import style_transform
t = style_transform(224)(Image.open("/kaggle/working/gd/APTOS/nodr/002c21358ce6.png").convert("RGB"))
print("style view:", tuple(t.shape), float(t.min()), float(t.max()))   # expect (3,224,224)

# Task 4 — torch alignment PARITY vs numpy oracle
import numpy as np, torch
from soda.train.alignment import torch_severity, ProjectionHead, PrototypeBank, l_consist, l_proto, l_mono
from soda import alignment_math as am
rng = np.random.default_rng(0); logits = rng.normal(size=(8,4))
np.testing.assert_allclose(torch_severity(torch.tensor(logits)).numpy(), am.severity_from_logits(logits), atol=1e-10)
z = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(6,4)),dtype=torch.float32), dim=1)
dom = torch.tensor([0,0,1,1,0,1]); gr = torch.tensor([1,1,1,2,2,1])
bank = PrototypeBank(2, dim=4, momentum=0.9); bank.update(z, dom, gr)
bnp = np.zeros((2,5,4)); fnp = np.zeros((2,5),bool); bnp,fnp = am.ema_update(bnp,fnp,z.numpy(),dom.numpy(),gr.numpy(),0.9)
np.testing.assert_allclose(bank.bank.numpy(), bnp, atol=1e-6)
print("parity OK; l_proto grad:", l_proto(z.requires_grad_(), dom, gr, bank).requires_grad)

# Task 5 — align loaders (style view + domain sampler)
from soda.train.data import build_align_loaders
tr,va,te,samp = build_align_loaders("/kaggle/working/gd",
    ["DEEPDR","FGADR","IDRID","MESSIDOR","RLDR"],"APTOS", domains_per_batch=4, per_domain=4)
x,xs,y,d = next(iter(tr)); print("train:", tuple(x.shape), tuple(xs.shape), "domains:", sorted(set(d.tolist())))
print("distinct domains (expect 4):", len(set(d.tolist())), "| test size:", len(te.dataset))  # 3662 for APTOS

# Task 6 — checkpoint round-trip
import torch.nn as nn
from soda.train.checkpoint import save_trainable, load_trainable
m = nn.Sequential(nn.Linear(4,3), nn.Linear(3,2))
for p in m[0].parameters(): p.requires_grad=False
save_trainable(m, "/kaggle/working/ck.pt", {"seed":42})
m2 = nn.Sequential(nn.Linear(4,3), nn.Linear(3,2)); cfg = load_trainable(m2, "/kaggle/working/ck.pt")
print("saved-keys only trainable + round-trip:", torch.allclose(m[1].weight, m2[1].weight), "cfg:", cfg)
```
Expected: style view (3,224,224); parity assertions pass + l_proto grad True; train batch has x & x_style
`[16,3,224,224]` spanning 4 distinct domains, test size 3662; checkpoint saves only trainable keys and round-trips.

## 3. Task 8 — DEV GATE (do NOT skip; gate before the 18-run sweep) — record in `results/phase2-devgate.md`

### 3a. Stability (full SODA, APTOS, ViT-S for speed)
```
!cd {REPO_ROOT} && python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vits14 \
    --head corn --lora --use-consist --use-proto --use-mono --target APTOS \
    --epochs 30 --lambda1 0.1 --lambda2 0.1 --seed 42 --out soda_devgate_APTOS.json
```
Watch printed `val_auc` + `scale`. PASS if: no NaN/inf; val_auc rises during warm-up (scale 0.00); val_auc
does NOT collapse once λ ramps in (scale → 1.00).

### 3b. λ sweep (ViT-S, APTOS, λ1,λ2 ∈ {0.05, 0.1, 0.3})
Run full SODA on APTOS for the 9 (λ1,λ2) settings (`--epochs 30`), out `soda_l{λ1}_{λ2}_APTOS.json`.
**Gate:** at least one setting gives full-SODA **QWK ≥ 0.8596** (the DINOv2+CORN APTOS baseline) OR
**catastrophic ≤ 0.0431**, without an AUC collapse (AUC ≥ ~0.79). Pick the best (λ1*, λ2*).
If none qualifies: STOP, re-check loss wiring (scale schedule, bank fill, mono sign) before the full sweep.

## 4. Task 9 — Full DG ablation ladder (18 runs) at ViT-B, λ* — record in `results/phase2-results.md`
For each target in {APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR} (split across the 2 accounts), run 3 rungs:
```
# +L_consist
!cd {REPO_ROOT} && python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head corn --lora \
    --use-consist --target <T> --epochs 50 --lambda1 <L1*> --lambda2 <L2*> --seed 42 --out soda_c_<T>.json
# +L_consist +L_proto
!cd {REPO_ROOT} && python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head corn --lora \
    --use-consist --use-proto --target <T> --epochs 50 --lambda1 <L1*> --lambda2 <L2*> --seed 42 --out soda_cp_<T>.json
# full SODA (+L_mono), save weights
!cd {REPO_ROOT} && python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head corn --lora \
    --use-consist --use-proto --use-mono --target <T> --epochs 50 --lambda1 <L1*> --lambda2 <L2*> \
    --seed 42 --save-weights --weights-out soda_full_<T>.pt --out soda_full_<T>.json
```
Persist the `soda_full_<T>.pt` weights (*Save Version* / a Kaggle models dataset). Paste each JSON back
here and I'll assemble the ablation table (per-term marginal QWK/catastrophic/AUC + full-SODA 6-avg vs
GDRNet 82.6 / DECO 83.7).

## Notes carried from the code review
- `--batch-size` is ignored for SODA runs (effective batch = `domains-per-batch × per-domain` = 16). Tune
  batch via `--domains-per-batch` / `--per-domain` instead.
- SODA result JSONs omit `qwk_ci` (present in ERM JSONs) — expected; the headline metrics (auc/qwk/cat/mae) are all there.
- Baseline to beat this phase: your own DINOv2+CORN (results/phase1-results.md). Bars: GDRNet 82.6, DECO 83.7.
