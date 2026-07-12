# SODA Phase 1 — Kaggle Runbook (fresh `soda/train/` harness)

Executes plan Tasks 3–8 on Kaggle. Tasks 1–2 (pure-numpy) and the code for Tasks 3–6 are
already done and committed locally; here you (a) smoke-test the torch harness, (b) pass the
ResNet-50 reproduction **gate** (Task 7), then (c) produce the first DINOv2 / CORN numbers
(Task 8). Split the 6 DG rotations across your two Kaggle accounts to parallelize.

---

## 0. ⚠️ LAYOUT GOTCHA — read first (differs from the Phase-0 DGDR runbook)

The **fresh** loader (`soda/train/data.py`) resolves each image as `os.path.join(root, rel)`
where `rel` is the split-file path like `APTOS/moderate_npdr/xxx.png`. So it expects the
dataset folders **directly under `root`** and splits under `root/splits/`:

```
/kaggle/working/gd/
  APTOS/ DEEPDR/ FGADR/ IDRID/ MESSIDOR/ RLDR/   <- datasets DIRECTLY under root
  splits/                                          <- *_train.txt / *_crossval.txt
```

This is **NOT** the `root/images/<DS>` layout the Phase-0 DGDR runbook builds. Do not reuse
that symlink block verbatim. Masks are **not needed** in Phase 1 (ERM/DINOv2 use no masks).

### Data setup (fresh-harness layout)
```bash
mkdir -p /kaggle/working/gd/splits
for DS in APTOS DEEPDR FGADR IDRID MESSIDOR RLDR; do
  ln -s /kaggle/input/<ds-slug>/FundusDG_mini/$DS /kaggle/working/gd/$DS
done
cp /kaggle/input/<ds-slug>/FundusDG_mini/splits/*.txt /kaggle/working/gd/splits/
# sanity: expect 3662 for APTOS (2921 train + 741 crossval)
!wc -l /kaggle/working/gd/splits/APTOS_train.txt /kaggle/working/gd/splits/APTOS_crossval.txt
```

## 1. Code + deps
Upload/clone the `soda` repo (repo root contains the `soda/` package dir). From the repo root:
```bash
!pip install -r soda/train/requirements-kaggle.txt
# run everything from the repo root so `python -m soda.train.run` resolves the package
```
The engine imports `soda.metrics` (pure-numpy, already in the repo) — no extra install.

---

## 2. Smoke tests (plan Tasks 3–6, Step "Kaggle smoke test") — record in `results/phase1-smoke.md`

```python
import sys; sys.path.append("/kaggle/working/soda")   # path to repo root
# Task 3 — data
from soda.train.data import build_loaders
tr, va, te = build_loaders("/kaggle/working/gd",
    ["DEEPDR","FGADR","IDRID","MESSIDOR","RLDR"], "APTOS", img_size=224, batch_size=16)
xb, yb, db = next(iter(tr))
print(xb.shape, yb[:8].tolist(), sorted(set(db.tolist())))
print("train/val/test sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
# EXPECT xb == [16,3,224,224]; labels in 0..4; test size == 3662 for APTOS.

# Task 4 — backbones + LoRA
import torch
from soda.train.backbones import build_backbone
for name, lora in [("resnet50", False), ("dinov2_vits14", True), ("dinov2_vitb14", True)]:
    m, d = build_backbone(name, lora=lora); m.eval()
    out = m(torch.randn(2, 3, 224, 224))
    tr_ = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    print(name, "feat_dim", d, "out", tuple(out.shape), "trainable%", round(100*tr_/tot, 2))
# EXPECT feat_dim 2048/384/768; out (2, feat_dim); LoRA trainable% well under 5%.

# >>> Task-4 LoRA adapter check — which modules ACTUALLY got a LoRA adapter <<<
# NOTE: check adapter attachment (lora_A present), NOT module names. patch_embed.proj is a
# permanent part of the architecture and always appears in named_modules() whether or not
# LoRA touched it — enumerating names would give a false "patch_embed present" every time.
mB, _ = build_backbone("dinov2_vitb14", lora=True)
adapted = [n for n, m in mB.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]
print("num LoRA-adapted modules:", len(adapted), "| patch_embed adapted:",
      any("patch_embed" in n for n in adapted))
# EXPECT with target_modules=["attn.qkv","attn.proj"]: 24 adapted, patch_embed adapted False.
# (The earlier bare ["qkv","proj"] wrapped patch_embed.proj too -> 25 adapted; fixed in eeabb67.)

# Task 5 — heads/loss/decode
from soda.train.heads import SoftmaxHead, CORNHead, compute_loss, predict_class_probs, predict_grade
feat = torch.randn(8, 64); y = torch.randint(0, 5, (8,))
sh, ch = SoftmaxHead(64, 5), CORNHead(64, 5)
print("losses", float(compute_loss("softmax", sh(feat), y, 5)),
                float(compute_loss("corn", ch(feat), y, 5)))
cp = predict_class_probs("corn", ch(feat)); print("corn rowsum", cp.sum(1)[:3])
print("grades", predict_grade("corn", ch(feat))[:8])
# EXPECT finite positive losses; CORN rows sum ~1.0; grades ints 0..4.
```

---

## 3. Task 7 — VALIDATION GATE (do NOT skip; do NOT proceed to Task 8 until it passes)

ResNet-50 + softmax + CE on target=APTOS, **full fine-tune, SGD+nesterov, lr 1e-3** (mirrors
DGDR ERM; no `--lora`). This tests the data pipeline, not the optimizer.
```bash
!python -m soda.train.run --root /kaggle/working/gd --backbone resnet50 \
    --head softmax --target APTOS --epochs 100 --lr 1e-3 --optim sgd \
    --batch-size 16 --seed 0 --out erm_APTOS.json
```
Read `erm_APTOS.json` → `auc`. **GATE: test AUC in ~[0.73, 0.77]** (DGDR reference 0.7497).
Record in `results/phase1-validation.md`.
- IN band → harness validated; proceed to Task 8.
- OUT of band → STOP. Diff the fresh harness vs DGDR (transforms, split reading, test =
  train+crossval combine, class order). Do not trust any DINOv2 number until this passes.

Note: with the Phase-1 hardening fix, the reported test metrics are for the **best
source-validation checkpoint** (not the last epoch); `val_best_auc` is also in the JSON.

---

## 4. Task 8 — First new numbers (after the gate passes) — record in `results/phase1-results.md`

### 4a. Single-target lever check (target=APTOS)
```bash
!python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head softmax \
    --lora --target APTOS --epochs 50 --lr 1e-4 --optim adamw --seed 0 --out dinov2erm_APTOS.json
!python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head corn \
    --lora --target APTOS --epochs 50 --lr 1e-4 --optim adamw --seed 0 --out dinov2corn_APTOS.json
```
Sanity: DINOv2 should be at least competitive with ResNet-ERM on APTOS AUC; CORN's QWK should
be ≥ softmax's (the ordinal payoff). A lower number is a real finding, not a failure.

### 4b. Full DG sweep — DINOv2+CORN and DINOv2-ERM (softmax ablation), all 6 targets
For each target in {APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR}, sources = the other five
(the CLI derives them via `dg_config.dg_sources_for_target`):
```bash
!python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head corn \
    --lora --target <TARGET> --epochs 50 --lr 1e-4 --optim adamw --seed 0 --out dinov2corn_<TARGET>.json
!python -m soda.train.run --root /kaggle/working/gd --backbone dinov2_vitb14 --head softmax \
    --lora --target <TARGET> --epochs 50 --lr 1e-4 --optim adamw --seed 0 --out dinov2erm_<TARGET>.json
```
**Parallelize:** run 3 targets on account A, 3 on account B (a single rotation fits in one
~12h session). For cheap iteration first, swap `dinov2_vitb14` → `dinov2_vits14`, then re-run
the recorded numbers on ViT-B.

### 4c. Record the DG table
In `results/phase1-results.md`, build a target × metric table for ResNet-ERM (Task 7 +
any other targets), DINOv2-ERM (softmax), DINOv2+CORN. Report per-target and the 6-target
**average AUC and QWK**. This is the Phase-1 headline: the backbone lever (ResNet→DINOv2)
and the ordinality lever (softmax→CORN), isolated, before any alignment loss.

---

## Notes / caveats carried from the code review
- **AUC guard:** a per-epoch validation AUC prints as `nan` only if a val fold is missing a DR
  grade (won't abort the run). The final test AUC uses the target domain (all grades present).
- **Seeding:** `--seed` seeds numpy/torch/cuda; vary it if you want a variance estimate against
  the ±0.02 gate tolerance.
- **Result JSON** records `backbone/head/target/epochs/lora/lr/optim/seed` under `config` — keep
  the JSONs; they are the audit trail for the results table.
- Bars to beat (for later phases): DG avg AUC — DECO 83.7, GDRNet 82.6, ERM 75.9.
