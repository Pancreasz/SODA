# Reproduce ERM + GDRNet on Kaggle (harness calibration)

Goal: confirm our splits/harness reproduce the published GDRBench DG numbers before
trusting anything downstream. **ERM DG-average AUC ≈ 75.9** is the primary calibration
anchor (ERM uses no masks). **GDRNet DG-average AUC ≈ 82.6** is the secondary check
(GDRNet uses the regenerated masks — a gap here may reflect mask fidelity vs the original
download, not a harness bug).

## 1. Data upload
1. Zip `D:/fundus/GDRBench/FundusDG_mini` (images + regenerated `masks/` + `splits/`) into a
   private Kaggle Dataset. If upload is slow/large, upload only the six DG datasets
   (APTOS DEEPDR FGADR IDRID MESSIDOR RLDR) + their masks + splits; DDR/EYEPACS are not
   needed for the DG calibration.
2. Note the mounted path, e.g. `/kaggle/input/gdrbench-fundusdg/FundusDG_mini`.

## 2. Layout reconciliation (important — see Phase 0 finding)
The DGDR dataloader expects `root/images/<DATASET>/...`, `root/masks/<DATASET>/...`,
`root/splits`. But our split files are **root-relative** (`IDRID/severe_npdr/x.jpg`, and
EYEPACS flat as `EYEPACS/Images/x.jpeg`) with the dataset folders directly under the root —
there is no `images/` parent on disk. So build the structure the loader expects with symlinks:
```
mkdir -p /kaggle/working/gd/{images,splits,masks}
for DS in APTOS DEEPDR FGADR IDRID MESSIDOR RLDR; do
  ln -s /kaggle/input/<ds-slug>/FundusDG_mini/$DS   /kaggle/working/gd/images/$DS
  ln -s /kaggle/input/<ds-slug>/FundusDG_mini/masks/$DS /kaggle/working/gd/masks/$DS
done
cp /kaggle/input/<ds-slug>/FundusDG_mini/splits/* /kaggle/working/gd/splits/
```
Then `--root /kaggle/working/gd`. (EYEPACS is test-only in DG and not among the six DG
targets, so its flat `Images/` layout does not affect this calibration.)

## 3. Clone DGDR and install
Upload `D:/fundus/DGDR` as a Kaggle Dataset or push to a repo, then:
```
!git clone <path-or-repo of DGDR> dgdr && cd dgdr && pip install -r requirements.txt
```

## 4. Run ERM (leave-one-domain-out, 6 rotations)
For each target in {APTOS, DEEPDR, FGADR, IDRID, MESSIDOR, RLDR}, use the OTHER five as
sources. Concrete example (target = IDRID):
```
!cd dgdr && python main.py --root /kaggle/working/gd --algorithm ERM --dg_mode DG \
    --source-domains APTOS DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output erm_IDRID
```
Repeat, rotating the held-out target across all six. Record each run's best test AUC.

## 5. Run GDRNet (same 6 rotations)
```
!cd dgdr && python main.py --root /kaggle/working/gd --algorithm GDRNet --dg_mode DG \
    --source-domains <the other 5> --target-domains <target> --output gdr_<target>
```

## 6. Calibration check
Average the six target AUCs per method. Record a method × target × AUC table plus the two
averages in `results/baseline-calibration.md`.
- **ERM DG-average ≈ 75.9** → confirms splits/harness are correct. If ERM is far off (>~±2),
  STOP and debug the data layout before trusting anything downstream.
- **GDRNet DG-average ≈ 82.6** → confirms masks + FundusAug path. A gap here most likely
  means regenerated-mask fidelity, not a harness bug.

## Notes
- Kaggle sessions cap at ~12 h; a single ERM DG rotation on ResNet-50 is well within that.
- Import the SODA metrics (`soda/metrics`) into the notebook for consistent AUC/QWK scoring:
  `import sys; sys.path.append('/kaggle/input/<soda-slug>/soda'); from soda.metrics.auc import macro_ovo_auc`.
