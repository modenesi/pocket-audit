# Diabetes readmission (UCI Diabetes 130-US Hospitals)

The UCI Diabetes 130-US Hospitals data hold 101,766 hospital encounters of patients with diabetes (1999–2008).
We exclude the 2,423 encounters that ended in death or hospice.

**What we audit.** A LightGBM model of readmission within 30 days, using only information known at discharge (the
feature list is `configs/features_uci.yaml`). Contrasts: Black vs. White (primary), female vs. male and Hispanic
vs. White. Patients are the independent units; the audit uses α = .025.

**Data.** Public (CC BY 4.0). See [`data/uci_diabetes/README.md`](../../data/uci_diabetes/README.md).

**Commands.**

Run the WiDS δ audit too (`applications/wids_icu/`): the supplement figure reads both datasets.

```bash
python -m pocket_audit discover --config configs/features_uci.yaml --run-dir results/uci_v1 \
  --alpha .025 --n-jobs 8 --inference wild_restricted_mammen --n-boot 4999 --abstain-quantile .01 \
  --project-expected-events --se-floor --restricted-loss --min-graph-profiled-share .8 --sbm-tree-cut
python -m pocket_audit audit --run-dir results/uci_v1
mkdir -p results/figures
python applications/paper_figures.py --out results/figures      # the supplement figure
```
