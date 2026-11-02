# ICU mortality (WiDS Datathon 2020)

The WiDS Datathon 2020 data hold 91,713 ICU stays from 147 hospitals, with first-day measurements, in-hospital
mortality, and the APACHE IVa predicted probability of death.

**What we audit.** Two predictions of in-hospital mortality: APACHE IVa and a LightGBM model trained on
development hospitals. Stays without a valid APACHE probability are excluded from the APACHE audit. Contrasts:
Black vs. White (primary), female vs. male and Hispanic vs. White. Hospitals are the independent units: whole
hospitals are assigned to development, discovery or audit, and the intervals allow for correlation within a
hospital. Because there are few hospitals, the audit uses α = .0125.

**Data.** Restricted access: you need a credentialed PhysioNet account and must sign its data-use agreement. See
[`data/wids2020/README.md`](../../data/wids2020/README.md).

**Commands.** The first audit (δ, κ) runs from the command line with the feature list in `configs/`:

```bash
python -m pocket_audit discover --config configs/features_wids.yaml --run-dir results/wids_v1 \
  --alpha .0125 --n-jobs 8 --inference wild_restricted_mammen --n-boot 4999 --abstain-quantile .01 \
  --project-expected-events --se-floor --restricted-loss --min-graph-profiled-share .8 --sbm-tree-cut
python -m pocket_audit audit --run-dir results/wids_v1
```

The audit of μ ("where") reuses those frozen pocket maps:

```bash
python applications/wids_icu/mu_audit.py validate --out results/wids_mu --reps 500 --n-boot 1999 --n-jobs 30
python applications/wids_icu/mu_audit.py audit    --out results/wids_mu --n-boot 4999
```

In the paper, the μ analysis is labeled exploratory: it was added after the outcomes of the first audit had been
read.
