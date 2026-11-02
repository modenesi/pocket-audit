# Care-management algorithm (Obermeyer et al., 2019)

Obermeyer et al. (*Science*, 2019) showed that a widely used care-management algorithm, which predicts cost,
under-served Black patients when cost was read as health need. Their public repository includes a synthetic
version of the data: 48,784 patient-years, generated to resemble the original.

**What we audit.** The outcome is a patient's number of active chronic conditions in a year; the pockets are
learned from the previous year's records. Three predictions are audited:

- `score`: the algorithm's risk score, mapped to a prediction of chronic illness;
- `cost_label`: a model we train on the cost label;
- `health_label`: a model we train on the health label.

Contrasts: Black vs. White (primary) and female vs. male. Each patient-year is treated as an independent unit.

**Data.** See [`data/obermeyer2019/README.md`](../../data/obermeyer2019/README.md). In short:

```bash
python applications/obermeyer/download_data.py     # fetches the pinned version and checks its SHA-256
```

**Commands.**

```bash
python applications/obermeyer/audit_obermeyer.py discover --run-dir results/obermeyer --n-jobs 16
python applications/obermeyer/audit_obermeyer.py describe --run-dir results/obermeyer
python applications/obermeyer/audit_obermeyer.py validate --run-dir results/obermeyer --reps 500 --n-boot 1999 --n-jobs 24 --engine gemm
python applications/obermeyer/audit_obermeyer.py audit    --run-dir results/obermeyer
```

`discover` fits the two trained models on development rows and freezes the pocket maps on discovery rows. `describe`
and `validate` never read an audit outcome; `audit` reads them once. Results are in `results/obermeyer/`.
