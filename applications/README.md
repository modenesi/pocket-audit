# Applications

The paper audits four real prediction settings. Each folder has a README with the commands; the data come from
their original sources (see `../data/`), since we do not redistribute any of them.

| Folder | Data | Prediction audited | Where in the paper |
|---|---|---|---|
| [`obermeyer/`](obermeyer/) | Synthetic data released by Obermeyer et al. (2019) | A care-management score, as a prediction of chronic illness, and two models trained on cost and health labels | Main text |
| [`wids_icu/`](wids_icu/) | WiDS Datathon 2020 (ICU stays at 147 hospitals) | APACHE IVa and a LightGBM model of in-hospital mortality | Main text |
| [`uci_diabetes/`](uci_diabetes/) | UCI Diabetes 130-US Hospitals | A LightGBM model of 30-day readmission | Supplement |
| [`chest_xray/`](chest_xray/) | NIH ChestX-ray14 | Two public chest X-ray classifiers, for pleural effusion | Supplement |

Run every command from the repository root.

## Order of the stages

The application scripts run in stages, and only the last one reads the audit observations' outcomes:

1. `discover`: split the data, fit the assessed models where needed, learn and freeze the pocket maps.
2. `describe`: profile the pockets on discovery rows, so they can be named.
3. `validate`: simulate outcomes on the real audit rows and frozen pockets, and check the audit's false-flag rate
   and coverage before the real audit is run.
4. `audit`: read the audit outcomes and run the audit, once.

## Shared tables and figures

After the applications have run:

```bash
mkdir -p results/tables results/figures
python applications/figure_applications.py      # Figure 3: care-management and ICU audits
python applications/tables_main.py              # Supplement: validation of the mu audits
python applications/paper_figures.py --out results/figures   # Supplement: UCI audit; the masking schematic
```
