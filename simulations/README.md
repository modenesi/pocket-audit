# Simulations

These scripts run every simulation in the paper and turn the results into its tables and figures. They need no
data, except the semi-synthetic designs, which use real WiDS and UCI features (see `../data/`).

Run everything from the repository root. Each command writes to its own directory under `results/`, and the
table scripts expect the directory names shown below. Every replicate records its seed and settings and is
skipped if it already exists, so an interrupted run resumes where it stopped. `--workers` sets the number of
parallel processes; change it to fit your machine. The full set is a multi-day computation; start with the
quick check at the end of this page.

## What each script does

| Script | What it runs or makes |
|---|---|
| `continuous_ladder.py` | Continuous outcomes, independent observations (specs 1–5, masking specs 1m and 4zm) |
| `smoke_ladder.py` | Binary outcomes, n = 50,000, independent observations or 150 dependent units (cells D0–DX1) |
| `semisynthetic_app.py` | Real WiDS or UCI features with simulated outcomes and a planted error |
| `semisynthetic_family_summary.py` | Coverage summary of the dependence stress check |
| `mu_localization_sims.py` | "Where" simulations: a local miscalibration μ planted on saved frozen maps |
| `union_audit.py`, `union_audit_binary.py`, `union_audit_semi.py` | The combined forest + ridge audit, re-audited from saved replicates |
| `paper_v2_tables.py` | Table 1 (false flags) and Figure 2 (localization scoreboard) |
| `mu_sims_tables.py` | The tables of the "where" simulations (main text and supplement) |
| `union_table.py` | Supplement table of the combined audit |
| `mf_sweep_summary.py` | Supplement table of the forest's `max_features` sensitivity |
| `utail_mixing.py` | Supplement figure on a one-tail nonlinear region |

## Commands used in the paper

### 1. Continuous outcomes

```bash
python simulations/continuous_ladder.py learned --out-dir results/contladder_20261005/conf_v1_P_a \
  --specs 1 2 3 4 4c 4z 5 --arms P --reps 0 150 --alt-reps 200 --n-boot 4999 --base-seed 2026100590 --workers 12
python simulations/continuous_ladder.py learned --out-dir results/contladder_20261005/conf_v1_P_b \
  --specs 1 2 3 4 4c 4z 5 --arms P --reps 150 500 --alt-reps 200 --n-boot 4999 --base-seed 2026100590 --workers 12
python simulations/continuous_ladder.py learned --out-dir results/contladder_20261005/night_comp \
  --specs 1 4z --arms P --n-total 10000 --deltas 0 .4 --reps 0 200 --n-boot 1999 --base-seed 2026100600 --workers 12
python simulations/continuous_ladder.py learned --out-dir results/contladder_20261005/night_mask \
  --specs 1m 4zm --arms P --n-total 20000 --deltas 0 .4 .6 --reps 0 150 --n-boot 1999 --base-seed 2026100600 --workers 12
# fresh seeds for the combined audit, and a null-only check at n = 20,000
python simulations/continuous_ladder.py learned --out-dir results/union_conf_20261007/n10k \
  --specs 1 2 3 4 4c 4z 5 --arms P --deltas 0 .4 --reps 0 200 --n-boot 1999 --base-seed 2026100720 --workers 12
python simulations/continuous_ladder.py learned --out-dir results/union_conf_20261007/n20k_mask \
  --specs 1m 4zm --arms P --n-total 20000 --deltas 0 .4 --reps 0 150 --n-boot 1999 --base-seed 2026100720 --workers 12
python simulations/continuous_ladder.py learned --out-dir results/nullcheck_20261007/n20k \
  --specs 1 4z --arms P --n-total 20000 --deltas 0 --reps 0 400 --n-boot 1999 --base-seed 2026100710 --workers 12
```

### 2. Binary outcomes

Independent observations are audited at α = .025 and 150 dependent units (cells DH0, DH1) at α = .0125.

```bash
B="--abstain-quantile 0.01 --restricted-loss --se-floor"
C="--out-dir results/confirm_20261004 --base-seed 2026100480 --workers 8"
python simulations/smoke_ladder.py learned $C --cells D0 --reps 0 450 --alpha .025 $B
python simulations/smoke_ladder.py learned $C --cells D1 D3 DN2 --reps 0 40 --alpha .025 $B
python simulations/smoke_ladder.py learned $C --cells DH0 --reps 0 450 --alpha .0125 $B
python simulations/smoke_ladder.py learned $C --cells DH1 --reps 0 50 --alpha .0125 $B
python simulations/smoke_ladder.py learned --out-dir results/night_binary_20261006 --cells DM1 DX0 DX1 D1 DN2 \
  --reps 0 30 --alpha .025 $B --extra-baselines --base-seed 2026100620 --workers 8
```

(The paper's runs split some of these replicate ranges across calls; a replicate's seed depends only on the base
seed, the cell and the replicate index, so one call gives the same replicates.)

### 3. Real features, simulated outcomes

These need the WiDS and UCI data in `data/` (see `data/wids2020/README.md` and `data/uci_diabetes/README.md`).

```bash
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/semi_wids \
  --cells null vent05 vent10 sepsis10 --reps 0 20 --workers 10
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/semi_wids_enc \
  --cells null vent10 vent05 --reps 0 50 --role-unit row --workers 10
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/semi_wids_enc \
  --cells sepsis10 --reps 0 20 --role-unit row --workers 10
python simulations/semisynthetic_app.py --dataset uci --out-dir results/app_20261005/semi_uci \
  --cells null inpt05 inpt10 --reps 0 40 --workers 10
python simulations/semisynthetic_app.py --dataset uci --out-dir results/app_20261005/semi_uci_x \
  --cells inpt10 --reps 0 20 --extra-baselines --workers 8
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/semi_wids_enc_x \
  --cells vent10 --reps 0 20 --role-unit row --extra-baselines --workers 6
```

### 4. Forest settings (supplement)

`$B` is defined in step 2.

```bash
for mf in sqrt 0.33 0.5 1.0; do
  python simulations/continuous_ladder.py learned --out-dir results/mfsweep_20261007/cont_mf$mf \
    --specs 1 4 4c 4z 5 --arms P --deltas 0 .4 --reps 0 100 --n-boot 1999 --base-seed 2026100700 \
    --max-features $mf --workers 12
done
python simulations/smoke_ladder.py learned --out-dir results/mfsweep_20261007/bin_mf1.0 --cells D1 DN2 DM1 \
  --reps 0 30 --alpha .025 $B --extra-baselines --base-seed 2026100620 --max-features 1.0 --workers 8
for mf in 1.0 0.33; do
  python simulations/semisynthetic_app.py --dataset wids --out-dir results/mfsweep_20261007/semi_wids_enc_mf$mf \
    --cells vent10 --reps 0 20 --role-unit row --max-features $mf --workers 6
done
```

### 5. "Where" simulations

These reuse the frozen maps saved by steps 1 and 2; nothing is rediscovered.

```bash
python simulations/mu_localization_sims.py run --out results/mu_sims_20261101 --workers 16
python simulations/mu_localization_sims.py summarize --out results/mu_sims_20261101
# retrospective: the single-map level that would hold the false-flag rate at 5% in each design (reported in the text)
python simulations/mu_localization_sims.py calibrate --out results/mu_sims_20261101 --workers 16
```

### 5b. Dependence stress check (supplement)

Outcomes share a hospital-level shock; the WiDS features are real (needs the WiDS data).

```bash
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/dep_wids_enc_rho20 \
  --cells null --reps 0 100 --n-boot 1999 --role-unit row --hosp-rho .2 --family --workers 6
python simulations/semisynthetic_app.py --dataset wids --out-dir results/app_20261005/dep_wids_hosp_rho20 \
  --cells null --reps 0 100 --n-boot 1999 --hosp-rho .2 --family --workers 12
python simulations/semisynthetic_family_summary.py --runs results/app_20261005/dep_wids_enc_rho20 \
  results/app_20261005/dep_wids_hosp_rho20 --out results/tables
```

### 6. Combined audit, tables and figures

`union_audit_semi.py` reloads the WiDS and UCI data, like step 3.

```bash
mkdir -p results/tables results/figures
```

```bash
python simulations/union_audit.py --set design --out results/tables
python simulations/union_audit.py --set fresh --out results/tables
python simulations/union_audit.py --set null20k --out results/tables
python simulations/union_audit_binary.py --out results/tables --max-reps 150
python simulations/union_audit_binary.py --out results/tables --cells D0 DH0 --tag _nulls450
python simulations/union_audit_semi.py --out results/tables
python simulations/union_table.py
python simulations/paper_v2_tables.py --out results/tables --fig-dir results/figures
python simulations/paper_v2_tables.py --table-from results/tables --out results/tables \
  --mu-flags results/mu_sims_20261101/false_flags.csv     # adds the mu block to Table 1
python simulations/mu_sims_tables.py --run results/mu_sims_20261101 --out results/tables/s8_mu_sims_tables.tex
python simulations/mf_sweep_summary.py --out results/tables
python simulations/utail_mixing.py --out results/tables --fig-dir results/figures
```

## A quick check first

One replicate of one design runs in seconds and shows the output layout:

```bash
python simulations/continuous_ladder.py learned --out-dir results/try --specs 1 --arms P --n-total 6000 \
  --deltas .4 --reps 0 1 --n-boot 199 --workers 1
```
