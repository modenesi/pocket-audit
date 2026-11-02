# WiDS Datathon 2020

ICU stays from the Global Open Source Severity of Illness Score (GOSSIS) initiative, released for the Women in Data
Science (WiDS) Datathon 2020: 91,713 stays from 147 hospitals, with first-day measurements, in-hospital mortality
and the APACHE IVa predicted probability of death.

**Access.** Restricted. The data are on PhysioNet: https://physionet.org/content/widsdatathon2020/1.0.0/

1. Register for a PhysioNet account.
2. Sign this dataset's data-use agreement, and meet any other requirement shown on its access page.
3. Download the files. Do not share them or commit them anywhere; the agreement forbids redistribution.

**Put here.**

| File | SHA-256 |
|---|---|
| `training_v2.csv` | `27861c711f8c196c2a5c858005075fff61df4fb48f6d350c5120cdf45317adeb` |

The loader checks this hash and stops if the file differs. Check it yourself with `sha256sum training_v2.csv`.

**Used by.** `configs/features_wids.yaml`, the ICU application (`applications/wids_icu/`), and the semi-synthetic
simulations (`simulations/semisynthetic_app.py --dataset wids`).
