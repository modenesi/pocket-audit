# UCI Diabetes 130-US Hospitals (1999–2008)

Hospital encounters of patients with diabetes at 130 US hospitals: 101,766 encounters from 71,518 patients
(Strack et al., *BioMed Research International*, 2014).

**Access.** Public, under CC BY 4.0, from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008

Download the zip file from that page and unzip it here.

**Put here.**

| File | SHA-256 |
|---|---|
| `diabetic_data.csv` | `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` |
| `IDS_mapping.csv` | (code descriptions; not read by the code) |

The loader checks the hash of `diabetic_data.csv` and stops if it differs.

**Used by.** `configs/features_uci.yaml`, the diabetes application (`applications/uci_diabetes/`), and the
semi-synthetic simulations (`simulations/semisynthetic_app.py --dataset uci`).
