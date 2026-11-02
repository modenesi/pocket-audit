# Obermeyer et al. (2019) synthetic data

Synthetic patient-years released with Obermeyer, Powers, Vogeli and Mullainathan, "Dissecting racial bias in an
algorithm used to manage the health of populations", *Science* 366 (2019). The authors generated them from their
analytic file with the `synthpop` package; they are not real patients. 48,784 rows, 160 columns.

**Access.** Public, at https://gitlab.com/labsysmed/dissecting-bias. That repository has no license file, so we do
not redistribute the data; download them from the source.

**Get it.**

```bash
python applications/obermeyer/download_data.py
```

The script clones the repository at the pinned commit `f7a3b4d86e0dd3c71b1bd28a6706dc5fd85548ad` (2022-09-03),
copies the files here and checks the hash.

**Put here.**

| File | SHA-256 |
|---|---|
| `data_new.csv` | `5341f90f3a1d330557620af1c734e552abe0ad57053055dd13dbccc6d8384d74` |
| `data_dictionary.md` | (the authors' variable descriptions) |

**Used by.** The care-management application (`applications/obermeyer/`).
