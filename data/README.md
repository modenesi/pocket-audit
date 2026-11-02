# Data

Nothing in this folder is committed except these READMEs: every dataset keeps its own license and terms of use, so
you download it from its source. Each subfolder's README says where to get the data, which files to put there, and
how to check that you have the same version we used (SHA-256).

| Folder | Dataset | Access | Used by |
|---|---|---|---|
| [`wids2020/`](wids2020/) | WiDS Datathon 2020 (ICU) | PhysioNet account and the dataset's data-use agreement | ICU application; semi-synthetic simulations |
| [`obermeyer2019/`](obermeyer2019/) | Obermeyer et al. (2019) synthetic data | Public; downloaded by a script | Care-management application |
| [`uci_diabetes/`](uci_diabetes/) | UCI Diabetes 130-US Hospitals | Public (CC BY 4.0) | Diabetes application; semi-synthetic simulations |
| [`nih_cxr14/`](nih_cxr14/) | NIH ChestX-ray14 | Public (NIH Clinical Center) | Chest X-ray application |

The simulations in `simulations/` need none of these, except the semi-synthetic designs (WiDS and UCI).
