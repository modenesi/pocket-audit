<h1 align="center">pocket-audit</h1>

<p align="center"><b>Where is a prediction model miscalibrated, and for whom?</b><br>
Local audits in outcome-learned pockets.</p>

---

When decisions act on a model's predicted values, its calibration matters. Yet overall and group-level averages, and
ranking metrics such as the AUC, can hide sets of observations for which a prediction model systematically over- or
under-predicts. Concentrated errors barely move an average, and opposite errors offset each other.

`pocket-audit` asks two questions of any model's predictions:

- **Where** is the model miscalibrated? For each *pocket* of similar observations, it estimates the pocket's
  average prediction error, μ = E[prediction − outcome | pocket].
- **For whom?** Within each pocket, it estimates the gap in that error between two groups,
  δ = μ(focal group) − μ(reference group).

A *pocket* is a set of observations that are similar in how the outcome is predicted from their features. The
pockets are learned from the data, not chosen by hand. The audit needs only features, outcomes and fixed
predictions, not access to the assessed model's internals (similarity comes from a separate auxiliary forest), so it
also works for proprietary scores.

Because the pockets are learned from the outcome, not from any model's errors, one frozen map can audit several
assessed models and compare groups within shared pockets. The forest emphasizes predictive features and needs no
rescaling of numeric features or imputation of their missing values.

## How it works

1. **Split.** Observations, or the independent units that hold them (hospitals, patients), are split into
   *development*, *discovery* and *audit* sets. The audit set is never used to form pockets.
2. **Learn pockets on the discovery set.** A random forest predicts the outcome from the features. Two observations
   are similar when they often land in the same leaves, which weighs every feature by its relevance to the outcome.
   A nested stochastic block model groups the resulting similarity graph into pockets. A ridge-regression grouping
   runs alongside as a linear companion. Protected attributes never enter the similarity.
3. **Freeze and assign.** The pocket map is frozen. Each audit observation is assigned to a pocket, or *abstains*
   when it fits no pocket well.
4. **Audit.** For every pocket with enough data (the *support* rule), μ and δ are estimated with confidence
   intervals that hold jointly across pockets: a wild cluster bootstrap with a max-t critical value, which allows
   for correlated observations within units. A pocket is *flagged* when its interval excludes zero. The *combined
   audit* runs the forest and ridge maps at α/2 each. Pockets with too little data are reported as untested, not
   as clean.

For tested pockets, the intervals also say which pocket-average error sizes are excluded at the stated confidence
level. No flag alone does not establish a small error, and opposite errors can still cancel within a pocket.

**Limits of the error control.** The intervals target simultaneous coverage conditional on the frozen maps. Error
control was evaluated in the paper's simulation designs, not guaranteed for arbitrary data. With few, unequal
dependent units (for example, a few dozen hospitals), the μ audit showed inflated false flags and needs a stricter
level, calibrated separately.

## Install

The default pocket map uses [graph-tool](https://graph-tool.skewed.de/), which is on conda-forge but not on PyPI,
so conda is the easiest route:

```bash
git clone https://github.com/modenesi/pocket-audit.git
cd pocket-audit
conda env create -f environment.yml      # creates the "pocket-audit" environment, installs the package
conda activate pocket-audit
pytest                                   # about 10 seconds
```

Without graph-tool, `pip install -e .` still works; use the Leiden pockets (`run_sbm=False, run_leiden=True`).
To rerun the paper's scripts, also install `pip install -e ".[paper]"` (statsmodels and matplotlib).

**Versions used in the paper.** Python 3.13.1, graph-tool 2.88, NumPy 2.2.2, pandas 2.2.3, SciPy 1.15.1,
scikit-learn 1.9.0, LightGBM 4.7.0, python-igraph 1.0.0, leidenalg 0.12.0. `environment.yml` installs compatible
recent versions; stochastic partitions and the last digits of results can differ across versions.

## Quickstart

```bash
python examples/quickstart.py            # about a minute
```

The example simulates 20,000 patients. Among those with a high value of one feature, the model over-predicts
risk for everyone, and by more for group B; elsewhere it is calibrated. The audit learns 11 pockets and flags one
of them, the pocket that holds most of those patients, for both μ ("where") and δ ("for whom"):

```
 kind pocket  estimate  ci_low  ci_high  flag
delta      5     0.161   0.057    0.266  True
   mu    all     0.015   0.004    0.026  True
   mu      5     0.125   0.078    0.171  True
```

(Only the flagged rows are shown here.) The overall error is 0.015; the error in pocket 5 is about eight times larger.

### On your own data

```python
from pocket_audit.audit import SupportRule
from pocket_audit.data import AuditData, Contrast
from pocket_audit.pipeline import PipelineConfig, audit, discover

data = AuditData(
    name="my_model", X=features_df, y=outcome,               # features: never include protected attributes
    unit=hospital_id, row_id=row_id,                          # unit: the independent unit (or one per row)
    protected=protected_df,                                   # e.g. a "race" column
    contrasts=[Contrast("black_vs_white", "race", focal="Black", reference="White")],
    primary_contrast="black_vs_white",
)
cfg = PipelineConfig(include_mu=True, alpha=0.025, n_jobs=8,                # combined audit: .025 per map
                     support=SupportRule(min_n=100, min_units=15))
frozen = discover(data, cfg, preds={"my_model": predictions})   # stage 1: pockets, from non-audit rows only
res = audit(data, frozen)                                        # stage 2: run once
res["results"]                                                   # one row per pocket x estimand
```

For the combined audit at .05, combine the flags of the forest map (`partition == "sbm"`) and the ridge map
(`partition == "ridge_kmeans"`), each audited at .025. The other partitions in `res["results"]` are comparison
groupings, not part of the combined audit.

The command line runs the δ (and κ) audit from a YAML feature list (see `configs/`), writing every run to a new
directory with a manifest. To include μ, use the Python API with `include_mu=True`.

```bash
pocket-audit discover --config configs/features_uci.yaml --run-dir results/my_run
pocket-audit audit    --run-dir results/my_run
```

## Repository layout

| Folder | What it holds |
|---|---|
| `src/pocket_audit/` | The method: splits, similarity forest and graph, pockets, assignment, audit and inference |
| `tests/` | Unit tests, including small deterministic simulations |
| `examples/` | The quickstart |
| `configs/` | Feature allowlists for the two clinical datasets |
| `simulations/` | Every simulation in the paper, plus the scripts that turn them into its tables and figures |
| `applications/` | The four real-data applications in the paper |
| `data/` | Empty. One folder per dataset, each with a README on how to obtain it |

## Reproducing the paper

- **Simulations** need no data except the two semi-synthetic designs: [`simulations/README.md`](simulations/README.md).
- **Applications**: [`applications/README.md`](applications/README.md). We do not redistribute any dataset. Each
  `data/<dataset>/README.md` says where to get it and where to put it.

Each application's settings were frozen in a written protocol before its audit was run, and each audit was run
once. The ICU analysis of μ is labeled exploratory because those audit outcomes had already been read for an earlier
audit of δ. The scripts enforce the order: the `discover`, `describe` and `validate` stages do not use audit outcomes
to fit models, construct pockets or select analysis settings.

## Citation

If you use this software, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> Modenesi, B., Chen, T., Najafzadehkhoei, S., Wang, L., Diwan, A., and Knudsen, B. (2026). Where Is a Prediction
> Model Miscalibrated, and for Whom? Local Audits in Outcome-Learned Pockets. Manuscript.

## License and acknowledgments

MIT license; see [`LICENSE`](LICENSE). The datasets keep their own licenses and terms of use.

This work was supported by Rocket Companies and the Michigan Institute for Data and AI in Society (MIDAS) at the
University of Michigan. The views expressed are the authors' own and do not represent those of any company or
institution, including the authors' employers and the funders.
