"""Supplement table of the protocol-v2 validation of mu (and delta) from the saved validation summaries.

    python applications/tables_main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results"
OUT = ROOT / "results/tables/s5_mu_validation.tex"
NAMES = {"lightgbm": "LightGBM", "apache": "APACHE IVa", "score": "health-mapped score", "cost_label": "cost label",
         "health_label": "health label", "sbm_cut": "tree cut", "sbm": "SBM", "ridge_kmeans": "ridge",
         "combined": "combined", "null": "null", "effect": "planted error", "weak_null": "weak null"}


def pct(x, lo, hi):
    return f"{100 * x:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def rows(df: pd.DataFrame, data: str) -> list[str]:
    out = []
    for r in df.itertuples():
        fwer = pct(r.fwer, r.fwer_lo, r.fwer_hi) if r.null_coords_per_rep > 0 else "--"
        out.append(f"{data} & {NAMES[r.scenario]} & {NAMES[r.gen_model]} & {NAMES[r.map]} & {r.replicates} & "
                   f"{r.null_coords_per_rep:.0f} & {fwer} & {100 * r.simultaneous_coverage:.1f} \\\\")
    return out


def main():
    w = pd.read_csv(RUN / "wids_mu/validation_summary.csv", keep_default_na=False)
    o = pd.read_csv(RUN / "obermeyer/validation_summary.csv", keep_default_na=False)
    for d in (w, o):
        for c in ("fwer", "fwer_lo", "fwer_hi", "simultaneous_coverage", "null_coords_per_rep"):
            d[c] = d[c].astype(float)
    order = {"null": 0, "weak_null": 1, "effect": 2}
    w = w.sort_values(["scenario", "gen_model", "map"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    o = o.sort_values(["scenario", "gen_model", "map"], key=lambda s: s.map(order) if s.name == "scenario" else s)
    body = rows(w, "ICU") + ["\\addlinespace"] + rows(o, "Care mgmt.")
    tex = r"""\section{Validation of the Local Calibration Error}
\label{app:muval}

Protocol v2 required a validation gate before the new analyses read any audit outcome. Each replicate keeps the real
audit observations, their units, groups and frozen pocket assignments, and draws new outcomes so that the generating
model is calibrated in every pocket (\emph{null}); calibrated in each pocket on average but not at every observation
(\emph{weak null}, ICU only: $p\,[1 + 0.3\,z\,(1-p)]$ with $z = \pm 1$ split at the median heart rate within
each tree-cut pocket, rescaled to the pocket mean; the factor $(1-p)$, added to the protocol's $p(1 + 0.3z)$ in
implementation, bounds the unrescaled perturbation, and final probabilities were range-checked after rescaling); or miscalibrated in a fixed region (\emph{planted error}: true risk 5 points lower
for ventilated stays; 0.5 more conditions for patient-years with at least three prior-year conditions). ICU outcomes
are Bernoulli with a hospital shock $p(1 + v_h(1-p))$, $v_h = s\,U_h$, $U_h \sim \mathrm{Uniform}(-1, 1)$, which keeps
the marginal mean at $p$; $s$ was set from the dispersion of observed-to-expected deaths across non-audit hospitals
(0.68 for LightGBM, 0.56 for APACHE~IVa). Care-management outcomes add residuals resampled within deciles of the fitted
value from development calibration rows and centred within each decile. ICU families hold $\mu$ for both models at
$\alpha = 0.0125$ per map; care-management families hold $\mu$, $\delta$ and $\kappa$ for the three models and two
contrasts at $\alpha = 0.025$ per map ($0.05$ for the combined audit). The false-flag rate counts replicates with any
flagged coordinate whose true value is zero; coverage is the share of replicates in which every tested $\mu$ and
$\delta$ interval covers its true value ($\kappa$ is fixed under this conditional outcome resampling, so these checks do not validate its coverage under repeated sampling of units). Bootstrap $B = 1{,}999$.

\begin{table}[h]
\caption{Validation of the protocol-v2 audits: false-flag rate (\%, 95\% Wilson interval) over true-null coordinates
and simultaneous coverage (\%).}
\label{tab:muval}
\scriptsize
\begin{tabular}{@{}lllrrrlr@{}}
\toprule
Data & Scenario & Generating model & Map & Replicates & Null coords. & False flags & Coverage \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    OUT.write_text(tex)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
