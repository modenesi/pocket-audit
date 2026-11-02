"""Paper v2 figures (2026-10-07): masking schematic and real-data application figure.

- fig_masking: a schematic (not a result). Simulated patients in two features; the model over-predicts risk
  for focal patients in one corner and under-predicts it in another, so the global focal-minus-reference gap is
  zero while two pockets carry opposite gaps.
- fig_application: frozen v1 audits (protocol committed before outcomes were read). (a) UCI, primary SBM
  partition, LightGBM, Black vs White: pocket estimates with simultaneous intervals. (b) Interval-based
  equivalence share E(t): share of all focal audit patients (abstained included in the denominator) whose
  pocket was tested and whose simultaneous interval lies inside (-t, t).

Usage: python applications/paper_figures.py --out results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3 (ACM requirement; 2026-10-10)
ROOT = Path(__file__).resolve().parents[1]


def masking(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    n = 900
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    s = rng.random(n) < .3
    region = x1 > np.quantile(x1, .7)
    up, dn = region & (x2 > 0), region & (x2 <= 0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a.scatter(x1[~s], x2[~s], s=6, c="#bdbdbd", lw=0, label="reference group")
    a.scatter(x1[s & ~region], x2[s & ~region], s=9, c="#6a51a3", lw=0, label="focal group, calibrated")
    a.scatter(x1[s & up], x2[s & up], s=18, c="#d7301f", marker="^", lw=0, label="focal, risk over-predicted")
    a.scatter(x1[s & dn], x2[s & dn], s=18, c="#2171b5", marker="v", lw=0, label="focal, risk under-predicted")
    q = np.quantile(x1, .7)
    a.axvline(q, color="black", lw=.8, ls=":")
    a.axhline(0, xmin=(q - a.get_xlim()[0]) / np.ptp(a.get_xlim()), color="black", lw=.8, ls=":")
    a.text(2.25, 2.55, "pocket A", ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=1))
    a.text(2.25, -2.75, "pocket B", ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", pad=1))
    a.set_xlabel("feature 1 (strongly related to the outcome)", fontsize=8.5)
    a.set_ylabel("feature 2", fontsize=8.5)
    a.legend(fontsize=7, loc="upper left", frameon=False, markerscale=1.4)
    a.set_title("(a) Where the model performs poorly for the focal group", fontsize=9.5)
    a.tick_params(labelsize=7.5)

    labels = ["Global\n(all observations)", "Pocket A", "Pocket B", "Other\npockets"]
    est = np.array([0.0, .10, -.10, 0.0])
    half = np.array([.025, .055, .055, .03])
    colors = ["#525252", "#d7301f", "#2171b5", "#969696"]
    b.bar(range(4), est, color=colors, width=.6, alpha=.85)
    b.errorbar(range(4), est, yerr=half, fmt="none", ecolor="black", capsize=3, lw=1)
    b.axhline(0, color="black", lw=.8)
    b.set_xticks(range(4), labels, fontsize=8)
    b.set_ylabel("calibration gap, focal − reference\n(predicted − observed risk)", fontsize=8.5)
    b.set_ylim(-.19, .19)
    b.tick_params(labelsize=7.5)
    b.set_title("(b) The global audit sees nothing; pockets do", fontsize=9.5)
    b.text(3.4, -.18, "illustrative values", ha="right", va="bottom", fontsize=7, color="#737373", style="italic")
    fig.tight_layout()
    fig.savefig(out / "fig_masking.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_masking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def e_curve(res: pd.DataFrame, model: str, contrast: str, partition: str, ts: np.ndarray) -> np.ndarray:
    d = res[(res.partition == partition) & (res.kind == "delta") & (res.family == "primary") & (res.model == model)
            & (res.contrast == contrast)]
    nF = float(d[d.pocket.astype(str) == "all"].n_focal.iloc[0])
    loc = d[(d.pocket.astype(str) != "all") & (d.status == "tested")]
    return np.array([loc.n_focal[(loc.ci_low > -t) & (loc.ci_high < t)].sum() / nF for t in ts])


def application(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = ROOT / "results"
    uci = pd.read_csv(run / "uci_v1/audit_results.csv")
    wids = pd.read_csv(run / "wids_v1/audit_results.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    d = uci[(uci.partition == "sbm") & (uci.kind == "delta") & (uci.family == "primary") & (uci.model == "lightgbm")
            & (uci.contrast == "black_vs_white")]
    glob = d[d.pocket.astype(str) == "all"].iloc[0]
    loc = d[(d.pocket.astype(str) != "all") & (d.status == "tested")].sort_values("estimate").reset_index(drop=True)
    n_untested = int(((d.pocket.astype(str) != "all") & (d.status != "tested")).sum())
    x = np.arange(len(loc))
    a.errorbar(x, loc.estimate, yerr=[loc.estimate - loc.ci_low, loc.ci_high - loc.estimate], fmt="none",
               ecolor="#6baed6", lw=1.2)
    a.scatter(x, loc.estimate, s=8 + 60 * loc.n_focal / loc.n_focal.max(), c="#08519c", zorder=3)
    a.axhline(0, color="black", lw=.8)
    a.axhspan(glob.ci_low, glob.ci_high, color="#fdae6b", alpha=.35, lw=0)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color="#fdae6b", alpha=.35, label="global interval (all patients)")], fontsize=7.5,
             frameon=False, loc="upper left")
    a.set_xticks([])
    a.set_xlabel(f"{len(loc)} tested pockets, ordered by estimate ({n_untested} more could not be tested)", fontsize=8.5)
    a.set_ylabel("δ: Black − White calibration gap\n(LightGBM, simultaneous interval)", fontsize=8.5)
    a.set_title("(a) UCI readmission: no pocket flagged", fontsize=9.5)
    a.tick_params(labelsize=7.5)

    ts = np.linspace(0, .15, 151)
    for lab, res, model, ls, col in [("WiDS, LightGBM", wids, "lightgbm", "-", "#08519c"),
                                     ("WiDS, APACHE IVa", wids, "apache", "--", "#6baed6"),
                                     ("UCI, LightGBM", uci, "lightgbm", "-", "#d94801")]:
        b.step(ts, e_curve(res, model, "black_vs_white", "sbm", ts), where="post", ls=ls, color=col, label=lab, lw=1.6)
    b.set_xlabel("tolerance t (percentage points)", fontsize=8.5)
    b.set_ylabel("share of Black audit stays / encounters\nin pockets that exclude gaps beyond ±t", fontsize=8.5)
    b.set_ylim(0, 1.02)
    b.set_xticks([0, .05, .1, .15], ["0", "5", "10", "15"])
    b.legend(fontsize=7.5, frameon=False, loc="lower right")
    b.set_title("(b) What the intervals rule out, E(t)", fontsize=9.5)
    b.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig_application.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_application.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    masking(args.out)
    application(args.out)


if __name__ == "__main__":
    main()
