"""Who shares a pocket with the affected patients in the binary one-tail scenario (DN2)? (2026-10-10)

Read-only diagnostic of saved replicates; nothing is refitted. Risk is x1 + 0.8 (x2^2 - 1) and the planted
over-prediction affects focal patients with x2 above its 80th percentile, so the lower tail (x2 below the 20th
percentile) has the same risk contribution but carries no error. For each replicate we take the frozen SBM
pocket labels of the audit patients (audit_inputs.npz) and, for the pockets that contain affected patients,
describe their focal members: affected (upper tail), lower tail, and everyone else. Pockets are weighted by
the number of affected patients they hold, so the shares describe the pocket of a typical affected patient.

Arms, paired by replicate seed (base seed 2026100620):
- default forest (max_features = sqrt(p)): results/night_binary_20261006/learned/DN2
- every feature at every split (max_features = 1.0): results/mfsweep_20261007/bin_mf1.0/learned/DN2

Usage: python simulations/utail_mixing.py --out results/tables --fig-dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"default": ROOT / "results/night_binary_20261006/learned/DN2",
        "all": ROOT / "results/mfsweep_20261007/bin_mf1.0/learned/DN2"}
Q20, Q80 = norm.ppf(.2), norm.ppf(.8)        # region_mask uses standard-normal quantiles (smoke_ladder.Q)
BINS = np.linspace(-3.5, 3.5, 36)


def replicate(path: Path):
    z = np.load(path / "audit_inputs.npz", allow_pickle=True)
    x, s, a, lab = z["x"], z["s"].astype(bool), z["affected"].astype(bool), z["labels_sbm"]
    risk = x[:, 0] + .8 * (x[:, 1] ** 2 - 1)
    foc = s & (lab >= 0)                       # focal patients assigned to a pocket (abstained: label -1)
    aff = a & foc
    pockets, n_aff = np.unique(lab[aff], return_counts=True)
    w = n_aff / n_aff.sum()
    shares = np.zeros(3)
    hist = np.zeros(len(BINS) - 1)
    for c, wc in zip(pockets, w):
        m = foc & (lab == c)
        lower = x[m, 1] < Q20
        shares += wc * np.array([a[m].mean(), lower.mean(), 1 - a[m].mean() - lower.mean()])
        h, _ = np.histogram(x[m, 1], BINS)
        hist += wc * h / m.sum()
    inpk = foc & np.isin(lab, pockets)
    other = inpk & ~a & (x[:, 1] >= Q20)
    return dict(affected=shares[0], lower_tail=shares[1], other=shares[2], n_pockets_with_affected=len(pockets),
                risk_affected=risk[aff].mean(), risk_lower=risk[inpk & (x[:, 1] < Q20)].mean(),
                risk_other=risk[other].mean(), x1_affected=x[aff, 0].mean(), x1_other=x[other, 0].mean()), hist


def figure(hists: dict, summ: pd.DataFrame, fig_dir: Path, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25]))
    g = np.linspace(-3.2, 3.2, 300)
    a.plot(g, .8 * (g ** 2 - 1), color="black", lw=1.6)
    a.axvspan(Q80, 3.2, color="#d7301f", alpha=.18, lw=0)
    a.axvspan(-3.2, Q20, color="#969696", alpha=.25, lw=0)
    a.text((Q80 + 3.2) / 2, 8.1, "affected:\nfocal observations'\nrisk over-predicted", ha="center", va="top", fontsize=7.5,
           color="#a50f15")
    a.text((Q20 - 3.2) / 2, 8.1, "same contribution,\nno error", ha="center", va="top", fontsize=7.5, color="#525252")
    a.set_xlim(-3.2, 3.2)
    a.set_ylim(-1.2, 8.4)
    a.set_xlabel("U-shaped feature $x_2$", fontsize=8.5)
    a.set_ylabel("contribution to the log-odds", fontsize=8.5)
    a.set_title("(a) The scenario", fontsize=9.5)
    a.tick_params(labelsize=7.5)

    mid = (BINS[:-1] + BINS[1:]) / 2
    width = BINS[1] - BINS[0]
    b.axvspan(Q80, 3.5, color="#d7301f", alpha=.18, lw=0)
    b.axvspan(-3.5, Q20, color="#969696", alpha=.25, lw=0)
    for arm, col, lab in [("default", "#6baed6", "default forest"), ("all", "#08519c", "all features at every split")]:
        s = summ.loc[arm]
        b.step(mid, hists[arm] / width, where="mid", color=col, lw=1.8,
               label=f"{lab}: {s.affected:.0%} affected, {s.lower_tail:.0%} lower tail")
    b.set_xlim(-3.5, 3.5)
    b.set_ylim(0, .86)
    b.set_xlabel("U-shaped feature $x_2$ of focal observations in an affected observation's pocket", fontsize=8.5)
    b.set_ylabel("density", fontsize=8.5)
    b.legend(fontsize=7.3, frameon=False, loc="upper left")
    b.set_title("(b) Who shares a pocket with the affected observations", fontsize=9.5)
    b.tick_params(labelsize=7.5)
    fig.tight_layout()
    for d in (fig_dir, out):
        fig.savefig(d / "fig_utail.pdf", bbox_inches="tight")
        fig.savefig(d / "fig_utail.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "results/figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reps = sorted(set(p.name for p in ARMS["default"].glob("rep_*")) & set(p.name for p in ARMS["all"].glob("rep_*")))
    rows, hists = [], {}
    for arm, base in ARMS.items():
        hs = []
        for r in reps:
            d, h = replicate(base / r)
            rows.append(dict(arm=arm, rep=r, **d))
            hs.append(h)
        hists[arm] = np.mean(hs, axis=0)
    per_rep = pd.DataFrame(rows)
    summ = per_rep.drop(columns="rep").groupby("arm").mean()
    rng_ = per_rep.groupby("arm")[["affected", "lower_tail"]].agg(["min", "max"])
    per_rep.to_csv(args.out / "utail_mixing_reps.csv", index=False)
    summ.to_csv(args.out / "utail_mixing_summary.csv")
    figure(hists, summ, args.fig_dir, args.out)
    (args.out / "utail_mixing_provenance.json").write_text(json.dumps(
        dict(script="simulations/utail_mixing.py", arms={k: str(v.relative_to(ROOT)) for k, v in ARMS.items()},
             paired_replicates=reps, q20=Q20, q80=Q80), indent=1))
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())
    print(rng_.round(3).to_string())


if __name__ == "__main__":
    main()
