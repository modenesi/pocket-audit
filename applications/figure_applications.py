"""Figure of the protocol-v2 applications: where (mu) and for whom (delta), from saved audit outputs.

    python applications/figure_applications.py

(a) care-management data, mu by forest (SBM) pocket; (b) the same pockets, Black - White delta; (c) ICU, mu by tree-cut
pocket (exploratory). Pocket names come from discovery-row profiles (results/obermeyer/pocket_profiles_discovery.csv and
results/wids_v1/pocket_descriptions_sbm_cut.csv); a marker is filled when the pocket is flagged.
Drawn at the manuscript's text width (430 pt), so font sizes are final.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                            "xtick.major.width": 0.6, "ytick.major.width": 0.6})
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results"
OUT = ROOT / "results/figures"
INK, MUTED = "#0b0b0b", "#6b6a66"

OB_NAMES = {"all": "All", 0: "No labs, low cost", 1: "Hypertension, many conditions", 2: "Labs, some conditions",
            3: "No labs, higher cost", 4: "Labs, few conditions"}
ICU_NAMES = {"all": "All", 0: "Neurological", 1: "Ventilated", 2: "Low risk, no blood gas", 3: "Moderate"}
OB_MODELS = [("score", "Health-mapped score", "#b2182b", "o"), ("cost_label", "Cost-label model", "#e08214", "s"),
             ("health_label", "Health-label model", "#2171b5", "D")]
ICU_MODELS = [("apache", "APACHE IVa", "#b2182b", "o"), ("lightgbm", "LightGBM", "#2171b5", "D")]


def _key(p):
    return -1 if p == "all" else int(p)


def panel(ax, df, names, models, title, xlabel, scale=1.0, sizes=None, labels=True):
    """Forest plot: one row per pocket (top: all), one offset interval per model."""
    pockets = sorted(df.pocket.unique(), key=_key)
    y = -np.arange(len(pockets), dtype=float)
    step = 0.62 / len(models)
    for k, (m, _, col, mk) in enumerate(models):
        d = df[df.model == m].set_index("pocket")
        for i, p in enumerate(pockets):
            if p not in d.index:
                continue
            r = d.loc[p]
            yi = y[i] + 0.31 - step * (k + 0.5)
            if r.status != "tested":
                if k == 0:
                    ax.text(0, y[i], "untested", ha="center", va="center", fontsize=6, color=MUTED,
                            bbox=dict(fc="white", ec="none", pad=0.5))
                continue
            ax.plot([scale * r.ci_low, scale * r.ci_high], [yi, yi], color=col, lw=1.1, solid_capstyle="butt")
            ax.plot(scale * r.estimate, yi, mk, ms=3.6 if mk == "o" else 3.2, mfc=col if r.flag else "white", mec=col,
                    mew=0.9, zorder=3)
    for yy in y[1:] + 0.5:
        ax.axhline(yy, color="#e4e3df", lw=0.5, zorder=0)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    lab = [names.get(p, str(p)) + (f" ({sizes[p]:.0%})" if sizes and p in sizes and p != "all" else "")
           for p in pockets]
    ax.set_yticks(y, lab if labels else [""] * len(y))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y[-1] - 0.5, 0.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=7.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(ax, models, title, anchor):
    h = [Line2D([0], [0], color=c, lw=1.1, marker=mk, ms=3.6 if mk == "o" else 3.2, mfc=c, mec=c)
         for _, _, c, mk in models]
    leg = ax.legend(h, [lab for _, lab, _, _ in models], fontsize=6.5, frameon=False, loc="upper left",
                    bbox_to_anchor=anchor, handlelength=1.6, title=title, title_fontsize=6.5, alignment="left")
    ax.add_artist(leg)


def main():
    ob = pd.read_csv(RUN / "obermeyer/audit_results.csv", keep_default_na=False, na_values=[""])
    ob["pocket"] = ob.pocket.map(lambda p: p if p == "all" else int(float(p)))
    sbm = ob[ob.partition == "sbm"]
    mu = sbm[(sbm.kind == "mu")]
    de = sbm[(sbm.kind == "delta") & (sbm.contrast == "black_vs_white")]
    icu = pd.read_csv(RUN / "wids_mu/audit_mu.csv", keep_default_na=False, na_values=[""])
    icu["pocket"] = icu.pocket.map(lambda p: p if p == "all" else int(float(p)))
    icu = icu[icu["map"] == "sbm_cut"]
    ob_size = {p: n / mu[mu.pocket == "all"].n_focal.iloc[0] for p, n in zip(mu.pocket, mu.n_focal)}
    icu_size = {p: n / icu[icu.pocket == "all"].n_focal.iloc[0] for p, n in zip(icu.pocket, icu.n_focal)}

    fig = plt.figure(figsize=(5.95, 3.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], hspace=0.62, wspace=0.08,
                          left=0.335, right=0.985, top=0.95, bottom=0.11)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    panel(a, mu, OB_NAMES, OB_MODELS, "(a) Care management: where",
          "$\\mu_c$ (conditions), predicted $-$ observed", sizes=ob_size)
    panel(b, de, OB_NAMES, OB_MODELS, "(b) Care management: for whom",
          "$\\delta_c$ (conditions), Black $-$ White", labels=False)
    panel(c, icu, ICU_NAMES, ICU_MODELS, "(c) ICU mortality: where (exploratory)",
          "$\\mu_c$, predicted $-$ observed mortality (points)", scale=100, sizes=icu_size)
    key = fig.add_subplot(gs[1, 1])
    key.axis("off")
    legend(key, OB_MODELS, "Care management (a, b)", (0.06, 1.1))
    legend(key, ICU_MODELS, "ICU (c)", (0.06, 0.45))
    key.text(0.62, 0.36, "filled: flagged\nopen: interval\nincludes zero", fontsize=6, color=MUTED, va="top", ha="left",
             transform=key.transAxes)
    fig.savefig(OUT / "fig_apps.pdf")
    fig.savefig(OUT / "fig_apps.png", dpi=200)
    print(f"wrote {OUT / 'fig_apps.pdf'}")


if __name__ == "__main__":
    main()
