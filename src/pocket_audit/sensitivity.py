"""Synthetic fixed-design detectability demonstration, distinct from empirical Q."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .audit import AuditConfig,run_audit
from .data import Contrast


def power_threshold(grid, power, target=.8):
    smooth=IsotonicRegression(increasing=True,y_min=0,y_max=1).fit_transform(grid,power)
    hits=np.flatnonzero(smooth>=target)
    if not len(hits):return np.nan,True
    j=hits[0]
    if j==0:return float(grid[0]),False
    return float(np.interp(target,smooth[j-1:j+1],grid[j-1:j+1])),False


def fixed_design_detectability(pocket,focal,unit,p_common,grid,cfg:AuditConfig,
                               n_rep=100,seed=2026100302,shared_frac=0.,n_mc_boot=200):
    """Equal residual-distribution reference within pockets, plus focal prediction shift.

    Uses common random outcomes across shifts. This is a proof of concept for this
    fixed design and bootstrap setting, not power for the adaptive discovery pipeline.
    """
    pocket,focal,unit,p_common=map(np.asarray,(pocket,focal,unit,p_common))
    focal=focal.astype(bool); grid=np.asarray(grid,float)
    if (grid<0).any() or (np.diff(grid)<=0).any():raise ValueError('use increasing nonnegative shifts')
    if (p_common+grid.max()>1).any():raise ValueError('shift infeasible; do not clip')
    for c in np.unique(pocket):
        if np.unique(p_common[pocket==c]).size!=1:raise ValueError('common prediction must be constant within each pocket')
    prot=pd.DataFrame({'S':np.where(focal,'1','0')})
    contrasts=[Contrast('S1_vs_S0','S','1','0')]
    rng=np.random.default_rng(seed)
    ids,inverse=np.unique(unit,return_inverse=True)
    pockets=np.unique(pocket[pocket>=0])
    detected=np.zeros((n_rep,len(grid),len(pockets)),bool)
    eligible=None
    for r in range(n_rep):
        shared=rng.random(len(ids))[inverse]
        u=np.where(rng.random(len(unit))<shared_frac,shared,rng.random(len(unit)))
        y=(u<p_common).astype(int)
        for j,shift in enumerate(grid):
            table,_=run_audit(y,{'model':p_common+shift*focal},pocket,unit,prot,contrasts,cfg)
            design=table.design_eligible.to_numpy()
            if eligible is None:eligible=design
            if not np.array_equal(design,eligible):raise ValueError('design eligibility changes with shift; freeze family explicitly')
            for k,c in enumerate(pockets):
                local=table[(table.kind=='delta') & (table.pocket==int(c))]
                detected[r,j,k]=bool(local.flag_positive.any())
    records=[];thresholds=[]
    boot_rng=np.random.default_rng(seed+1)
    resamples=boot_rng.integers(0,n_rep,(n_mc_boot,n_rep))
    for k,c in enumerate(pockets):
        power=detected[:,:,k].mean(0)
        threshold,censored=power_threshold(grid,power)
        draws=np.array([power_threshold(grid,detected[ix,:,k].mean(0))[0] for ix in resamples])
        observed=draws[np.isfinite(draws)]
        weight=(focal & (pocket==c)).sum()/focal.sum()
        thresholds.append(dict(pocket=int(c),weight=weight,threshold80=threshold,censored=censored,
                               mc_lower=np.quantile(observed,.025) if len(observed) else np.nan,
                               mc_upper=np.quantile(observed,.975) if len(observed) else np.nan,
                               mc_censored_share=1-len(observed)/n_mc_boot))
        for shift,power_ in zip(grid,power):records.append(dict(pocket=int(c),shift=shift,power=power_))
    thresholds=pd.DataFrame(thresholds)
    curve=pd.DataFrame([dict(threshold=t,a_estimate=thresholds.loc[thresholds.threshold80<=t,'weight'].sum(),
                            censored_share=thresholds.loc[thresholds.censored,'weight'].sum()) for t in grid])
    return pd.DataFrame(records),thresholds,curve
