import numpy as np
import pandas as pd
import pytest
from pocket_audit.evaluation import person_metrics, q_curve
from pocket_audit.pipeline import PipelineConfig, select_level


def coords():
    return pd.DataFrame(dict(pocket=[0,1,2], supported=[True,True,False],
                             status=['tested','tested','not_eligible'], estimate=[.2,-.2,np.nan],
                             ci_low=[.1,-.3,np.nan], ci_high=[.3,-.1,np.nan]))


def test_people_weight_sizes_wrong_sign_and_unknown():
    pocket=np.array([0]*100+[1]+[2])
    bias=np.full(102,.2)
    result=person_metrics(pocket,np.ones(102,bool),bias,coords())
    assert result['person_tp']==100
    assert result['person_wrong_sign']==1
    assert result['person_fn']==2
    assert result['person_recall']==100/102
    assert result['person_precision']==100/101
    assert result['person_unknown_share']==1/102
    assert result['person_rmse']**2 == pytest.approx(result['person_between_mse']+result['person_within_mse'])


def test_no_flags_and_no_positive_conventions():
    c=coords(); c.loc[:,'ci_low']=-.3; c.loc[:,'ci_high']=.3
    r=person_metrics([0,0],[True,True],[.1,.1],c)
    assert r['person_f1']==0 and np.isnan(r['person_precision'])
    r=person_metrics([0,0],[True,True],[0.,0.],c)
    assert np.isnan(r['person_f1']) and np.isnan(r['person_recall'])


def test_q_both_directions_unknown_and_duplicates():
    q=q_curve([0,0,1,2],[True]*4,coords(),[0,.2,.4])
    assert np.all(q.q_lower <= q.q_estimate)
    assert np.all(q.q_estimate <= q.q_upper)
    assert np.all(q.unknown_share==.25)
    assert q.query('direction=="positive"').q_estimate.iloc[0]==.5
    assert q.query('direction=="negative"').q_estimate.iloc[0]==.25
    with pytest.raises(ValueError):
        q_curve([0],[True],pd.concat([coords(),coords()]),[0])


def test_actual_selector_coarse_ties_disabled_floor_and_fallback():
    levels=pd.DataFrame(dict(level=[0,1,2],n_supported=[2,2,1],n_pockets_profiled=[10,4,2],
                             coverage_focal=[np.nan,.1,.1],coverage_reference=[np.nan,.1,.1],
                             graph_profiled_share=[.9,.9,.9]))
    assert select_level(levels,PipelineConfig())==(1,'qualified')
    assert select_level(levels.iloc[:1],PipelineConfig())[0]==0
    assert select_level(levels,PipelineConfig(min_group_coverage=.8))==(2,'no_level_met_support_rule')
    levels.loc[1,'graph_profiled_share']=.1
    assert select_level(levels,PipelineConfig(min_graph_profiled_share=.8))[0]==0


def test_design_eligible_degenerate_remains_visible():
    from pocket_audit.audit import AuditConfig,SupportRule,run_audit
    from pocket_audit.data import Contrast
    n=200
    table,_=run_audit(np.zeros(n,int),{'m':np.full(n,.1)},np.zeros(n,int),np.arange(n),
        pd.DataFrame({'S':['1']*100+['0']*100}),[Contrast('S','S','1','0')],
        AuditConfig(support=SupportRule(min_n=10,min_units=1,min_effective_units=1,min_expected=0),n_boot=99))
    d=table.query('kind=="delta"')
    assert d.design_eligible.all() and d.supported.all()
    assert d.status.eq('degenerate').all() and not d.flag.any()


def test_partial_null_and_concentration_reproducibility():
    from pocket_audit.simulation import SimConfig,simulate,GRADED
    cfg=SimConfig(n=2000,n_units=50,s_unit_sd=2,**GRADED)
    a,info=simulate(cfg,17);b,other=simulate(cfg,17)
    assert np.array_equal(a.protected.S,b.protected.S)
    assert np.all(info['bias']==0) and not info['affected'].any()


def test_threshold_censoring_and_interpolation():
    from pocket_audit.sensitivity import power_threshold
    assert power_threshold([0,.1,.2],[.1,.6,1])[0]==pytest.approx(.15)
    assert power_threshold([0,.1],[0,.5])[1]


def test_duplicate_truth_join_is_rejected():
    from pocket_audit.simulation import replicate_metrics
    row=dict(partition='sbm',kind='delta',model='model',contrast='S1_vs_S0',pocket=0)
    with pytest.raises(pd.errors.MergeError):
        replicate_metrics(None,None,None,{'results':pd.DataFrame([row])},pd.DataFrame([row,row]),None)
