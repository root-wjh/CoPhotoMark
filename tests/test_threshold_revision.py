import pytest
from cophotomark.threshold_revision import rates, curve, working_threshold
from cophotomark.statistics import calibrate
from cophotomark.protocol import Config

def row(score,label,source,**extra):
    return dict(score=score,label=label,source_id=source,mode='auto',split='validation',
                structural_ok=True,decision='REJECTED',**extra)

def test_frozen_point_is_selected_without_test_labels():
    val=[row(.9,'genuine','v1'),row(.8,'genuine','v2'),row(.7,'negative','v3')]
    threshold=working_threshold(curve(val,48),'0.01')
    test=[row(.6,'genuine','t1'),row(.95,'negative','t2')]
    result=rates(test,threshold)
    assert .7<threshold<=.8
    assert result['FAR']==1 and result['FRR']==1

def test_structural_and_undecidable_failures_remain_in_denominator():
    a=row(1,'genuine','a'); a['structural_ok']=False
    b=row(1,'genuine','b'); b['decision']='UNDECIDABLE'
    c=row(.5,'negative','c')
    assert rates([a,b,c],.8)['FRR']==1
    assert rates([a,b,c],.8)['undecidable_genuine']==1
    assert rates([a,b,c],0)['true_accepts']==0

def test_primary_calibration_rejects_test_leakage():
    a=row(1,'genuine','a'); a['split']='test'
    with pytest.raises(ValueError,match='validation'):
        calibrate([a,row(.5,'negative','b')],Config(),.01)

def test_infeasible_target_is_not_silently_relaxed():
    rows=[row(1,'genuine','a'),row(1,'negative','b')]
    assert working_threshold(curve(rows,48),'0.01') is None

def test_repeated_keys_do_not_increase_independent_source_count():
    neg=[row(.2,'negative','same') for _ in range(100)]
    result=rates([row(1,'genuine','p')]+neg,.8)
    assert result['negative_trials']==100 and result['negative_sources']==1
    assert result['zero_FA_upper95_by_independent_source']==pytest.approx(.95)
