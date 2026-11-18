import pytest
from cophotomark.face_count_availability import summarize

def row(source,n,decision='ACCEPTED',**kwargs):
    return dict(source_id=source,enrolled_faces=n,decision=decision,status='ok',
        split='test',label='genuine',mode='blind',attack='clean',trial_id=source,
        threshold=.8125,config_sha256='cfg',**kwargs)

def test_failures_remain_and_negative_validation_excluded():
    rows=[row('a',2),row('b',2,'UNDECIDABLE'),row('c',2,'REJECTED')]
    rows[2]['status']='error'
    rows += [{**row('d',2),'label':'negative'},{**row('e',2),'split':'validation'}]
    s=next(x for x in summarize(rows) if x['grouping']=='exact' and x['attack']=='clean')
    assert s['trials']==3 and s['source_images']==3
    assert s['acceptance_pct']==pytest.approx(100/3)
    assert s['errors']==1 and s['undecidable']==1

def test_repetitions_do_not_count_as_independent_images():
    rows=[row('a',4),{**row('a',4),'trial_id':'a2'},row('b',4,'REJECTED')]
    s=next(x for x in summarize(rows) if x['grouping']=='coarse' and x['attack']=='clean')
    assert s['source_images']==2 and s['trials']==3 and s['acceptance_pct']==50
    assert s['enrolled_faces']=='>=4'

def test_duplicates_and_threshold_retuning_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        summarize([row('a',2),row('a',2)])
    with pytest.raises(ValueError,match='Threshold'):
        summarize([row('a',2),{**row('b',3),'threshold':1}])

def test_n_is_enrollment_count_not_query_detection_count():
    s=summarize([row('a',6,'REJECTED',detected_faces=5)])
    assert all(x['enrolled_faces'] in ('6','>=4') for x in s)
    assert all(x['acceptance_pct']==0 and x['ci95_low_pct'] is None for x in s)
