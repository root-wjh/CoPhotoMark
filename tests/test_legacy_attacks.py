import random
import cv2
import numpy as np
import pytest
from cophotomark.attacks import transform, default_plan

@pytest.fixture
def host():
    return np.random.default_rng(74).integers(0,256,(113,157,3),dtype=np.uint8)

def test_legacy_crop_matches_original_python_random(host):
    seed=20260626
    rng=random.Random(seed)
    h,w=host.shape[:2]
    ch,cw=int(h*.965),int(w*.965)
    x=rng.randint(0,w-cw); y=rng.randint(0,h-ch)
    expected=cv2.resize(host[y:y+ch,x:x+cw],(w,h))
    assert np.array_equal(transform(host,dict(name='crop',crop_ratio=.965),random.Random(seed)),expected)

def test_obsolete_crop_spec_is_not_silently_reinterpreted(host):
    with pytest.raises(ValueError,match='Obsolete'):
        transform(host,dict(name='crop',removed_area=.035),np.random.default_rng(3))

def test_legacy_blur_resize_rotation_jpeg(host):
    h,w=host.shape[:2]; rng=np.random.default_rng(1)
    assert np.array_equal(transform(host,dict(name='blur',sigma=2),rng),cv2.GaussianBlur(host,(5,5),sigmaX=2))
    expected=cv2.resize(cv2.resize(host,(int(w*.9),int(h*.9))),(w,h))
    assert np.array_equal(transform(host,dict(name='resize',scale=.9),rng),expected)
    matrix=cv2.getRotationMatrix2D((w//2,h//2),5,1)
    expected=cv2.warpAffine(host,matrix,(w,h),borderMode=cv2.BORDER_REFLECT)
    assert np.array_equal(transform(host,dict(name='rotate',degrees=5),rng),expected)
    ok,encoded=cv2.imencode('.jpg',host,[cv2.IMWRITE_JPEG_QUALITY,50]); assert ok
    assert np.array_equal(transform(host,dict(name='jpeg',quality=50),rng),cv2.imdecode(encoded,1))

def test_cropout_is_single_rectangle_preserves_rest_and_input():
    host=np.full((113,157,3),211,dtype=np.uint8); original=host.copy()
    spec=dict(name='cropout',area_fraction=.3,fill=127)
    out=transform(host,spec,np.random.default_rng(42))
    mask=np.all(out==127,axis=2); ys,xs=np.where(mask)
    ch,cw=round(113*np.sqrt(.3)),round(157*np.sqrt(.3))
    assert out.shape==host.shape and np.array_equal(host,original)
    assert mask.sum()==ch*cw
    assert ys.max()-ys.min()+1==ch and xs.max()-xs.min()+1==cw
    assert np.all(out[~mask]==211)
    assert np.array_equal(out,transform(host,spec,np.random.default_rng(42)))
    for fraction in (0,1):
        out=transform(host,{**spec,'area_fraction':fraction},np.random.default_rng(42))
        assert np.all(out==(211 if fraction==0 else 127))
    with pytest.raises(ValueError):
        transform(host,{**spec,'area_fraction':1.1},np.random.default_rng(42))

def test_default_plan_explicit_legacy_crop_and_blur():
    plan=default_plan()
    assert next(p for p in plan if p['name']=='crop')['crop_ratio']==.965
    assert all(p['kernel']==5 for p in plan if p['name']=='blur')
