from types import SimpleNamespace
import sys
import numpy as np
import pytest
from cophotomark.attacks import face_attack, exact_reorder_boxes
from cophotomark.lsbmeta import embed_metadata_lsb, extract_metadata_lsb
from cophotomark.watermark import qim_reconstruct
from cophotomark.protocol import Config
from cophotomark.detector import FixedBoxes, InsightFaceDetector
from cophotomark.engine import enroll, verify
from cophotomark.experiments import enroll_manifest, build_trials, evaluate
from cophotomark.io import write_image, write_json, sha256_file, read_image

def test_qim_is_paper_codebook_and_old_config_is_refused():
    values = np.array([-41., -9., 4., 16., 31., 54.])
    for bit in (0, 1):
        reconstructed = qim_reconstruct(values, bit, 18)
        assert np.allclose(np.mod(reconstructed, 36), bit * 18)
        assert np.all(np.abs(reconstructed - values) <= 18)
    with pytest.raises(ValueError):
        Config(version=1)

def test_lsb_does_not_modify_pixels_outside_frame():
    host = np.random.default_rng(7).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    metadata = {"test": "embedded"}
    marked = embed_metadata_lsb(host, metadata)
    assert extract_metadata_lsb(marked) == metadata
    from cophotomark.io import canonical
    bits = 64 + len(canonical(metadata)) * 8
    assert np.array_equal(host.reshape(-1, 3)[bits:], marked.reshape(-1, 3)[bits:])

def test_unequal_reorder_moves_whole_patch_without_crop():
    host = np.random.default_rng(4).integers(60, 190, (180, 350, 3), dtype=np.uint8)
    boxes = [[10, 20, 82, 116], [190, 20, 298, 140]]
    moved_boxes = exact_reorder_boxes(boxes, host.shape)
    marked = face_attack(host, boxes, {"name": "reorder_exact"}, np.random.default_rng(2))
    for dest, source in zip(moved_boxes, boxes[-1:] + boxes[:-1]):
        x1,y1,x2,y2 = dest
        a,b,c,d = source
        assert np.array_equal(marked[y1:y2,x1:x2], host[b:d,a:c])
    with pytest.raises(ValueError, match="preserve whole ROIs"):
        exact_reorder_boxes([[0,0,70,70],[80,0,180,90]], (90,180,3))

def test_scenario_generation_never_reads_sidecar_and_logs_skips(tmp_path):
    boxes = [[12,20,108,116], [180,20,276,116]]
    detector = FixedBoxes(boxes)
    config = Config(max_candidates=1, blind_offsets=(0,))
    key = tmp_path/'key.json'
    write_json(key, {"master_key_hex": bytes(range(32)).hex()})
    sources=[]
    for index in range(2):
        path=tmp_path/f'src{index}.png'
        write_image(path,np.random.default_rng(index).integers(65,190,(150,300,3),dtype=np.uint8))
        sources.append({"path":str(path),"source_id":str(index),"source_sha256":sha256_file(path),
                        "group_id":str(index),"dataset":"control","split":"development"})
    records=enroll_manifest(sources,tmp_path/'enrolled',detector,config,[key],'test')
    assert all(r['status']=='enrolled' and r['metadata_path'] is None for r in records)
    assert not list((tmp_path/'enrolled').glob('*.cpm.json'))
    trials=build_trials(records,tmp_path/'attacks',plan=[{"name":"clean","label":"genuine"}],face_operations=True)
    required={'delete','duplicate','reorder','reorder_exact','insert_marked','insert_unmarked','replace','copy_marked_face','detector_miss_control'}
    assert required.issubset({r['attack'] for r in trials})
    assert len({r['trial_id'] for r in trials})==len(trials)
    assert (tmp_path/'attacks/scenario_coverage.csv').is_file()
    assert all(r['metadata_path'] is None for r in trials)
    clean=next(r for r in trials if r['attack']=='clean')
    scores=evaluate([clean],tmp_path/'scores',detector,config,modes=('auto',),warmup=0)
    assert scores[0]['metadata_channel']=='image_lsb' and scores[0]['decision']=='ACCEPTED'
    for name in ('duplicate','reorder_exact'):
        trial=next(r for r in trials if r['attack']==name)
        answer=verify(read_image(trial['image_path']),detector,config,bytes(range(32)),
                      extract_metadata_lsb(read_image(clean['image_path']))['claim'],mode='blind')
        assert answer['decision']==('ACCEPTED' if name=='reorder_exact' else 'REJECTED')

def test_cuda_session_silent_fallback_is_rejected(tmp_path, monkeypatch):
    model=tmp_path/'models/buffalo_l'
    model.mkdir(parents=True)
    (model/'det_test.onnx').write_bytes(b'test model stub')
    monkeypatch.setitem(sys.modules,'onnxruntime',SimpleNamespace(get_available_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider']))
    class FakeAnalysis:
        def __init__(self,**kwargs):
            self.models={'detection':SimpleNamespace(session=SimpleNamespace(get_providers=lambda:['CPUExecutionProvider']))}
        def prepare(self,**kwargs): pass
    monkeypatch.setitem(sys.modules,'insightface.app',SimpleNamespace(FaceAnalysis=FakeAnalysis))
    with pytest.raises(RuntimeError,match='refusing silent CPU fallback'):
        InsightFaceDetector(root=tmp_path,provider='cuda')
