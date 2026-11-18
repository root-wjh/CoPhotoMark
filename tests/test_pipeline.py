from pathlib import Path
import numpy as np
import pytest
from cophotomark.detector import FixedBoxes
from cophotomark.io import write_image, write_json, sha256_file, read_json
from cophotomark.protocol import Config
from cophotomark.experiments import enroll_manifest, build_trials, evaluate, summarize
from cophotomark.statistics import calibrate
from cophotomark.dataset import split_sources
from cophotomark.attacks import face_attack, transform

def test_complete_validation_then_test_and_leak_guard(tmp_path):
    config=Config(max_candidates=1,blind_offsets=(0,))
    detector=FixedBoxes([[0,0,96,96]])
    key=tmp_path/"key.json"
    write_json(key,{"master_key_hex":bytes(range(32)).hex()})
    sources=[]
    for index in range(4):
        path=tmp_path/f"source{index}.png"
        write_image(path,np.random.default_rng(index).integers(65,190,(96,96,3),dtype=np.uint8))
        sources.append({"path":str(path),"source_id":str(index),"source_sha256":sha256_file(path),
                        "group_id":str(index),"dataset":"synthetic_control","split":"validation" if index<2 else "test"})
    records=enroll_manifest(sources,tmp_path/"enrolled",detector,config,[key],"test")
    assert all(r["status"]=="enrolled" for r in records)
    trials=build_trials(records,tmp_path/"attacks",plan=[{"name":"clean","label":"genuine"}],face_operations=False)
    protocol=read_json(tmp_path/"attacks/attack_protocol.json")
    assert protocol["protocol_version"]=="legacy_crop_blur_rect_cropout_v1"
    assert protocol["rotation_border"]=="BORDER_REFLECT"
    validation=[r for r in trials if r["split"]=="validation"]
    scores=evaluate(validation,tmp_path/"val",detector,config,modes=("blind",),warmup=0)
    calibrated=calibrate(scores,config,target_far=0.01)
    tests=[r for r in trials if r["split"]=="test"]
    tested=evaluate(tests,tmp_path/"test",detector,config,modes=("blind",),calibration=calibrated,warmup=0)
    assert all(r["status"]=="ok" for r in tested)
    assert all(r["decision"]=="ACCEPTED" for r in tested if r["label"]=="genuine")
    assert all(r["decision"]!="ACCEPTED" for r in tested if r["label"]=="negative")
    assert summarize(tested,records)["counts"]["enrolled_face_observations"]==4
    with pytest.raises(ValueError,match="leakage"):
        evaluate([{**tests[0],"source_id":validation[0]["source_id"]}],tmp_path/"bad",detector,config,modes=("blind",),calibration=calibrated)
    with pytest.raises(ValueError,match="Detector/voting"):
        evaluate(tests,tmp_path/"bad",detector,config,calibration=calibrated,use_voting=False)

def test_default_trial_reads_metadata_from_inside_the_image(tmp_path):
    config=Config(max_candidates=1,blind_offsets=(0,))
    detector=FixedBoxes([[0,0,96,96]])
    key=tmp_path/"key.json"
    write_json(key,{"master_key_hex":bytes(range(32)).hex()})
    sources=[]
    for index in range(2):
        path=tmp_path/f"src{index}.png"
        write_image(path,np.random.default_rng(100+index).integers(65,190,(96,96,3),dtype=np.uint8))
        sources.append({"path":str(path),"source_id":str(index),"source_sha256":sha256_file(path),
                        "group_id":str(index),"dataset":"synthetic_control","split":"validation"})
    records=enroll_manifest(sources,tmp_path/"enrolled",detector,config,[key],"test")
    assert all(r["status"]=="enrolled" for r in records)
    assert all(r["metadata_channel"]=="image_embedded_ylsb" for r in records)
    trials=build_trials(records,tmp_path/"attacks",plan=[{"name":"clean","label":"genuine"}],face_operations=False)
    # Clean trials carry NO external sidecar reference: the image is the container.
    assert all(t["metadata_path"] is None for t in trials if t["attack"]=="clean")
    scores=evaluate(trials,tmp_path/"scores",detector,config,modes=("auto",),warmup=0)
    clean=[r for r in scores if r["attack"]=="clean"]
    assert all(r["status"]=="ok" and r["decision"]=="ACCEPTED" for r in clean)
    assert all(r["metadata_channel"]=="image_lsb" and r["actual_mode"]=="metadata" for r in clean)

def test_pixel_duplicate_groups_do_not_cross_splits(tmp_path):
    image=np.random.default_rng(4).integers(10,240,(32,32,3),dtype=np.uint8)
    rows=[]
    for index,ext in enumerate((".png",".bmp")):
        path=tmp_path/(str(index)+ext)
        write_image(path,image)
        rows.append({"path":str(path),"source_id":str(index),"source_sha256":sha256_file(path),"group_id":str(index),
                     "dataset":"test","status":"ok","detected_faces":1})
    result=split_sources(rows)
    assert result[0]["split"]==result[1]["split"]
    assert result[0]["group_id"]==result[1]["group_id"]

def test_collusion_requires_same_host_and_transforms_are_explicit():
    image=np.full((100,100,3),128,dtype=np.uint8)
    rng=np.random.default_rng(42)
    with pytest.raises(ValueError):
        face_attack(image,[[0,0,80,80]],{"name":"collusion"},rng,peers=[image])
    other=image.copy(); other[:]=130
    assert face_attack(image,[[0,0,80,80]],{"name":"collusion"},rng,peers=[image,other]).mean()==129
    assert transform(image,{"name":"resize","scale":0.5},rng).shape==image.shape
    assert transform(image,{"name":"metadata_strip"},rng).shape==image.shape
