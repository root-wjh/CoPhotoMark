"""Clean drift/real-detector diagnostics that earlier fixed-frame tests missed."""
from dataclasses import replace
import numpy as np
import pytest
from cophotomark.detector import FixedBoxes
from cophotomark.engine import enroll, verify
from cophotomark.protocol import Config

KEY = bytes(range(32))
BOXES = [[24, 24, 120, 120], [192, 24, 288, 120]]

def phase_config():
    # One representative for each block phase; 144 geometry candidates plus
    # seven other block/stride candidates. No threshold change.
    return Config(blind_offsets=(0,-1,1,-2,2,-3,3,-4,4,-5,5,6), max_candidates=160)

@pytest.mark.parametrize("drift", [3, -3, 5, -5])
def test_full_phase_recovers_clean_three_to_five_pixel_drift(drift):
    config = phase_config()
    host = np.random.default_rng(42).integers(65,190,(150,320,3),dtype=np.uint8)
    result = enroll(host,FixedBoxes(BOXES),config,KEY)
    boxes = [[b[0]+drift,b[1]-drift,b[2],b[3]] for b in BOXES]
    answer = verify(result["image"],FixedBoxes(boxes),config,KEY,result["claim"],mode="blind")
    assert answer["decision"] == "ACCEPTED", answer
    assert answer["raw_score"] == 1.0
    assert answer["candidate_count"] <= len(BOXES)*config.max_candidates

def test_wider_phase_does_not_restore_missing_face_or_accept_wrong_key():
    config = phase_config()
    host = np.random.default_rng(42).integers(65,190,(150,320,3),dtype=np.uint8)
    result = enroll(host,FixedBoxes(BOXES),config,KEY)
    missing = verify(result["image"],FixedBoxes(BOXES[:1]),config,KEY,result["claim"],mode="blind")
    assert missing["decision"] == "REJECTED" and missing["reason"] == "missing_or_extra_face"
    wrong = verify(result["image"],FixedBoxes(BOXES),config,bytes(reversed(KEY)),result["claim"],mode="blind")
    assert wrong["decision"] != "ACCEPTED"

def test_post_embedding_lost_detection_is_logged_without_filtering():
    class DropsFace:
        def __init__(self): self.calls = 0
        def detect(self,image):
            self.calls += 1
            return BOXES if self.calls == 1 else BOXES[:1]
    host = np.random.default_rng(42).integers(65,190,(150,320,3),dtype=np.uint8)
    result = enroll(host,DropsFace(),Config(),KEY)
    assert result["enrolled_faces"] == 2  # never change count to hide failure
    check = result["clean_detection_consistency"]
    assert check["final_detected_faces"] == 1
    assert check["unmatched_original_faces"] == 1
    assert not check["matched_detection_set"]
