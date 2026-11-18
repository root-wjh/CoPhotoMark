import copy
from dataclasses import replace
import numpy as np
import pytest
from cophotomark.detector import FixedBoxes
from cophotomark.engine import enroll, verify, set_structure, validate_metadata
from cophotomark.io import canonical, read_json, writable
from cophotomark.protocol import Config, new_claim, payload, subkey, position_list
from cophotomark.watermark import decode_roi
from cophotomark.statistics import calibrate, binomial_reference, zero_error_upper

KEY = bytes(range(32))
OTHER = bytes(reversed(range(32)))
BOXES = [[16, 16, 112, 112], [144, 16, 240, 112]]

@pytest.fixture
def host():
    return np.random.default_rng(42).integers(65, 190, (144, 256, 3), dtype=np.uint8)

@pytest.fixture
def example(host):
    config = Config()
    result = enroll(host, FixedBoxes(BOXES), config, KEY)
    return config, result

def test_full_clean_roundtrip(example):
    config, result = example
    for mode in ("metadata", "blind", "auto"):
        answer = verify(result["image"], FixedBoxes(BOXES), config, KEY, result["claim"], result["metadata"], mode)
        assert answer["decision"] == "ACCEPTED", answer
        assert sorted(f["index"] for f in answer["faces"]) == [1, 2]
        assert answer["candidate_count"] <= config.max_candidates*2
    assert result["quality"]["embedded_bit_instances"] > result["quality"]["protocol_bits"]

def test_missing_claim_is_undecidable(example):
    config, result = example
    answer = verify(result["image"], FixedBoxes(BOXES), config, KEY, None, result["metadata"])
    assert answer["decision"] == "UNDECIDABLE"

def test_deleted_missed_face_rejected(example):
    config, result = example
    answer = verify(result["image"], FixedBoxes(BOXES[:1]), config, KEY, result["claim"], mode="blind")
    assert answer["decision"] == "REJECTED"
    assert answer["reason"] == "missing_or_extra_face"

def test_blind_recovers_one_pixel_detection_drift(example):
    config, result = example
    drifted = [[b[0]+1,b[1]-1,b[2],b[3]] for b in BOXES]
    assert verify(result["image"],FixedBoxes(drifted),config,KEY,result["claim"],mode="blind")["decision"] == "ACCEPTED"

def test_copy_duplicate_index_rejected(example):
    config, result = example
    image = result["image"].copy()
    image[16:112, 144:240] = image[16:112, 16:112]
    answer = verify(image, FixedBoxes(BOXES), config, KEY, result["claim"], mode="blind")
    assert answer["decision"] == "REJECTED"
    assert answer["reason"] == "duplicated_or_missing_index"

def test_order_invariant_rearrangement(example):
    config, result = example
    image = result["image"].copy()
    image[16:112, 144:240], image[16:112, 16:112] = result["image"][16:112,16:112], result["image"][16:112,144:240]
    answer = verify(image, FixedBoxes(BOXES[::-1]), config, KEY, result["claim"], mode="blind")
    assert answer["decision"] == "ACCEPTED"

def test_face_attack_reorder_exact_is_interpolation_free(example):
    from cophotomark.attacks import face_attack
    import numpy as np
    config, result = example
    rng = np.random.default_rng(7)
    moved = face_attack(result["image"], BOXES, {"name": "reorder_exact"}, rng)
    # Cyclic permutation of two equal-size boxes must move every pixel exactly once.
    a = result["image"]
    for dest, source in zip(BOXES, [BOXES[-1], BOXES[0]]):
        assert np.array_equal(moved[dest[1]:dest[3], dest[0]:dest[2]],
                              a[source[1]:source[3], source[0]:source[2]])
    answer = verify(moved, FixedBoxes(BOXES), config, KEY, result["claim"], mode="blind")
    assert answer["decision"] == "ACCEPTED"

def test_additional_unmarked_face_rejected(example):
    config, result = example
    image = np.random.default_rng(3).integers(65,190,(144,384,3),dtype=np.uint8)
    image[:, :256] = result["image"]
    answer = verify(image, FixedBoxes(BOXES+[[272,16,368,112]]), config, KEY, result["claim"], mode="blind")
    assert answer["decision"] != "ACCEPTED"

def test_wrong_key_rejected(example):
    config, result = example
    assert verify(result["image"], FixedBoxes(BOXES), config, OTHER, result["claim"], mode="blind")["decision"] != "ACCEPTED"

def test_new_session_replay_rejected(example):
    config, result = example
    claim = new_claim("different", config)
    assert verify(result["image"], FixedBoxes(BOXES), config, KEY, claim, result["metadata"])["decision"] != "ACCEPTED"

def test_invalid_metadata_falls_back_and_does_not_authenticate_pixels(example):
    config, result = example
    metadata = copy.deepcopy(result["metadata"])
    metadata["faces"][0]["index"] = 99
    answer = verify(result["image"], FixedBoxes(BOXES), config, KEY, result["claim"], metadata)
    assert answer["mode"] == "blind"
    assert answer["metadata_status"] == "invalid_hmac"
    image = np.full_like(result["image"], 127)
    answer = verify(image, FixedBoxes(BOXES), config, KEY, result["claim"], result["metadata"])
    assert answer["decision"] != "ACCEPTED"

def test_valid_metadata_does_not_hide_new_face(example):
    config, result = example
    boxes = BOXES+[[16,114,112,143]]
    answer = verify(result["image"], FixedBoxes(boxes), config, KEY, result["claim"], result["metadata"])
    assert answer["decision"] == "UNDECIDABLE"

def test_blind_does_not_read_sidecar(example):
    config, result = example
    answer = verify(result["image"], FixedBoxes(BOXES), config, KEY, result["claim"], {"malformed": object()}, mode="blind")
    assert answer["decision"] == "ACCEPTED"

def test_metadata_is_carried_inside_the_image_by_default(example):
    from cophotomark.lsbmeta import extract_metadata_lsb
    config, result = example
    extracted = extract_metadata_lsb(result["image"])
    assert extracted is not None
    assert extracted["hmac"] == result["metadata"]["hmac"]
    # No sidecar argument: the verifier must authenticate (M, sigma_meta) from
    # the channel embedded inside the image itself.
    answer = verify(result["image"], FixedBoxes(BOXES), config, KEY, result["claim"])
    assert answer["decision"] == "ACCEPTED", answer
    assert answer["metadata_channel"] == "image_lsb"
    assert answer["mode"] == "metadata"

def test_embedded_metadata_survives_png_roundtrip(example):
    import cv2
    from cophotomark.lsbmeta import extract_metadata_lsb
    config, result = example
    ok, encoded = cv2.imencode(".png", result["image"])
    assert ok
    reread = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert extract_metadata_lsb(reread)["hmac"] == result["metadata"]["hmac"]
    answer = verify(reread, FixedBoxes(BOXES), config, KEY, result["claim"])
    assert answer["decision"] == "ACCEPTED" and answer["metadata_channel"] == "image_lsb"

def test_corrupted_embedded_metadata_falls_back_to_blind(example):
    import cv2
    from cophotomark.lsbmeta import HEADER_BITS
    config, result = example
    ycc = cv2.cvtColor(result["image"], cv2.COLOR_BGR2YCrCb)
    flat = ycc[:, :, 0].reshape(-1)
    # Flip ~200 payload-bit LSBs (past the 64-bit header); any bit error must
    # invalidate the HMAC-authenticated channel, never be partially trusted.
    positions = HEADER_BITS + np.linspace(0, 2400, 200).astype(int)
    flat[positions] ^= 1
    damaged = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    answer = verify(damaged, FixedBoxes(BOXES), config, KEY, result["claim"])
    assert answer["mode"] == "blind"
    assert answer["metadata_channel"] == "none"
    # Either the JSON no longer parses (absent) or it parses but its HMAC
    # fails; both must discard the channel entirely.
    assert answer["metadata_status"] in {"absent", "invalid_hmac", "schema", "claim_config_or_version_mismatch"}
    assert answer["decision"] == "ACCEPTED"  # watermark evidence itself is intact

def test_jpeg_destroys_embedded_metadata_channel(example):
    from cophotomark.attacks import jpeg
    config, result = example
    attacked = jpeg(result["image"], 85)
    answer = verify(attacked, FixedBoxes(BOXES), config, KEY, result["claim"])
    assert answer["mode"] == "blind" and answer["metadata_channel"] == "none"

def test_metadata_strip_erases_in_image_channel_but_keeps_watermark(example):
    from cophotomark.attacks import transform
    from cophotomark.lsbmeta import extract_metadata_lsb
    config, result = example
    stripped = transform(result["image"], {"name": "metadata_strip"}, np.random.default_rng(0))
    assert extract_metadata_lsb(stripped) is None
    auto = verify(stripped, FixedBoxes(BOXES), config, KEY, result["claim"])
    assert auto["mode"] == "blind" and auto["metadata_channel"] == "none"
    assert auto["decision"] == "ACCEPTED"
    # Forcing the metadata pipeline after stripping is undecidable, never a
    # silent acceptance on missing metadata.
    forced = verify(stripped, FixedBoxes(BOXES), config, KEY, result["claim"], mode="metadata")
    assert forced["decision"] == "UNDECIDABLE"

def test_embedded_metadata_signed_by_other_key_is_ignored(example):
    config, result = example
    answer = verify(result["image"], FixedBoxes(BOXES), config, OTHER, result["claim"])
    assert answer["metadata_channel"] == "none" and answer["mode"] == "blind"
    assert answer["decision"] != "ACCEPTED"

def test_payload_layout_and_key_separation():
    config = Config()
    claim = new_claim("test", config)
    bits = payload(3, 2, config, KEY, claim)
    assert len(bits) == 64
    assert np.packbits(bits[:16]).tolist() == [3,2]
    assert config.tag_bits == 48
    assert subkey(KEY,"watermark") != subkey(KEY,"metadata")
    assert position_list(config,12,KEY,claim) == position_list(config,12,KEY,claim)
    assert position_list(config,12,KEY,claim) != position_list(config,12,OTHER,claim)
    with pytest.raises(ValueError):
        payload(256,1,config,KEY,claim)

@pytest.mark.parametrize("kwargs", [{"stride":6},{"block":8},{"step":float('nan')},{"max_faces":True},{"payload_bits":256}])
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

def test_128_bit_roundtrip(host):
    config = Config(payload_bits=128,block=16,stride=16)
    result = enroll(host,FixedBoxes(BOXES),config,KEY)
    assert verify(result["image"],FixedBoxes(BOXES),config,KEY,result["claim"],mode="blind")["decision"] == "ACCEPTED"

def test_ablations_show_failure_modes(host):
    for variant in ("no_count", "no_id", "roi_qim"):
        config = Config(variant=variant)
        result = enroll(host,FixedBoxes(BOXES),config,KEY)
        assert verify(result["image"],FixedBoxes(BOXES),config,KEY,result["claim"],mode="blind")["decision"] == "ACCEPTED"
        if variant == "no_count":
            answer = verify(result["image"],FixedBoxes(BOXES[:1]),config,KEY,result["claim"],mode="blind")
            assert answer["decision"] == "ACCEPTED"
        if variant == "no_id":
            image = result["image"].copy()
            image[16:112,144:240] = image[16:112,16:112]
            assert verify(image,FixedBoxes(BOXES),config,KEY,result["claim"],mode="blind")["decision"] == "ACCEPTED"

def test_calibration_requires_validation_and_both_classes():
    with pytest.raises(ValueError):
        calibrate([{"split":"test"}],Config())
    with pytest.raises(ValueError):
        calibrate([{"split":"validation","label":"genuine"}],Config())

def test_calibration_discrete_thresholds():
    base = {"split":"validation","mode":"blind","structural_ok":True,"decision":"REJECTED","candidate_count":8}
    rows = [{**base,"source_id":"a","label":"genuine","score":0.95},
            {**base,"source_id":"b","label":"negative","score":0.6}]
    selected = calibrate(rows,Config())["modes"]["blind"]["threshold"]
    assert selected > 0.6 and selected <= 0.95
    assert abs(zero_error_upper(3000) - 0.000998079) < 1e-8
    assert binomial_reference(48,0,8)["union_bound_reference"] == 8/2**48

def test_original_write_guard_and_duplicate_json(tmp_path):
    with pytest.raises(ValueError):
        writable("C:/Users/dell'/PycharmProjects/CoPhotoMark/do_not_write.txt")
    path = tmp_path/"duplicate.json"
    path.write_text('{"v":1,"v":2}')
    with pytest.raises(ValueError):
        read_json(path)
