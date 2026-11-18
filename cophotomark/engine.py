from __future__ import annotations
import hmac
import time
import numpy as np
from .detector import clip_boxes, iou
from .io import canonical
from .lsbmeta import embed_metadata_lsb, extract_metadata_lsb
from .protocol import authenticate_metadata, new_claim, payload, validate_claim
from .quality import measure
from .watermark import block_count, candidate_configs, decode_roi, embed_roi

class EnrollmentFailure(ValueError):
    def __init__(self, message, detected, eligible, excluded):
        super().__init__(message)
        self.statistics = {"detected_faces": detected, "eligible_faces": eligible, "excluded_faces": excluded,
                           "enrolled_faces": 0}

def crop(image, box):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]

def enroll(image, detector, config, master, message="CoPhotoMark research", claim=None):
    start = time.perf_counter()
    claim = claim or new_claim(message, config)
    validate_claim(claim, config)
    detect_start = time.perf_counter()
    detected = detector.detect(image)
    detection_ms = (time.perf_counter() - detect_start)*1000
    boxes, exclusions = [], []
    for box in detected:
        roi = crop(image, box)
        reason = None
        if min(roi.shape[:2]) < config.min_roi:
            reason = "short_edge_below_min_roi"
        elif block_count(roi, config) < config.min_blocks:
            reason = "insufficient_qualified_blocks"
        elif any(iou(box, previous) > 0 for previous in boxes):
            reason = "overlapping_enrollment_roi"
        if reason:
            exclusions.append({"bbox": box, "reason": reason})
        else:
            boxes.append(box)
    if not boxes or len(boxes) > config.max_faces:
        raise EnrollmentFailure(f"No enrollable face or face-count overflow; detected={len(detected)}, excluded={len(exclusions)}",
                                len(detected), len(boxes), len(exclusions))
    out = image.copy()
    counts, records, words = [], [], []
    embed_start = time.perf_counter()
    for index, box in enumerate(boxes, 1):
        bits = payload(len(boxes), index, config, master, claim)
        roi, count = embed_roi(crop(image, box), bits, config, master, claim)
        x1, y1, x2, y2 = box
        out[y1:y2, x1:x2] = roi
        counts.append(count)
        words.append(np.packbits(bits).tobytes().hex())
        records.append({"bbox": list(box), "index": index, "blocks": count})
    # Verify the final uint8/color-roundtrip image, not pre-save floating coefficients.
    for record in records:
        decoded = decode_roi(crop(out, record["bbox"]), config.block, config.stride, config, master, claim)
        if not decoded["valid"] or decoded.get("score", 0) < 1.0:
            raise EnrollmentFailure("Enrollment self-check failed after color/uint8 roundtrip; do not silently drop face",
                                    len(detected), len(boxes), len(exclusions))
    body = {"version": 1, "claim": claim, "config": config.to_dict(), "shape": list(image.shape), "faces": records,
            "excluded_faces": exclusions}
    metadata = {**body, "hmac": authenticate_metadata(body, master)}
    # Paper Algorithm 1: embed (M, sigma_meta) INTO the image (Y-plane LSB).
    # The PNG carries metadata; key + trusted expected claim remain external.
    # Re-check watermark evidence on the final shipped pixels.
    try:
        out = embed_metadata_lsb(out, metadata)
    except (ValueError, RuntimeError) as exc:
        raise EnrollmentFailure(f"In-image metadata enrollment failed: {exc}",
                                len(detected), len(boxes), len(exclusions)) from exc
    for record in records:
        decoded = decode_roi(crop(out, record["bbox"]), config.block, config.stride, config, master, claim)
        if not decoded["valid"] or decoded.get("score", 0) < 1.0:
            raise EnrollmentFailure("Enrollment self-check failed after embedding in-image metadata; do not silently drop face",
                                    len(detected), len(boxes), len(exclusions))
    quality = measure(image, out, boxes, counts, config, words)
    frame_bits = 64 + len(canonical(metadata)) * 8
    quality.update(metadata_frame_bits=frame_bits,
                   metadata_frame_bits_per_image_pixel=frame_bits / (image.shape[0] * image.shape[1]),
                   distortion_scope="ROI DCT watermark plus raster-prefix Y-LSB metadata; quality measured on final PNG pixels")
    # Distinct from fixed-box watermark self-check: inspect the actual detector
    # on delivered pixels. Keep unstable enrollments in the denominator.
    final_boxes = detector.detect(out)
    overlaps = sorted(((iou(a, b), i, j) for i, a in enumerate(detected)
                       for j, b in enumerate(final_boxes)), reverse=True)
    original_matched, final_matched = set(), set()
    for overlap, i, j in overlaps:
        if overlap >= 0.25 and i not in original_matched and j not in final_matched:
            original_matched.add(i)
            final_matched.add(j)
    consistency = {"original_detected_faces": len(detected), "final_detected_faces": len(final_boxes),
                   "unmatched_original_faces": len(detected)-len(original_matched),
                   "unmatched_final_faces": len(final_boxes)-len(final_matched),
                   "final_detected_boxes": final_boxes,
                   "matched_detection_set": len(original_matched) == len(detected) == len(final_boxes),
                   "note": "Post-embedding detection diagnostic only; not identity ground truth or an eligibility filter"}
    return {"image": out, "claim": claim, "metadata": metadata, "quality": quality,
            "detected_faces": len(detected), "enrolled_faces": len(boxes), "excluded_faces": len(exclusions),
            "coverage": "all_detected_faces" if not exclusions else "partial_detected_faces",
            "clean_detection_consistency": consistency,
            "detection_ms": detection_ms, "embedding_ms": (time.perf_counter()-embed_start)*1000,
            "total_ms": (time.perf_counter()-start)*1000}

def validate_metadata(metadata, config, master, claim):
    if metadata is None:
        return None, "absent"
    try:
        if not isinstance(metadata, dict) or len(canonical(metadata)) > 1024*1024:
            raise ValueError("metadata type/size")
        body = {k: v for k, v in metadata.items() if k != "hmac"}
        if not isinstance(metadata.get("hmac"), str) or not hmac.compare_digest(authenticate_metadata(body, master), metadata["hmac"]):
            raise ValueError("invalid_hmac")
        if set(body) != {"version", "claim", "config", "shape", "faces", "excluded_faces"}:
            raise ValueError("schema")
        if body["version"] != 1 or type(body["version"]) is not int or body["claim"] != claim or body["config"] != config.to_dict():
            raise ValueError("claim_config_or_version_mismatch")
        shape = body["shape"]
        if not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3 or any(type(x) is not int or x < 1 or x > 100000 for x in shape):
            raise ValueError("shape")
        faces = body["faces"]
        if not isinstance(faces, list) or not 1 <= len(faces) <= config.max_faces:
            raise ValueError("faces")
        for index, record in enumerate(faces, 1):
            if not isinstance(record, dict) or set(record) != {"bbox", "index", "blocks"}:
                raise ValueError("face_schema")
            if type(record["index"]) is not int or record["index"] != index:
                raise ValueError("face_index")
            if type(record["blocks"]) is not int or not config.min_blocks <= record["blocks"] <= 100000000:
                raise ValueError("block_count")
            box = record["bbox"]
            if not isinstance(box, list) or any(type(x) is not int for x in box) or clip_boxes([box], shape)[0] != box:
                raise ValueError("bbox")
        if not isinstance(body["excluded_faces"], list) or len(body["excluded_faces"]) > 10000:
            raise ValueError("exclusions")
        return body, "valid"
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return None, str(exc)

def set_structure(faces, variant):
    if not faces or any(not f.get("valid", False) for f in faces):
        return False, "invalid_or_missing_face_evidence"
    if variant == "roi_qim":
        return True, "roi_baseline_any_face"
    indices = [f["index"] for f in faces]
    counts = [f["count"] for f in faces]
    if variant == "no_count":
        return len(indices) == len(set(indices)), "count_check_disabled"
    if len(set(counts)) != 1:
        return False, "inconsistent_counts"
    if len(faces) != counts[0]:
        return False, "missing_or_extra_face"
    if variant == "no_id":
        return True, "index_uniqueness_disabled"
    if sorted(indices) != list(range(1, counts[0]+1)):
        return False, "duplicated_or_missing_index"
    return True, "complete_unique_set"

def verify(image, detector, config, master, claim, metadata=None, mode="auto", threshold=1.0, use_voting=True):
    if mode not in {"auto", "metadata", "blind"} or not 0 < threshold <= 1:
        raise ValueError("Invalid verification mode/threshold")
    start = time.perf_counter()
    result = {"decision": "UNDECIDABLE", "reason": "missing_expected_claim", "score": 0.0,
              "structural_ok": False, "faces": [], "candidate_count": 0, "mode": mode,
              "config_sha256": config.fingerprint, "tag_bits": config.tag_bits,
              "metadata_channel": "none"}
    if claim is None:
        result["total_ms"] = (time.perf_counter()-start)*1000
        return result
    validate_claim(claim, config)
    metadata_start = time.perf_counter()
    # Default metadata source is the channel embedded INSIDE the image (Y LSB,
    # paper Algorithm 2 ExtractLSB). An explicit sidecar argument is accepted
    # only for debug/preserved-channel controls. Blind mode reads neither.
    metadata_channel = "none"
    if mode == "blind":
        body, meta_status = None, "ignored_for_blind_control"
    else:
        supplied = metadata
        if supplied is not None:
            metadata_channel = "sidecar"
        else:
            supplied = extract_metadata_lsb(image)
            metadata_channel = "image_lsb" if supplied is not None else "none"
        body, meta_status = validate_metadata(supplied, config, master, claim)
        if body is None:
            metadata_channel = "none"
    result.update(metadata_status=meta_status, metadata_channel=metadata_channel,
                  metadata_ms=(time.perf_counter()-metadata_start)*1000)
    if mode == "metadata" and body is None:
        result.update(reason="metadata_required_but_invalid", total_ms=(time.perf_counter()-start)*1000)
        return result
    actual_mode = "metadata" if body is not None else "blind"
    result["mode"] = actual_mode
    detect_start = time.perf_counter()
    boxes = detector.detect(image)
    result.update(detection_ms=(time.perf_counter()-detect_start)*1000, detected_faces=len(boxes))
    # Metadata speeds decoding but never substitutes for detecting current faces.
    matches = {}
    if body is not None and body["shape"] == list(image.shape):
        pairs = sorted([(iou(box, r["bbox"]), i, j) for i, box in enumerate(boxes) for j, r in enumerate(body["faces"])], reverse=True)
        used = set()
        for overlap, i, j in pairs:
            if overlap >= 0.25 and i not in matches and j not in used:
                matches[i] = j
                used.add(j)
    decode_start = time.perf_counter()
    faces = []
    for number, box in enumerate(boxes):
        roi = crop(image, box)
        if min(roi.shape[:2]) < config.min_roi:
            faces.append({"valid": False, "reason": "unqualified_current_face", "score": 0.0, "bbox": box})
            continue
        candidates = []
        if body is not None:
            if number in matches:
                candidates.append((body["faces"][matches[number]]["bbox"], config.block, config.stride))
            if not candidates or candidates[0][0] != box:
                candidates.append((box, config.block, config.stride))
        else:
            # Blind search: cover the full block-grid phase lattice at the public
            # default geometry. blind_offsets must contain one representative per
            # residue class mod config.block (e.g. 0,+-1..+-5,6 for 12-px blocks),
            # ordered near-to-far; bottom/right drift changes only the number of
            # available blocks.
            candidates = []
            for dy in config.blind_offsets:
                for dx in config.blind_offsets:
                    geometry = [max(0, box[0]+dx), max(0, box[1]+dy), box[2], box[3]]
                    candidate = (geometry, config.block, config.stride)
                    if candidate not in candidates:
                        candidates.append(candidate)
            for b, s in candidate_configs(config):
                candidate = (box, b, s)
                if candidate not in candidates:
                    candidates.append(candidate)
            candidates = candidates[:config.max_candidates]
        decoded = []
        for geometry, block, stride in candidates:
            if min(crop(image, geometry).shape[:2]) < block:
                continue
            result["candidate_count"] += 1
            try:
                item = decode_roi(crop(image, geometry), block, stride, config, master, claim, use_voting)
            except ValueError:
                continue
            item["geometry"] = geometry
            decoded.append(item)
            # Computational short-circuit only: a valid perfect decode already
            # maximizes (valid, score) under the selection key; later candidates
            # can differ only in tie diagnostics (blocks count).
            if item["valid"] and item["score"] >= 1.0:
                break
        best = max(decoded, key=lambda x: (x["valid"], x["score"], x.get("blocks", 0))) if decoded else {"valid": False, "reason": "no_candidate", "score": 0.0}
        faces.append({**best, "bbox": box, "candidates_tested": len(decoded)})
    result.update(faces=faces, decoding_ms=(time.perf_counter()-decode_start)*1000)
    # The intentionally weak ROI baseline uses any face, to quantify the benefit of set checking.
    evidence = [f for f in faces if f.get("valid")] if config.variant == "roi_qim" else faces
    structural_ok, reason = set_structure(evidence, config.variant)
    raw_score = (max if config.variant == "roi_qim" else min)([f["score"] for f in evidence], default=0.0)
    result.update(structural_ok=structural_ok, reason=reason, raw_score=raw_score, score=raw_score if structural_ok else 0.0)
    unavailable = not boxes or any(f["reason"] in {"unqualified_current_face", "insufficient_blocks", "no_candidate"} for f in faces)
    if unavailable and config.variant != "roi_qim":
        result.update(decision="UNDECIDABLE", reason="current_face_evidence_unavailable", structural_ok=False, score=0.0)
    elif not structural_ok or raw_score < threshold:
        result.update(decision="REJECTED", reason=reason if not structural_ok else "tag_score_below_threshold")
    else:
        result["decision"] = "ACCEPTED"
    result.update(total_ms=(time.perf_counter()-start)*1000,
                  assertion_scope="enrolled_watermark_evidence_only; no physical co-presence or pre-enrollment authenticity proof")
    return result
