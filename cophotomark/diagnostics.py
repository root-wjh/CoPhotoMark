from pathlib import Path
from .detector import iou
from .io import read_image, read_jsonl, write_csv, write_json
from .lsbmeta import extract_metadata_lsb

def diagnose_clean(run, output):
    run, output = Path(run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty diagnosis directory")
    enrollments = read_jsonl(run / "enrolled" / "enrollment.jsonl")
    records = {(r["source_id"], r["key_id"]): r for r in enrollments}
    scores = read_jsonl(run / "scores" / "scores.jsonl")
    diagnoses, face_rows = [], []
    metadata_cache = {}
    for row in scores:
        if row.get("attack") != "clean":
            continue
        record = records[(row["source_id"], row["key_id"])]
        image_path = record["watermarked_path"]
        if image_path not in metadata_cache:
            metadata_cache[image_path] = extract_metadata_lsb(read_image(image_path))
        metadata = metadata_cache[image_path]
        original_faces = metadata["faces"] if metadata else []
        candidate_config = metadata["config"] if metadata else {}
        offsets = candidate_config.get("blind_offsets", [])
        current_faces = row.get("faces", [])
        original_count = record["enrolled_faces"]
        detected = row.get("detected_faces", len(current_faces))
        outside_count, imperfect = 0, 0
        for number, face in enumerate(current_faces):
            box = face["bbox"]
            reference = max(original_faces, key=lambda f: iou(box, f["bbox"]), default=None)
            overlap = iou(box, reference["bbox"]) if reference else 0
            dx = box[0]-reference["bbox"][0] if reference and overlap >= 0.25 else None
            dy = box[1]-reference["bbox"][1] if reference and overlap >= 0.25 else None
            outside = dx is not None and (-dx not in offsets or -dy not in offsets)
            outside_count += outside
            imperfect += not face.get("valid", False) or face.get("score", 0) < row.get("threshold", 1.0)
            face_rows.append({"source_id": row["source_id"], "source_path": record["path"], "mode": row["mode"],
                              "current_face": number, "current_bbox": box,
                              "nearest_enrollment_bbox": reference["bbox"] if reference else None,
                              "nearest_enrollment_index": reference["index"] if reference else None,
                              "bbox_iou": overlap, "left_drift_px": dx, "top_drift_px": dy,
                              "exact_original_origin_outside_configured_offsets": outside,
                              "note": "IoU nearest geometry diagnostic, not identity matching; correct block phase can differ by whole blocks",
                              "selected_geometry": face.get("geometry"), "decoded_count": face.get("count"),
                              "decoded_index": face.get("index"), "tag_score": face.get("score"), "reason": face.get("reason")})
        if row.get("decision") == "ACCEPTED":
            classification = "clean_accepted"
        elif detected != original_count:
            classification = "detected_face_count_changed"
        elif row.get("metadata_status") != "valid" and row["mode"] != "blind":
            classification = "metadata_loss_then_decode_failure"
        elif row["mode"] == "blind" and outside_count:
            classification = "blind_origin_search_insufficient_or_block_support_changed"
        else:
            classification = "decode_or_geometry_failure_requires_inspection"
        diagnoses.append({"source_id": row["source_id"], "source_path": record["path"], "mode": row["mode"],
                          "decision": row["decision"], "reason": row["reason"], "classification": classification,
                          "enrolled_faces": original_count, "current_detected_faces": detected,
                          "count_delta": detected-original_count, "metadata_status": row.get("metadata_status"),
                          "faces_with_original_origin_outside_offsets": outside_count,
                          "faces_below_working_threshold_or_invalid": imperfect})
    summary = {"purpose": "Offline analysis of existing logs; not rerun or revised performance evidence",
               "clean_mode_rows": len(diagnoses), "enrollment_attempts": len(enrollments),
               "enrollment_failures": sum(r.get("status") != "enrolled" for r in enrollments),
               "by_mode": {}}
    for mode in sorted({r["mode"] for r in diagnoses}):
        subset = [r for r in diagnoses if r["mode"] == mode]
        summary["by_mode"][mode] = {"trials": len(subset),
            "accepted": sum(r["decision"] == "ACCEPTED" for r in subset),
            "count_change_trials": sum(r["count_delta"] != 0 for r in subset),
            "failed_with_stable_count": sum(r["decision"] != "ACCEPTED" and r["count_delta"] == 0 for r in subset)}
    write_csv(output / "clean_failures.csv", diagnoses)
    write_csv(output / "clean_faces.csv", face_rows)
    write_json(output / "summary.json", summary)
    return summary
