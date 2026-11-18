from __future__ import annotations
import hashlib
import platform
import secrets
import sys
import time
import os
import subprocess
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from .attacks import default_plan, face_attack, transform, exact_reorder_boxes
from .engine import enroll, verify
from .detector import FixedBoxes
from .lsbmeta import extract_metadata_lsb
from .protocol import Config
from .io import canonical, read_image, read_json, read_jsonl, sha256_file, writable, write_csv, write_image, write_json, write_jsonl
from .statistics import accepted, clopper_pearson, cluster_bootstrap, binomial_reference

def load_key(path):
    value = read_json(path)
    raw = bytes.fromhex(value["master_key_hex"])
    if len(raw) != 32:
        raise ValueError("Expected a 256-bit master key")
    return raw

def environment(detector=None):
    import importlib.metadata as metadata
    versions = {}
    for name in ("numpy", "scipy", "opencv-python", "scikit-image", "insightface", "onnxruntime", "onnxruntime-gpu"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed under this distribution name"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "nvidia-smi unavailable"
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "gpu_inventory": gpu,
            "cv2_build": cv2.getBuildInformation(), "versions": versions,
            "detector": getattr(detector, "info", None), "timer": "perf_counter wall-clock",
            "compute_scope": "ONNX face detector uses recorded session provider; DCT, attacks and statistics use CPU"}

def enroll_manifest(sources, output, detector, config, key_paths, message, run_seed=20261031, debug_metadata=False):
    output = Path(output).resolve()
    outputs = []
    for key_index, key_path in enumerate(key_paths):
        master = load_key(key_path)
        for source in sources:
            record = {**source, "key_id": hashlib.sha256(master).hexdigest()[:16], "key_path": str(Path(key_path).resolve()),
                      "enrollment_seed": run_seed, "config_sha256": config.fingerprint}
            stem = source["source_id"].replace(":", "_") + f"_k{key_index}"
            try:
                image = read_image(source["path"])
                # Fixed seeds reproduce experimental nonces, not deployment key generation.
                from .protocol import new_claim
                claim = new_claim(message, config, image_id=source["source_id"])
                claim["nonce"] = hashlib.sha256(canonical({"seed": run_seed, "source": source["source_id"], "key_index": key_index})).hexdigest()[:32]
                result = enroll(image, detector, config, master, message, claim)
                image_path, meta_path, claim_path = output/f"{stem}.png", output/f"{stem}.cpm.json", output/f"{stem}.claim.json"
                write_image(image_path, result["image"])
                if debug_metadata:
                    write_json(meta_path, result["metadata"])
                write_json(claim_path, result["claim"])
                # Capture per-face bbox data for R9 face-size-band analysis.
                per_face_bboxes = [f["bbox"] for f in result["metadata"].get("faces", [])]
                record.update(status="enrolled", watermarked_path=str(image_path), metadata_path=str(meta_path) if debug_metadata else None, claim_path=str(claim_path),
                              metadata_channel="image_embedded_ylsb",
                              metadata_note="Metadata (M,sigma_meta) is inside the PNG; debug sidecar is disabled by default; expected claim remains externally trusted",
                              detected_faces=result["detected_faces"], enrolled_faces=result["enrolled_faces"], excluded_faces=result["excluded_faces"],
                              coverage=result["coverage"], quality=result["quality"], enrollment_ms=result["total_ms"],
                              per_face_bboxes=per_face_bboxes)
            except Exception as exc:
                record.update(getattr(exc, "statistics", {}))
                record.update(status="enrollment_failed", reason=f"{type(exc).__name__}: {exc}")
            outputs.append(record)
    write_jsonl(output/"enrollment.jsonl", outputs)
    write_json(output/"environment.json", environment(detector))
    return outputs

def build_trials(records, output, plan=None, repeats=1, seed=20261031, face_operations=True):
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    output = Path(output).resolve()
    successful = [r for r in records if r.get("status") == "enrolled"]
    trials, skipped = [], []
    plan = default_plan() if plan is None else plan
    for record in successful:
        source_id = record["source_id"]
        image = read_image(record["watermarked_path"])
        original = read_image(record["path"])
        # A generated enrollment must carry its metadata. A debug file may
        # never silently rescue an incorrectly delivered image.
        meta = extract_metadata_lsb(image)
        if meta is None:
            raise ValueError(f"Enrollment image has no embedded metadata: {record['watermarked_path']}")
        boxes = [r["bbox"] for r in meta["faces"]]
        # Use an independent source group inside the SAME split. Never pull a
        # donor from validation into test. Seeded choice varies by repetition.
        donors = [r for r in successful if r["split"] == record["split"]
                  and r["source_id"] != source_id and r["group_id"] != record["group_id"]]
        specs = list(plan)
        if face_operations:
            specs += [{"name": n, "label": "negative"} for n in ("delete", "duplicate", "replace", "copy_marked_face", "watermark_transplant", "insert_unmarked", "insert_marked", "boundary_tamper")]
            specs += [{"name": "reorder", "label": "genuine", "note": "Spatial order is outside the assertion; resampling can nevertheless destroy evidence"},
                      {"name": "reorder_exact", "label": "genuine", "note": "Whole-ROI translation without interpolation/cropping; infeasible layouts logged as skipped"}]
            specs += [{"name": "collusion", "label": "negative", "note": "Average independently keyed observations of the same host"},
                      {"name": "erase_reembed_wrong_key", "label": "negative", "sigma": 3.0},
                      {"name": "detector_miss_control", "label": "negative", "note": "Artificially suppress one detected face; controlled failure, not a visual attack"}]
        for index, spec in enumerate(specs):
            for repetition in range(repeats):
                trial_id = hashlib.sha256(canonical({"source": source_id, "key": record["key_id"], "spec": spec, "repeat": repetition, "seed": seed})).hexdigest()[:24]
                rng = np.random.default_rng(int(trial_id[:16], 16))
                donor_record = None
                try:
                    donor_image = donor_original = donor_box = None
                    donor_attacks = {"replace", "copy_marked_face", "watermark_transplant", "insert_unmarked", "insert_marked"}
                    if spec["name"] in donor_attacks:
                        if not donors:
                            raise ValueError("No independent donor source group within this split")
                        donor_record = donors[int(rng.integers(len(donors)))]
                        donor_image = read_image(donor_record["watermarked_path"])
                        donor_original = read_image(donor_record["path"])
                        donor_meta = extract_metadata_lsb(donor_image)
                        if donor_meta is None:
                            raise ValueError("Donor enrollment has no embedded metadata")
                        donor_box = donor_meta["faces"][int(rng.integers(len(donor_meta["faces"])))]["bbox"]
                    if spec["name"] == "collusion":
                        peers = [r for r in successful if r["source_id"] == source_id and r["split"] == record["split"]]
                        peer_images = [read_image(r["watermarked_path"]) for r in peers]
                        attacked = face_attack(image, boxes, spec, rng, peers=peer_images)
                    elif spec["name"] == "erase_reembed_wrong_key":
                        erased = image.copy()
                        for x1,y1,x2,y2 in boxes:
                            erased[y1:y2,x1:x2] = cv2.GaussianBlur(erased[y1:y2,x1:x2], (19,19), 3.0)
                        attack_key = hashlib.sha256(canonical({"trial_id": trial_id, "purpose": "public-wrong-reembedding-control"})).digest()
                        attacked = enroll(erased, FixedBoxes(boxes), Config(**meta["config"]), attack_key,
                                          claim=read_json(record["claim_path"]))["image"]
                    elif spec["name"] == "detector_miss_control":
                        attacked = image.copy()
                    elif spec["name"] in {s["name"] for s in default_plan()}:
                        attacked = transform(image, spec, rng)
                    else:
                        # Insertion/replacement of unmarked faces uses the unmarked donor.
                        chosen = donor_original if spec["name"] in {"replace", "insert_unmarked"} else donor_image
                        attacked = face_attack(image, boxes, spec, rng, chosen, donor_box, donor_original, original)
                    image_path = output/"images"/f"{trial_id}.png"
                    write_image(image_path, attacked)
                    # Default verification reads metadata from INSIDE the image
                    # (may survive pixel-preserving storage, fragile to edits).
                    # The external sidecar is attached only for the explicit
                    # preserved-channel control used to isolate ROI resync.
                    keep_metadata = spec.get("metadata_channel") == "preserved"
                    if keep_metadata and not record.get("metadata_path"):
                        raise ValueError("Preserved-sidecar control requires explicit debug metadata enrollment")
                    trials.append({"trial_id": trial_id, "source_id": source_id, "source_sha256": record["source_sha256"],
                                   "group_id": record["group_id"], "split": record["split"], "dataset": record["dataset"],
                                   "image_path": str(image_path), "claim_path": record["claim_path"],
                                   "metadata_path": record["metadata_path"] if keep_metadata else None,
                                   "key_path": record["key_path"], "key_id": record["key_id"],
                                   "attack": spec["name"], "attack_spec": spec, "label": spec.get("label", "stress"),
                                   "repetition": repetition, "seed": seed, "donor_source_id": donor_record["source_id"] if donor_record and spec["name"] in {"replace", "copy_marked_face", "watermark_transplant", "insert_unmarked", "insert_marked"} else None,
                                   "enrolled_faces": record["enrolled_faces"], "coverage": record["coverage"],
                                   "baseline_detected_faces": record["detected_faces"],
                                   "expected_face_boxes_control": exact_reorder_boxes(boxes, image.shape) if spec["name"] == "reorder_exact" else None,
                                   "source_short_edge": min(record.get("short_edges", [0])),
                                   "provenance": "author_generated_digital_simulation"})
                except Exception as exc:
                    skipped.append({"trial_id": trial_id, "source_id": source_id, "key_id": record["key_id"],
                                    "split": record["split"], "spec": spec, "repetition": repetition, "reason": str(exc)})
        # Wrong-key and wrong-message negatives must go through the SAME candidate search.
        for wrong_index in range(5):
            trial_id = hashlib.sha256(f"{source_id}:{record['key_id']}:wrong:{wrong_index}:{seed}".encode()).hexdigest()[:24]
            wrong_path = output/"wrong_keys"/f"{wrong_index}.json"
            if not wrong_path.exists():
                write_json(wrong_path, {"master_key_hex": secrets.token_hex(32), "purpose": "wrong-key experimental control"})
            trials.append({"trial_id": trial_id, "source_id": source_id, "source_sha256": record["source_sha256"], "group_id": record["group_id"],
                           "split": record["split"], "dataset": record["dataset"], "image_path": record["watermarked_path"],
                           "claim_path": record["claim_path"], "metadata_path": None, "key_path": str(wrong_path), "key_id": f"wrong_{wrong_index}",
                           "attack": "wrong_key", "label": "negative", "provenance": "author_generated_key_control",
                           "enrolled_faces": record["enrolled_faces"], "coverage": record["coverage"]})
        wrong_claim = read_json(record["claim_path"])
        wrong_claim["message"] += "::wrong-claim-control"
        claim_path = output/"wrong_claims"/(source_id.replace(":", "_")+"_"+record["key_id"]+".json")
        write_json(claim_path, wrong_claim)
        trials.append({"trial_id":hashlib.sha256((source_id+record["key_id"]+"wrong_claim").encode()).hexdigest()[:24],
            "source_id":source_id,"source_sha256":record["source_sha256"],"group_id":record["group_id"],"split":record["split"],
            "dataset":record["dataset"],"image_path":record["watermarked_path"],"claim_path":str(claim_path),"metadata_path":None,
            "key_path":record["key_path"],"key_id":record["key_id"],"attack":"wrong_claim","label":"negative",
            "provenance":"author_generated_message_control","enrolled_faces":record["enrolled_faces"],"coverage":record["coverage"]})
    write_jsonl(output/"trials.jsonl", trials)
    write_jsonl(output/"skipped_attacks.jsonl", skipped)
    coverage = []
    for name in sorted({r["attack"] for r in trials} | {r["spec"]["name"] for r in skipped}):
        generated = [r for r in trials if r["attack"] == name]
        missed = [r for r in skipped if r["spec"]["name"] == name]
        coverage.append({"attack": name, "generated": len(generated), "skipped": len(missed),
                         "source_images": len({r["source_id"] for r in generated}),
                         "full_coverage_multiface_trials": sum(r["coverage"] == "all_detected_faces" and r["enrolled_faces"] >= 2 for r in generated),
                         "skip_reasons": sorted({r["reason"] for r in missed})})
    write_csv(output/"scenario_coverage.csv", coverage)
    write_json(output/"attack_protocol.json", {"seed": seed, "repeats": repeats, "plan": plan,
               "jpeg": "OpenCV imencode/imdecode, quality specified; build recorded in environment.json",
               "protocol_version": "legacy_crop_blur_rect_cropout_v1",
               "interpolation": "INTER_LINEAR", "rotation_border": "BORDER_REFLECT", "blur_border": "BORDER_REFLECT_101",
               "crop_fraction": "retained side fraction; dimensions int(H*r), int(W*r); uniform random x then y; resize back",
               "blur_kernel": "5x5 default; sigma_x=sigma_y=specified sigma",
               "resize": "int-scaled dimensions followed by resize back to original size",
               "cropout": "single uniform-position rectangle, round(H*sqrt(f)), round(W*sqrt(f)); all channels filled with 127 by default; no resize; historical table provenance unconfirmed",
               "randomness": "per-trial NumPy Generator; reproduces distributions, not historical Python/global NumPy random streams",
               "storage": "lossless PNG after attack; JPEG uses in-memory OpenCV encode/decode before PNG",
               "order": "Each attack starts independently from enrolled PNG, except explicitly ordered platform simulation",
               "simulations_are_not": "real platform uploads or physical screenshots"})
    return trials

def evaluation_signature(detector, config, use_voting=True):
    detector_info = {k: v for k, v in getattr(detector, "info", {}).items() if k != "initialization_ms"}
    return hashlib.sha256(canonical({"detector": detector_info, "voting": use_voting, "config": config.to_dict()})).hexdigest()

def evaluate(trials, output, detector, config, modes=("blind", "auto"), calibration=None, use_voting=True, warmup=1):
    output = Path(output).resolve()
    signature = evaluation_signature(detector, config, use_voting)
    if calibration:
        if calibration["config_sha256"] != config.fingerprint:
            raise ValueError("Calibration/config mismatch")
        if calibration.get("evaluation_signature") != signature:
            raise ValueError("Detector/voting protocol differs from validation calibration")
        for mode in modes:
            selected = calibration.get("modes", {}).get(mode)
            if not selected or selected.get("threshold") is None:
                raise ValueError(f"No feasible frozen threshold for {mode}; inspect validation results before test")
        seen_ids = set(calibration["validation_source_ids"])
        seen_groups = set(calibration["validation_groups"])
        seen_hashes = set(calibration["validation_hashes"])
        for row in trials:
            if row["split"] != "test" or row["source_id"] in seen_ids or row["group_id"] in seen_groups or row["source_sha256"] in seen_hashes:
                raise ValueError("Test/validation leakage or non-test rows")
    elif any(r["split"] == "test" for r in trials):
        raise ValueError("Test evaluation requires a frozen validation calibration file")
    if trials and warmup:
        sample = read_image(trials[0]["image_path"])
        for _ in range(warmup):
            detector.detect(sample)
    results = []
    for trial in trials:
        for requested_mode in modes:
            row = dict(trial)
            row["evaluation_signature"] = signature
            row["requested_mode"] = requested_mode
            start = time.perf_counter()
            try:
                load_start = time.perf_counter()
                image = read_image(trial["image_path"])
                claim = read_json(trial["claim_path"]) if trial.get("claim_path") else None
                meta = read_json(trial["metadata_path"]) if trial.get("metadata_path") and requested_mode != "blind" else None
                master = load_key(trial["key_path"])
                io_ms = (time.perf_counter()-load_start)*1000
                threshold = 1.0
                if calibration:
                    selected = calibration["modes"].get(requested_mode)
                    if not selected or selected["threshold"] is None:
                        raise ValueError("No feasible frozen threshold for requested mode")
                    threshold = selected["threshold"]
                trial_detector = detector
                if trial.get("attack") == "detector_miss_control":
                    # Use current detections, then intentionally omit the largest face.
                    detected = detector.detect(image)
                    row["natural_detected_faces_before_control"] = len(detected)
                    detected = sorted(detected, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
                    trial_detector = FixedBoxes(detected[1:])
                    row["detection_failure_control"] = True
                result = verify(image, trial_detector, config, master, claim, meta, requested_mode, threshold, use_voting)
                actual_mode = result["mode"]
                row.update(result)
                # Calibration separates requested deployment pipelines, actual route remains logged.
                row.update(mode=requested_mode, actual_mode=actual_mode, threshold=threshold, io_ms=io_ms,
                           end_to_end_ms=(time.perf_counter()-start)*1000, status="ok", voting=use_voting)
                baseline_count = trial.get("baseline_detected_faces")
                current_count = result.get("detected_faces")
                if baseline_count is not None and current_count is not None:
                    row["detected_face_count_delta"] = current_count - baseline_count
                    if trial["attack"] in {"insert_marked", "insert_unmarked"}:
                        row["intended_count_change_observed"] = current_count > baseline_count
                    elif trial["attack"] in {"delete", "detector_miss_control"}:
                        row["intended_count_change_observed"] = current_count < baseline_count
            except Exception as exc:
                row.update(status="error", decision="UNDECIDABLE", structural_ok=False, score=0.0,
                           reason=f"{type(exc).__name__}: {exc}", mode=requested_mode,
                           end_to_end_ms=(time.perf_counter()-start)*1000)
            results.append(row)
            if len(results) % 20 == 0:
                print(f"evaluated {len(results)}/{len(trials)*len(modes)} mode-trials", flush=True)
    write_jsonl(output/"scores.jsonl", results)
    write_json(output/"environment.json", {**environment(detector), "warmup_detection_calls": warmup,
               "initialization_excluded_from_warm_latency": True, "io_included_in_end_to_end": True})
    return results

def summarize(rows, enrollments=None):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("mode"), row.get("attack"), row.get("label"))].append(row)
    summary = []
    for (mode, attack, label), subset in groups.items():
        def rate(sample):
            return sum(r.get("decision") == "ACCEPTED" for r in sample)/len(sample)
        n = len(subset)
        ac = sum(r.get("decision") == "ACCEPTED" for r in subset)
        rejected = sum(r.get("decision") == "REJECTED" for r in subset)
        undecidable = sum(r.get("decision") == "UNDECIDABLE" for r in subset)
        times = [r["end_to_end_ms"] for r in subset]
        summary.append({"mode": mode, "attack": attack, "label": label, "trials": n,
                        "source_images": len({r["source_id"] for r in subset}), "keys": len({r["key_id"] for r in subset}),
                        "accepted": ac, "rejected": rejected, "undecidable": undecidable,
                        "accept_rate": ac/n, "explicit_reject_rate": rejected/n, "undecidable_rate": undecidable/n,
                        "nonaccept_rate": (rejected+undecidable)/n, "accept_rate_cluster_ci95": cluster_bootstrap(subset, rate),
                        "latency_mean_ms": float(np.mean(times)), "latency_std_ms": float(np.std(times, ddof=1)) if n>1 else 0.0,
                        "latency_median_ms": float(np.median(times)), "latency_p95_ms": float(np.quantile(times, 0.95)),
                        "runtime_errors": sum(r.get("status") == "error" for r in subset)})
        observed = [r for r in subset if "intended_count_change_observed" in r]
        summary[-1].update(intended_count_change_assessed=len(observed),
                           intended_count_change_observed=sum(r["intended_count_change_observed"] for r in observed))
    counts = {"source_images": len({r["source_id"] for r in rows}), "trials_including_modes": len(rows),
              "distinct_attack_trials": len({r["trial_id"] for r in rows}),
              "enrollment_keys": len({r["key_id"] for r in rows if r.get("attack") != "wrong_key"}),
              "wrong_key_controls": len({r["key_id"] for r in rows if r.get("attack") == "wrong_key"}),
              "note": "Source-image cluster bootstrap; near-duplicate/identity grouping must be supplied before split."}
    if enrollments is not None:
        success = [r for r in enrollments if r["status"] == "enrolled"]
        counts.update(enrollment_attempts=len(enrollments), enrollment_failures=len(enrollments)-len(success),
                      detected_face_observations=sum(r.get("detected_faces", 0) for r in enrollments),
                      enrolled_face_observations=sum(r.get("enrolled_faces", 0) for r in success),
                      excluded_face_observations=sum(r.get("excluded_faces", 0) for r in enrollments),
                      partial_coverage_enrollments=sum(r.get("coverage") == "partial_detected_faces" for r in success),
                      full_coverage_multiface_enrollments=sum(r.get("coverage") == "all_detected_faces" and r.get("enrolled_faces", 0) >= 2 for r in success),
                      multi_face_successful_sources=len({r["source_id"] for r in success if r["enrolled_faces"]>=2}))
        counts["enrollment_by_split"] = {}
        for split in sorted({r["split"] for r in enrollments}):
            attempted=[r for r in enrollments if r["split"]==split]
            counts["enrollment_by_split"][split] = {"attempts":len(attempted),
                "successes":sum(r["status"]=="enrolled" for r in attempted),
                "source_images":len({r["source_id"] for r in attempted}),
                "enrollment_success_rate":sum(r["status"]=="enrolled" for r in attempted)/len(attempted)}
        # Operational rate keeps unsuccessful enrollment attempts in the denominator.
        counts["clean_operational_acceptance"] = {}
        for mode in sorted({r["mode"] for r in rows}):
            subsets=[r for r in enrollments if r["split"] in {x["split"] for x in rows}]
            rates=[]
            for record in subsets:
                clean=[r for r in rows if r["source_id"]==record["source_id"] and r["key_id"]==record["key_id"] and r["mode"]==mode and r["attack"]=="clean"]
                rates.append(sum(r["decision"]=="ACCEPTED" for r in clean)/len(clean) if clean else 0.0)
            counts["clean_operational_acceptance"][mode] = float(np.mean(rates)) if rates else None
    return {"counts": counts, "by_attack": summary}

def far_curve(rows):
    output=[]
    for mode in sorted({r["mode"] for r in rows}):
        negatives=[r for r in rows if r["mode"]==mode and r.get("attack")=="wrong_key" and r.get("status")=="ok"]
        if not negatives:
            continue
        bits=negatives[0]["tag_bits"]
        candidates=max(r.get("candidate_count",0) for r in negatives)
        for matches in range(1,bits+1):
            threshold=matches/bits
            count=sum(accepted(r,threshold) for r in negatives)
            output.append({"mode":mode,"threshold":threshold,"tag_bits":bits,"allowed_tag_errors":bits-matches,
                "trials":len(negatives),"source_images":len({r["source_id"] for r in negatives}),"false_accepts":count,
                "empirical_far":count/len(negatives),"trial_ci_independence_assumption":clopper_pearson(count,len(negatives)),
                **binomial_reference(bits,bits-matches,candidates)})
    return output

def summarize_r9(rows, enrollments):
    """R9 supplementary metrics: curves vs. enrollment count N and per-face size.

    Aggregates undecidable/exclusion rates by (mode, N) and by
    (mode, face-size-band) for benign operations, complementing the
    per-attack summarize() table with population-level curves.
    """
    # --- Undecidable / acceptance rate vs. enrollment face count N ---
    n_groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        n = row.get("enrolled_faces", 0)
        n_groups[(row["mode"], n)].append(row)
    by_n = []
    for (mode, n), subset in sorted(n_groups.items()):
        n_trials = len(subset)
        accepts = sum(r["decision"] == "ACCEPTED" for r in subset)
        rejects = sum(r["decision"] == "REJECTED" for r in subset)
        undecs = sum(r["decision"] == "UNDECIDABLE" for r in subset)
        # "exclusion rate" (R9) — faces that were detected but fell below
        # min_roi or min_blocks and were therefore excluded from enrollment.
        excluded_total = sum(r.get("excluded_faces", 0) for r in subset)
        detected_total = sum(r.get("detected_faces", 0) for r in subset)
        excl_rate = excluded_total / detected_total if detected_total > 0 else None
        benign = [r for r in subset if r.get("label") == "genuine"]
        benign_accept = sum(r["decision"] == "ACCEPTED" for r in benign) / len(benign) if benign else None
        by_n.append({
            "mode": mode, "enrolled_faces_N": n, "trials": n_trials,
            "accepted": accepts, "rejected": rejects, "undecidable": undecs,
            "accept_rate": accepts / n_trials,
            "undecidable_rate": undecs / n_trials,
            "explicit_reject_rate": rejects / n_trials,
            "excluded_faces_total": excluded_total,
            "detected_faces_total": detected_total,
            "exclusion_rate": excl_rate,
            "benign_accept_rate": benign_accept,
            "benign_count": len(benign),
        })

    # --- Per-face short-edge size analysis (from enrollment records) ---
    face_size_rows = []
    for rec in enrollments or []:
        if rec.get("status") != "enrolled":
            continue
        for bbox in rec.get("per_face_bboxes", []):
            x1, y1, x2, y2 = bbox
            short = min(x2 - x1, y2 - y1)
            if short < 40:
                band = "<40"
            elif short < 64:
                band = "40-63"
            elif short < 96:
                band = "64-95"
            elif short < 128:
                band = "96-127"
            else:
                band = "128+"
            face_size_rows.append({
                "source_id": rec.get("source_id"),
                "key_id": rec.get("key_id"),
                "split": rec.get("split", "unknown"),
                "short_edge": short, "band": band,
                "enrolled_faces": rec.get("enrolled_faces"),
            })
    # Face size is enrollment-level (no mode); report per split bands.
    face_size_summary = []
    for (split, band), subset in sorted(defaultdict(list, {(r.get("split", "unknown"), r["band"]): list() for r in face_size_rows}).items()):
        subset = [f for f in face_size_rows if f.get("split", "unknown") == split and f["band"] == band]
        if not subset:
            continue
        face_size_summary.append({
            "split": split, "band": band, "faces": len(subset),
            "sources": len({f["source_id"] for f in subset}),
            "mean_short_edge": float(np.mean([f["short_edge"] for f in subset])),
            "p10_short_edge": float(np.percentile([f["short_edge"] for f in subset], 10)),
            "median_short_edge": float(np.median([f["short_edge"] for f in subset])),
            "p90_short_edge": float(np.percentile([f["short_edge"] for f in subset], 90)),
        })

    # --- Undecidable by attack × N (for paper Fig.4-like curve) ---
    per_attack_n = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        per_attack_n[(row["mode"], row["attack"], row.get("enrolled_faces", 0))].append(row)
    by_attack_n = []
    for (mode, attack, n), subset in sorted(per_attack_n.items()):
        by_attack_n.append({
            "mode": mode, "attack": attack, "enrolled_faces_N": n,
            "trials": len(subset),
            "accept_rate": sum(r["decision"] == "ACCEPTED" for r in subset) / len(subset),
            "undecidable_rate": sum(r["decision"] == "UNDECIDABLE" for r in subset) / len(subset),
            "reject_rate": sum(r["decision"] == "REJECTED" for r in subset) / len(subset),
        })

    return {"by_N": by_n, "per_face_size": face_size_summary, "by_attack_N": by_attack_n}
