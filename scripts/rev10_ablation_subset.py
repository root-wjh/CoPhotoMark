import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophotomark.io import read_jsonl, write_jsonl


def header_ok(fc):
    return fc.get("count", 0) >= 1 and 1 <= fc.get("index", 0) <= 255


def face64(fc):
    return (16 if header_ok(fc) else 0) + fc.get("tag_matches", 0)


def decide(row, gamma):
    faces = row.get("faces") or []
    N = row["enrolled_faces"]
    det = row.get("detected_faces") or 0
    if not faces or det == 0:
        return "UNDECIDABLE"
    if det != N or len(faces) != N:
        return "REJECTED"
    ok = [fc for fc in faces if header_ok(fc)]
    if not ok:
        return "UNDECIDABLE"
    if len(ok) < N:
        return "REJECTED"
    if sorted(fc["index"] for fc in ok) != list(range(1, N + 1)):
        return "REJECTED"
    if any(fc.get("count") != N for fc in ok):
        return "REJECTED"
    m = min(face64(fc) for fc in faces) / 64.0
    return "ACCEPTED" if m >= gamma else "REJECTED"


run = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
rows = read_jsonl(run / "test" / "scores.jsonl") + read_jsonl(run / "validation" / "scores.jsonl")
by_src = defaultdict(list)
for r in rows:
    by_src[r["source_id"]].append(r)


def gate_pass(rs):
    for r in rs:
        if r["attack"] != "clean":
            continue
        gamma = 1.0 if r["requested_mode"] == "auto" else 0.8125
        if decide(r, gamma) != "ACCEPTED":
            return False
    return bool(rs)


cohort = {s for s, rs in by_src.items() if gate_pass(rs)}
enr = read_jsonl(run / "enrolled" / "enrollment.jsonl")
multi = {r["source_id"] for r in enr
         if r.get("status") == "enrolled" and r.get("enrolled_faces", 0) >= 2}
keep = cohort & multi
sources = [r for r in read_jsonl(run / "sources.jsonl") if r["source_id"] in keep]
write_jsonl(out, sources)
print(f"cohort={len(cohort)} multi_face={len(multi)} subset={len(sources)} -> {out}")
