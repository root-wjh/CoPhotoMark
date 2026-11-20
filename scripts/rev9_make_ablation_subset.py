import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophotomark.io import read_jsonl, write_jsonl

run = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
enr = read_jsonl(run / "enrolled" / "enrollment.jsonl")
key0 = min(r["key_id"] for r in enr if r.get("status") == "enrolled")
ok = [r for r in enr if r.get("status") == "enrolled" and r["key_id"] == key0]
val_ids = sorted({r["source_id"] for r in ok if r["split"] == "validation"})
test_ids = sorted({r["source_id"] for r in ok if r["split"] == "test"})
subset_ids = set(val_ids) | set(test_ids[: len(val_ids)])
sources = [r for r in read_jsonl(run / "sources.jsonl") if r["source_id"] in subset_ids]
write_jsonl(out, sources)
print(f"key0={key0} validation={len(val_ids)} test_pool={len(test_ids)} subset_sources={len(sources)} -> {out}")
