"""Read-only SHA256 comparison against the source snapshot created before authoring."""
import hashlib
import json
from pathlib import Path
root=Path("C:/Users/dell'/PycharmProjects/CoPhotoMark")
snapshot=Path(__file__).resolve().parents[1]/"docs/original_source_hashes.json"
before=json.loads(snapshot.read_text(encoding="utf-8"))
changed=[]
for relative,digest in before.items():
    path=root/relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        changed.append(relative)
if changed:
    raise SystemExit("Changed since snapshot (may include user edits): "+repr(changed))
print(f"Verified {len(before)} original Python/requirements files unchanged.")
