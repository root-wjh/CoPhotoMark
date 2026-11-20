import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from cophotomark.detector import FixedBoxes
from cophotomark.engine import enroll, verify
from cophotomark.io import write_image, write_json
from cophotomark.protocol import Config

output = Path(sys.argv[1] if len(sys.argv)>1 else "runs/synthetic_demo").resolve()
image = np.random.default_rng(42).integers(65,190,(144,256,3),dtype=np.uint8)
boxes = [[16,16,112,112],[144,16,240,112]]
key = bytes(range(32))  # Public test key, NEVER a deployment key.
config = Config()
result = enroll(image,FixedBoxes(boxes),config,key)
write_image(output/"marked.png", result["image"])
write_json(output/"claim.json",result["claim"])
write_json(output/"metadata.json",result["metadata"])
reports = {}
reports["clean"] = verify(result["image"],FixedBoxes(boxes),config,key,result["claim"],mode="blind")
reports["missed_face"] = verify(result["image"],FixedBoxes(boxes[:1]),config,key,result["claim"],mode="blind")
copy = result["image"].copy()
copy[16:112,144:240] = copy[16:112,16:112]
reports["duplicate"] = verify(copy,FixedBoxes(boxes),config,key,result["claim"],mode="blind")
write_json(output/"results.json",{"purpose":"SYNTHETIC PROTOCOL TEST ONLY; not paper evidence", "results":reports})
print({name:r["decision"] for name,r in reports.items()})
