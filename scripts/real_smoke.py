import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import secrets
from cophotomark.detector import InsightFaceDetector
from cophotomark.engine import enroll, verify
from cophotomark.io import read_image, write_image, write_json
from cophotomark.protocol import Config
from cophotomark.experiments import environment

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",default="runs/real_smoke")
    parser.add_argument("--limit",type=int,default=3)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    detector=InsightFaceDetector()
    config=Config()
    rows=[]
    for path in sorted(Path(args.input).rglob("*.jpg"))[:args.limit]:
        row={"source":str(path),"purpose":"REAL-IMAGE SMOKE TEST ONLY; not large-scale evaluation"}
        try:
            master=secrets.token_bytes(32)
            result=enroll(read_image(path),detector,config,master)
            row.update({k:result[k] for k in ("detected_faces","enrolled_faces","excluded_faces","total_ms","quality")})
            image_path=out/(path.stem+"_marked.png")
            write_image(image_path,result["image"])
            # Reload lossless persisted PNG before verification.
            row["blind"]=verify(read_image(image_path),detector,config,master,result["claim"],mode="blind")
            row["metadata"]=verify(read_image(image_path),detector,config,master,result["claim"],result["metadata"],mode="metadata")
            row["status"]="ok"
        except Exception as exc:
            row.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(path.name,row["status"],row.get("reason",row.get("blind",{}).get("decision")))
    write_json(out/"results.json",{"environment":environment(detector),"rows":rows})

if __name__=="__main__":
    main()
