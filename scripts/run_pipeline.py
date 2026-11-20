import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--config",default="configs/full64.json")
    parser.add_argument("--output",default="runs/full64")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--groups")
    parser.add_argument("--model-root", help="Parent containing models/buffalo_l; default ~/.insightface")
    parser.add_argument("--sources", help="Reuse frozen sources.jsonl for paired baselines/ablations; skip inventory/split")
    parser.add_argument("--key-files", nargs="+", help="Reuse experimental keys for paired baseline runs")
    parser.add_argument("--keys",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--target-far",type=float,default=0.001)
    parser.add_argument("--provider",choices=["cpu","cuda"],
                        help="InsightFace detection provider (GPU hosts: cuda). "
                             "Same provider is enforced across validation and test via the evaluation signature")
    args=parser.parse_args()
    if args.provider:
        os.environ["COPHOTOMARK_PROVIDER"]=args.provider
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Select a new empty output directory to avoid stale/mixed experiment artifacts")
    def run(*words):
        if words[0] in {"inventory", "enroll", "evaluate"} and args.model_root:
            words = (*words, "--model-root", args.model_root)
        command=[sys.executable,"-m","cophotomark",*map(str,words)]
        print("RUN:",subprocess.list2cmdline(command),flush=True)
        subprocess.run(command,cwd=root,check=True)
    if args.keys<1 or args.repeats<1:
        raise ValueError("keys/repeats must be positive")
    key_paths=[Path(p).resolve() for p in (args.key_files or [])]
    for index in range(0 if key_paths else args.keys):
        path=out/"keys"/f"key_{index}.json"
        run("keygen","--output",path)
        key_paths.append(path)
    source_path = Path(args.sources).resolve() if args.sources else out/"sources.jsonl"
    inv=["inventory","--input",args.input,"--dataset",args.dataset,"--config",args.config,"--output",out/"inventory.jsonl"]
    if args.limit:
        inv += ["--limit",args.limit]
    if args.groups:
        inv += ["--groups",args.groups]
    if not args.sources:
        run(*inv)
        run("split","--input",out/"inventory.jsonl","--output",source_path)
    else:
        from cophotomark.io import read_jsonl
        if not {"validation", "test"}.issubset({r["split"] for r in read_jsonl(source_path)}):
            raise ValueError("Frozen sources must contain validation and test")
    run("enroll","--input",source_path,"--config",args.config,"--keys",*key_paths,"--output",out/"enrolled")
    run("attacks","--enrollments",out/"enrolled/enrollment.jsonl","--output",out/"attacks","--repeats",args.repeats)
    run("evaluate","--trials",out/"attacks/trials.jsonl","--split","validation","--config",args.config,"--output",out/"validation")
    run("calibrate","--scores",out/"validation/scores.jsonl","--config",args.config,"--target-far",args.target_far,"--output",out/"calibration.json")
    run("evaluate","--trials",out/"attacks/trials.jsonl","--split","test","--config",args.config,"--calibration",out/"calibration.json","--output",out/"test")
    run("report","--scores",out/"test/scores.jsonl","--enrollments",out/"enrolled/enrollment.jsonl","--output",out/"report")
    run("plot","--scores",out/"test/scores.jsonl","--output",out/"report")
    # Also generate calibration ROC and validation summary
    run("evaluate","--trials",out/"attacks/trials.jsonl","--split","validation","--config",args.config,"--calibration",out/"calibration.json","--output",out/"validation",)
    print(f"\nPipeline complete. Artifacts in {out}")
    print("Key outputs for paper:")
    print("  report/by_attack.csv               ← 主表 Table 2 数据源")
    print("  report/far_theory_vs_empirical.csv ← R3 审稿 FAR 理论 vs 实测")
    print("  report/r9_by_N.csv                  ← R9 随合照人数 N 的曲线")
    print("  report/r9_face_size.csv             ← R9 人脸短边分布")
    print("  report/r9_plots/r9_by_N.png         ← 随 N 曲线图")
    print("  calibration.json                    ← 冻结阈值")
    print("  attacks/attack_protocol.json        ← 攻击协议精确描述（含 Crop/Cropout 定义）")

if __name__=="__main__":
    main()
