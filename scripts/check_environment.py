import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cophotomark.detector import InsightFaceDetector
from cophotomark.experiments import environment
from cophotomark.io import write_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--provider',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--model-root')
    parser.add_argument('--output')
    args=parser.parse_args()
    detector=InsightFaceDetector(root=args.model_root,provider=args.provider)
    detector.detect(np.full((640,640,3),127,dtype=np.uint8))
    report=environment(detector)
    report['inference_probe']='completed on loaded detection session'
    if args.output:
        write_json(args.output,report)
    print(report['detector'])
    print(report['gpu_inventory'])
    print(report['compute_scope'])

if __name__=='__main__':
    main()
