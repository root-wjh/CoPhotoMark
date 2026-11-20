import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cophotomark.diagnostics import diagnose_clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(diagnose_clean(args.run, args.output))

if __name__ == "__main__":
    main()
