"""Finish the already-running pre-verifier batch without a second pricing writer."""
from pathlib import Path
import json
import subprocess
import sys
import time

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    output=root/'experiments/artifacts/comparison_v3'
    print('[final evidence] waiting for all five forecast seeds and complete pricing tables',flush=True)
    while True:
        path=output/'comparison_status.json'
        if path.exists():
            status=json.loads(path.read_text('utf-8'))
            if status['pricing_complete'] and all(r['full_comparison_complete'] for r in status['forecast']):break
        time.sleep(15)
    print('[final evidence] starting full raw-evidence verification',flush=True)
    result=subprocess.run([sys.executable,'-u','-X','utf8','-m','experiments.comparison_v3','verify'],cwd=root)
    raise SystemExit(result.returncode)
