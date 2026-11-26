"""Run CPU pricing as each GPU-trained M5 seed becomes ready, then finalize.

This is a local batch job for the active experiment, not a recurring service.
It never starts a second writer for forecast checkpoints.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('experiments/artifacts/comparison_v3'))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--idle-minutes',type=float,default=30.)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];output=args.output.resolve()
    logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
    seeds=json.loads(Path(__file__).with_name('protocol.json').read_text('utf-8'))['seeds']
    def command(stage,extra=()):
        target=logs/f'{stage}_{int(time.time())}.log'
        argv=[sys.executable,'-u','-X','utf8','-m','experiments.comparison_v3',stage,
              '--output',str(output),'--workers',str(args.workers),*map(str,extra)]
        print('[pipeline] starting',stage,list(extra),'log:',target,flush=True)
        with target.open('w',encoding='utf-8') as handle:
            result=subprocess.run(argv,cwd=root,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{stage} failed ({result.returncode}); see {target}')
        print('[pipeline] finished',stage,list(extra),flush=True)
    for seed in seeds:
        selected=output/'forecast/m5'/f'seed_{seed}'/'selected.json'
        announced=False
        while not selected.exists():
            if not announced:print(f'[pipeline] waiting for M5 checkpoint seed {seed}; GPU forecast remains separate',flush=True);announced=True
            checkpoints=list((output/'forecast/m5').glob('seed_*/lr_*/progress.pt'))
            if checkpoints and time.time()-max(p.stat().st_mtime for p in checkpoints)>args.idle_minutes*60:
                raise RuntimeError('Forecast checkpoint heartbeat stopped; resume forecast before continuing pricing')
            time.sleep(15)
        command('pricing',['--seeds',seed])
        # Partial runs are labeled as such; completed pricing units will be reused.
        command('charts')
    command('pricing',['--seeds',*seeds])
    command('charts')
    command('verify')
    print('[pipeline] all configured seeds completed; final tables/figures available',flush=True)


if __name__=='__main__':main()
