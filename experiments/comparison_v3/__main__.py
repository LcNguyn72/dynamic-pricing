from __future__ import annotations

import argparse
from pathlib import Path
from .common import DEFAULT_OUTPUT,PREPARED,SEEDS,environment,write_json,read_json


def main():
    parser=argparse.ArgumentParser(description='Full-scope Song vs project comparison, resumable with raw evidence')
    parser.add_argument('stage',choices=('prepare','arimax','forecast','simulator-audit','pricing','charts','verify','all'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepared',type=Path,default=PREPARED)
    parser.add_argument('--datasets',nargs='+',choices=('uci','m5'),default=['uci','m5'])
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'environment_latest.json',environment())
    from . import data
    if args.stage in ('prepare','all'):data.prepare(args.output,args.prepared,args.datasets)
    if args.stage in ('arimax','all'):
        if 'm5' in args.datasets:data.fit_arimax(args.output,args.workers)
        for dataset in args.datasets:data.baseline_predictions(args.output,dataset)
    if args.stage in ('forecast','all'):
        for dataset in args.datasets:
            if dataset=='m5':
                from .forecast_full import train
            else:
                from .forecast import train
            train(args.output,dataset,args.seeds,args.device)
    if args.stage in ('simulator-audit','all'):
        from .simulator_audit import run
        run(args.output,args.workers)
    if args.stage in ('pricing','all'):
        from .pricing import run
        # Operational override for a long, already-running batch controller.
        # Per-store RNG and map order are independent of worker scheduling.
        execution=args.output/'execution.json'
        settings=read_json(execution) if execution.exists() else {}
        workers=settings.get('pricing_workers',args.workers)
        if type(workers) is not int or workers<1:raise ValueError('pricing_workers must be a positive integer')
        write_json(args.output/'execution_used_pricing.json',{'workers':workers,'cli_workers':args.workers,
                   'execution_override':settings,'scientific_configuration_changed':False})
        print(f'[pricing execution] workers={workers}',flush=True)
        run(args.output,args.seeds,workers)
    if args.stage in ('charts','all'):
        from .charts import build
        build(args.output)
    if args.stage in ('verify','all'):
        from .verify import run
        run(args.output,args.prepared)


if __name__=='__main__':main()
