"""Full-scope observed-price validation diagnostics; never a causal validation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .pricing_models import load_store,LagAdjustedKernelSimulator
from . import data
from .common import write_json


def _store(args):
    output,store=args;threadpool_limits(limits=1)
    series,gaussian,excluded=load_store(output,store)
    if not series:return []
    lag=LagAdjustedKernelSimulator(.75,ratio_limit=2.);lag.surfaces=gaussian.surfaces
    stats={};started=perf_counter()
    for key,item in series.items():
        surface=gaussian.surfaces[key]
        for i in range(item.validation_start,item.test_start):
            if not np.isfinite(item.prices[i]) or item.prices[i]<=0:continue
            context={'qty_lag_1':item.qty[i-1],'price_lag_1':item.prices[i-1],
                     'event_days':item.event[i],'snap_days':item.snap[i]}
            price=item.prices[i];actual=item.qty[i]
            scopes=['all']+(['in_train_support'] if surface.price_min<=price<=surface.price_max else [])
            for name,pred in [('last_week',item.qty[i-1]),('gaussian',gaussian.predict_mean(key,price,item.dates[i],context)),
                              ('lag_0.75',lag.predict_mean(key,price,item.dates[i],context))]:
                for scope in scopes:
                    row=stats.setdefault((name,scope),{'n':0,'sse':0.,'sae':0.})
                    row['n']+=1;row['sse']+=(pred-actual)**2;row['sae']+=abs(pred-actual)
    print(f'[simulator audit] {store}: {len(series)} pricing-eligible series, {perf_counter()-started:.1f}s',flush=True)
    return [{'store':store,'method':name,'scope':scope,**v} for (name,scope),v in stats.items()]


def run(output,workers=6):
    output=Path(output);dest=output/'simulator_audit';dest.mkdir(exist_ok=True)
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(_store,[(str(output),s) for s in data.shards(output,'m5')]):records.extend(rows)
    frame=pd.DataFrame(records);frame['rmse']=np.sqrt(frame['sse']/frame['n']);frame['mae']=frame['sae']/frame['n']
    frame.to_csv(dest/'by_store.csv',index=False)
    full=frame.groupby(['method','scope'],as_index=False)[['n','sse','sae']].sum()
    full['rmse']=np.sqrt(full['sse']/full['n']);full['mae']=full['sae']/full['n']
    baseline=float(full.loc[full['method'].eq('last_week')&full['scope'].eq('in_train_support'),'rmse'].iloc[0])
    full['gate_rmse_vs_last_week']=np.where(full['scope'].eq('in_train_support'),full['rmse']<=baseline,False)
    full.to_csv(dest/'metrics.csv',index=False)
    write_json(dest/'manifest.json',{'scope':'all eligible M5 series; observed validation prices only',
               'selection_uses_test':False,'does_not_validate_counterfactual_demand':True,
               'gate':'in-train-price-support RMSE <= matched previous-week RMSE',
               'pricing_comparisons_remain_conditional_on_simulator':True})
