from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import subprocess

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT=ROOT/'experiments/artifacts/comparison_v3'
PREPARED=ROOT/'final_system/runtime/prepared'
SEEDS=(42,123,456,789,1024)


def digest(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def read_json(path):return json.loads(Path(path).read_text('utf-8'))


def source_identity():
    files=[*Path(__file__).parent.glob('*.py'), ROOT/'experiments/song_forecast.py',
           ROOT/'experiments/pricing_simulator.py', ROOT/'final_system/engine.py',
           ROOT/'final_system/feedback.py', ROOT/'final_system/pipeline/estimators.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}


def environment():
    import importlib.metadata as meta
    packages={}
    for name in ('numpy','pandas','torch','scipy','statsmodels','stable-baselines3','gymnasium','matplotlib'):
        try:packages[name]=meta.version(name)
        except meta.PackageNotFoundError:packages[name]=None
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    return {'utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
            'platform':platform.platform(),'packages':packages,'git_head':git.stdout.strip(),
            'code_sha256':source_identity()}


def assert_same_keys(actual,expected):
    keys=['sku','week']
    a=actual[keys].copy();b=expected[keys].copy()
    for frame in (a,b):
        frame['sku']=frame['sku'].astype(str);frame['week']=pd.to_datetime(frame['week']).dt.strftime('%Y-%m-%d')
        if frame.duplicated(keys).any():raise ValueError('duplicate forecast keys')
    if not a.sort_values(keys).reset_index(drop=True).equals(b.sort_values(keys).reset_index(drop=True)):
        raise ValueError('forecast keys differ')


def forecast_metrics(actual,predicted):
    y=np.asarray(actual,dtype=float);p=np.asarray(predicted,dtype=float)
    if not len(y) or y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid forecast arrays')
    e=p-y;pos=y>0;den=np.square(y-y.mean()).sum()
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),
            'wape_pct':float(100*np.abs(e).sum()/y.sum()) if y.sum()>0 else None,
            'r2':float(1-np.square(e).sum()/den) if den>0 else None,'bias':float(e.mean()),
            'mape_positive_n':int(pos.sum()),
            'mape_positive_pct':float(100*np.mean(np.abs(e[pos])/y[pos])) if pos.any() else None}


def paired_interval(rows,method,reference,draws=10000,seed=20261126):
    selected=rows.loc[rows['method'].isin((method,reference)),['method','seed','revenue']]
    if selected.duplicated(['method','seed']).any():raise ValueError('duplicate method/seed')
    a=selected.loc[selected['method'].eq(method)].set_index('seed')['revenue'].sort_index()
    b=selected.loc[selected['method'].eq(reference)].set_index('seed')['revenue'].sort_index()
    if not len(a) or not a.index.equals(b.index):raise ValueError('paired seed sets differ')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b<=0):raise ValueError('invalid paired revenue')
    uplift=100*(a.to_numpy()/b.to_numpy()-1)
    rng=np.random.default_rng(seed)
    bootstrap=uplift[rng.integers(0,len(a),size=(draws,len(a)))].mean(axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return {'method':method,'reference':reference,'pairs':len(a),'mean_uplift_pct':float(uplift.mean()),
            'ci_low_pct':float(lo),'ci_high_pct':float(hi),'wins':int((uplift>0).sum()),
            'losses':int((uplift<0).sum()),'ties':int((uplift==0).sum()),
            'uncertainty_scope':'paired training/noise seeds conditional on fixed dataset, cohort and simulator',
            'few_seed_warning':len(a)<10}


def choose_grid(prices,quantities,current):
    p=np.asarray(prices,dtype=float);q=np.asarray(quantities,dtype=float)
    if p.shape!=q.shape or not len(p) or not np.isfinite(p*q).all():raise ValueError('Invalid grid')
    revenues=p*q;tie=np.flatnonzero(revenues.max()-revenues<=1e-8)
    return float(min(p[tie],key=lambda v:(abs(v-current),v)))


def roll_history(qty,prices,generated_qty,chosen_price):
    return np.r_[float(generated_qty),np.asarray(qty)[:-1]],np.r_[float(chosen_price),np.asarray(prices)[:-1]]


def train_scalers(weekly):
    rows=weekly.loc[weekly['split'].eq('train') & weekly['observed_sales'].astype(bool)]
    out={}
    for sku,g in rows.groupby('sku',sort=True):
        record={}
        for col in ('qty','price'):
            mean=float(g[col].mean());sd=float(g[col].std(ddof=0))
            record[col+'_mean']=mean
            record[col+'_std']=sd if sd>max(1e-8,abs(mean)*1e-8) else 1.
        out[str(sku)]=record
    return out
