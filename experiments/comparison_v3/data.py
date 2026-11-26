"""Verified full-scope V2 features, stored as memory mapped per-store arrays."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd

from experiments.phase5_data import load_contract, iter_weekly, windows_from_weekly, verify_shard_test_keys
from .common import PREPARED, ROOT, digest, read_json, write_json


def config():return read_json(Path(__file__).with_name('protocol.json'))


def prepare(output,prepared=PREPARED,datasets=('uci','m5')):
    from experiments.song_forecast import feature_arrays
    output=Path(output);contract=load_contract(prepared)
    identity={'phase1':contract['phase1_sha256'],'protocol':contract['protocol_sha256'],
              'feature_code':digest(ROOT/'experiments/song_forecast.py'),
              'loader_code':digest(Path(__file__)),'fit_scope':'train_only','scope':'full_v2'}
    directory=output/'data';directory.mkdir(parents=True,exist_ok=True)
    for dataset in datasets:
        dest=directory/dataset;dest.mkdir(exist_ok=True)
        for shard,weekly,scalers in iter_weekly(prepared,dataset,contract):
            marker=dest/f'{shard}.json'
            if marker.exists():
                record=read_json(marker)
                if record['identity']!=identity:raise ValueError(f'Stale feature cache {marker}; use a new output directory')
                if all(digest(dest/name)==value for name,value in record['files'].items()):
                    print(f'[data] verified reuse {dataset}/{shard}',flush=True);continue
            started=perf_counter()
            windows=windows_from_weekly(weekly)
            verify_shard_test_keys(prepared,dataset,contract,shard,windows)
            windows=windows.loc[windows['observed_sales'].astype(bool)].reset_index(drop=True)
            write_json(dest/f'{shard}_scalers.json',scalers)
            weekly.to_csv(dest/f'{shard}_weekly.csv.gz',index=False,compression='gzip')
            record={'identity':identity,'dataset':dataset,'shard':shard,'splits':{}}
            for split in ('train','validation','test'):
                rows=windows.loc[windows['split'].eq(split)].reset_index(drop=True)
                prefix=dest/f'{shard}_{split}'
                metadata=rows[['sku','week','qty','price','qty_lag_1','price_lag_1']].copy()
                metadata['mean_4_context']=rows[[f'qty_lag_{i}' for i in range(1,5)]].mean(axis=1)
                metadata.to_csv(str(prefix)+'_rows.csv.gz',index=False,compression='gzip')
                arrays={}
                for begin in range(0,len(rows),32768):
                    part=rows.iloc[begin:begin+32768]
                    h,c=feature_arrays(part,scalers)
                    mean=np.array([scalers[str(s)]['qty_mean'] for s in part['sku']],dtype=np.float32)
                    std=np.array([scalers[str(s)]['qty_std'] for s in part['sku']],dtype=np.float32)
                    y=(part['qty'].to_numpy(dtype=np.float32)-mean)/std
                    data={'history':h,'current':c,'target':y,'mean':mean,'std':std}
                    if not arrays:
                        arrays={name:np.lib.format.open_memmap(str(prefix)+'_'+name+'.npy',mode='w+',
                                dtype=value.dtype,shape=(len(rows),*value.shape[1:])) for name,value in data.items()}
                    for name,value in data.items():arrays[name][begin:begin+len(part)]=value
                for value in arrays.values():value.flush()
                arrays.clear();record['splits'][split]=len(rows)
            files=sorted(dest.glob(f'{shard}_*'))
            record['files']={p.name:digest(p) for p in files if p.is_file()}
            record['seconds']=perf_counter()-started
            write_json(marker,record)
            print(f'[data] {dataset}/{shard}: {record["splits"]}, {record["seconds"]:.1f}s',flush=True)
    write_json(directory/'manifest.json',{'identity':identity,'datasets':list(datasets),
               'frozen_test_keys':{d:contract['phase1'][d]['test_keys'] for d in datasets},
               'evidence_status':config()['evidence_status']})


def shards(output,dataset):
    directory=Path(output)/'data'/dataset
    return sorted(p.stem for p in directory.glob('*.json') if not p.name.endswith('_scalers.json'))


def arrays(output,dataset,shard,split):
    prefix=Path(output)/'data'/dataset/f'{shard}_{split}'
    return {name:np.load(str(prefix)+'_'+name+'.npy',mmap_mode='r',allow_pickle=False)
            for name in ('history','current','target','mean','std')}


def rows(output,dataset,shard,split):
    return pd.read_csv(Path(output)/'data'/dataset/f'{shard}_{split}_rows.csv.gz',dtype={'sku':str},parse_dates=['week'])


def weekly(output,dataset,shard):
    return pd.read_csv(Path(output)/'data'/dataset/f'{shard}_weekly.csv.gz',dtype={'sku':str},parse_dates=['week'])


def _fit_arimax(args):
    sku,qty,price=args
    from final_system.pipeline.estimators import fit_series,ARIMAX,_limit_threads
    _limit_threads()
    state,status=fit_series(np.asarray(qty,dtype=float),np.asarray(price,dtype=float),ARIMAX)
    return sku,state,status


def fit_arimax(output,workers=6):
    output=Path(output);target=output/'arimax';target.mkdir(exist_ok=True)
    for shard in shards(output,'m5'):
        path=target/f'{shard}.json'
        identity={'data':digest(output/'data/m5'/f'{shard}.json'),
                  'estimator':digest(ROOT/'final_system/pipeline/estimators.py')}
        if path.exists():
            record=read_json(path)
            if record['identity']!=identity:raise ValueError('ARIMAX identity differs')
            print(f'[arimax] reuse {shard}',flush=True);continue
        panel=weekly(output,'m5',shard);train=panel.loc[panel['split'].eq('train')]
        tasks=[]
        for sku,g in train.groupby('sku',sort=True):
            q=g['qty'].where(g['observed_sales'].astype(bool)).to_numpy(dtype=float)
            tasks.append((str(sku),q,g['price'].to_numpy(dtype=float)))
        start=perf_counter();states={};statuses={}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for sku,state,status in pool.map(_fit_arimax,tasks,chunksize=24):
                states[sku]=state;statuses[sku]=status
        write_json(path,{'identity':identity,'states':states,'status':statuses,'seconds':perf_counter()-start,
                         'fit_scope':'train_only','fit_count':sum(v is not None for v in states.values())})
        print(f'[arimax] {shard}: {sum(v is not None for v in states.values())}/{len(states)} fitted, {perf_counter()-start:.1f}s',flush=True)


def baseline_predictions(output,dataset):
    from final_system.engine import _price_point
    from final_system.schemas import WeekObservation
    output=Path(output);dest=output/'forecast'/dataset/'baselines';dest.mkdir(parents=True,exist_ok=True)
    for shard in shards(output,dataset):
        states=read_json(output/'arimax'/f'{shard}.json')['states'] if dataset=='m5' else {}
        for split in ('validation','test'):
            frame=rows(output,dataset,shard,split);features=arrays(output,dataset,shard,split)
            frame['naive_last']=frame['qty_lag_1']
            frame['mean_4']=frame['mean_4_context']
            if dataset=='m5':
                available=frame['sku'].map({k:v is not None for k,v in states.items()}).to_numpy(dtype=bool)
                forecast=frame['naive_last'].to_numpy().copy()
                # Vectorize exactly the serving ARIMAX one-step equation per series.
                for sku,indexes in frame.groupby('sku',sort=False).indices.items():
                    state=states[sku]
                    if state is None:continue
                    ix=np.asarray(indexes);c,beta,phi,_=state['params']
                    qlag=frame.iloc[ix]['qty_lag_1'].to_numpy()
                    plag=frame.iloc[ix]['price_lag_1'].to_numpy()
                    p=frame.iloc[ix]['price'].to_numpy()
                    z=c+phi*((qlag-state['qty_mean'])/state['qty_sd']-beta*(plag-state['price_mean'])/state['price_sd'])+beta*(p-state['price_mean'])/state['price_sd']
                    forecast[ix]=np.maximum(0,state['qty_mean']+state['qty_sd']*z)
                frame['arimax_fallback']=forecast;frame['arimax_available']=available
            frame.to_csv(dest/f'{shard}_{split}.csv.gz',index=False,compression='gzip')
            print(f'[baseline] {dataset}/{shard}/{split}: {len(frame)} keys',flush=True)
