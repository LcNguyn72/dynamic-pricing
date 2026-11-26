"""Final evidence gate: keys, hashes, complete seed matrix and raw revenue sums."""
from pathlib import Path
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

from . import data
from .common import ROOT,PREPARED,SEEDS,read_json,write_json,digest,assert_same_keys,environment


def check_prediction_values(frame,columns):
    for column in columns:
        if column not in frame or not np.isfinite(frame[column]).all() or (frame[column]<0).any():
            raise ValueError(f'Invalid demand prediction column: {column}')


def check_forecast_identity(record,checkpoint,dataset,seed,expected_data):
    identity=record['identity']
    if (identity['dataset']!=dataset or identity['seed']!=seed or identity['data']!=expected_data
            or checkpoint['identity']!=identity or record['fit_scope']!='train_only'):
        raise ValueError('Forecast checkpoint/seed/data identity differs')


def check_decision_frame(frame,expected,seed,simulator):
    methods=set(data.config()['pricing']['policies'])
    if not frame['seed'].eq(seed).all() or not frame['simulator'].eq(simulator).all():
        raise ValueError('Decision seed/simulator identity differs')
    if set(frame['method'])!=methods:raise ValueError('Decision methods differ')
    if frame.duplicated(['sku','week','method','seed']).any():raise ValueError('Duplicate decision key')
    for _,part in frame.groupby('method'):assert_same_keys(part,expected)
    joined=frame.assign(week=pd.to_datetime(frame['week'])).merge(expected,on=['sku','week'],validate='many_to_one',suffixes=('','_expected'))
    for column in ('train_price_min','train_price_max'):
        if not np.allclose(joined[column],joined[column+'_expected'],rtol=1e-10,atol=1e-10):
            raise ValueError('Decision bounds differ from train support')
    numeric=['price','simulated_qty','revenue','previous_qty','previous_price']
    if not np.isfinite(frame[numeric]).all().all() or (frame[['price','simulated_qty','revenue']]<0).any().any():
        raise ValueError('Nonfinite or negative decision value')
    if (frame['price']<frame['train_price_min']-1e-8).any() or (frame['price']>frame['train_price_max']+1e-8).any():
        raise ValueError('Out-of-support decision')
    if not np.allclose(frame['revenue'],frame['price']*frame['simulated_qty'],rtol=1e-10,atol=1e-8):
        raise ValueError('Revenue is not price times units')


def run(output,prepared=PREPARED):
    from .pricing import is_full_result,dependency_identity
    import torch
    output=Path(output);(output/'verification.json').unlink(missing_ok=True)
    manifest=read_json(output/'data/manifest.json');checked=0
    forecast_keys={}
    for dataset in ('uci','m5'):
        selected_records={}
        expected_data={s:digest(output/'data'/dataset/f'{s}.json') for s in data.shards(output,dataset)}
        for seed in SEEDS:
            selected=read_json(output/'forecast'/dataset/f'seed_{seed}'/'selected.json')
            path=output/selected['checkpoint']
            if digest(path)!=selected['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            check_forecast_identity(selected,checkpoint,dataset,seed,expected_data)
            selected_records[seed]=selected
        keys=[]
        for shard in data.shards(output,dataset):
            directory=output/'data'/dataset;record=read_json(directory/f'{shard}.json')
            for name,expected in record['files'].items():
                if digest(directory/name)!=expected:raise ValueError(f'Feature artifact changed: {name}')
                checked+=1
            reference=data.rows(output,dataset,shard,'test');keys.append(reference[['sku','week']])
            baseline=pd.read_csv(output/'forecast'/dataset/'baselines'/f'{shard}_test.csv.gz',dtype={'sku':str},parse_dates=['week'])
            assert_same_keys(baseline,reference)
            if not np.array_equal(baseline['qty'].to_numpy(),reference['qty'].to_numpy()):raise ValueError('Baseline labels differ')
            check_prediction_values(baseline,['naive_last','mean_4']+(['arimax_fallback'] if dataset=='m5' else []))
            for seed in SEEDS:
                selected=selected_records[seed]
                file=output/'forecast'/dataset/f'seed_{seed}'/'predictions'/f'{shard}_test.csv.gz'
                stamp=read_json(file.with_suffix('.json'))
                if stamp['file']!=digest(file) or stamp['checkpoint']!=selected['checkpoint_sha256']:raise ValueError('Prediction/model hash mismatch')
                frame=pd.read_csv(file,dtype={'sku':str},parse_dates=['week'])
                assert_same_keys(frame,reference)
                if not np.array_equal(frame['qty'].to_numpy(),reference['qty'].to_numpy()):raise ValueError('Neural labels differ')
                if not frame['seed'].eq(seed).all() or not frame['method'].eq('song_transformer').all():raise ValueError('Forecast row identity differs')
                check_prediction_values(frame,['prediction'])
        combined=pd.concat(keys,ignore_index=True)
        frozen=pd.read_csv(Path(prepared)/f'test_keys_{dataset}.csv',dtype={'sku':str},parse_dates=['week'])
        assert_same_keys(combined,frozen)
        if len(combined)!=manifest['frozen_test_keys'][dataset]['count']:raise ValueError('Incomplete global test scope')
        forecast_keys[dataset]=len(combined)
        print(f'[verify] {dataset}: all {len(combined):,} frozen keys, five checkpoint/prediction pairs',flush=True)
    scope=read_json(output/'pricing/scope.json');runs=pd.read_csv(output/'pricing/runs.csv')
    if len(scope['series'])!=scope['series_count'] or len(set(scope['series']))!=scope['series_count']:
        raise ValueError('Pricing roster count/uniqueness differs')
    if not is_full_result(runs,scope['series_count']):raise ValueError('Pricing matrix is incomplete')
    totals=[];decisions=0;dependencies=dependency_identity(output);expected_by_store={}
    for store in data.shards(output,'m5'):
        panel=data.weekly(output,'m5',store)
        panel=panel.loc[panel['sku'].isin(scope['series'])]
        expected=panel.loc[panel['split'].eq('test')].sort_values(['sku','week']).groupby('sku',sort=False).head(data.config()['pricing']['horizon_weeks'])[['sku','week']]
        train=panel.loc[panel['split'].eq('train')&panel['observed_sales'].astype(bool)]
        bounds=train.groupby('sku')['price'].agg(train_price_min='min',train_price_max='max').reset_index()
        expected=expected.merge(bounds,on='sku',validate='many_to_one')
        expected_by_store[store]=expected
    if sum(len(v) for v in expected_by_store.values())!=scope['series_count']*data.config()['pricing']['horizon_weeks']:
        raise ValueError('Expected pricing roster/calendar is incomplete')
    for seed in SEEDS:
        for kind in ('arimax','song'):
            directory=output/'pricing'/f'seed_{seed}'/kind;record=read_json(directory/'run.json')
            if record['identity']['dependencies']!=dependencies:raise ValueError('Pricing source/data dependencies changed')
            identity=record['identity']
            roster_hash=hashlib.sha256('\n'.join(sorted(scope['series'])).encode()).hexdigest()
            if (identity['seed']!=seed or identity['kind']!=kind or identity['roster_sha256']!=roster_hash
                    or identity['pricing']!=data.config()['pricing'] or identity['pretrain']!=selected_records[seed]['checkpoint_sha256']):
                raise ValueError('Pricing policy seed/roster/pretrain identity differs')
            for name,expected in record['files'].items():
                if digest(directory/name)!=expected:raise ValueError('Pricing policy or paired forecaster changed')
        directory=output/'pricing'/f'seed_{seed}'/'decisions'
        expected_files={f'{store}_{sim}.csv.gz':(store,sim) for store in expected_by_store for sim in data.config()['pricing']['simulators']}
        files=sorted(directory.glob('*.csv.gz'))
        if {p.name for p in files}!=set(expected_files):raise ValueError('Pricing store/simulator files missing')
        for file in files:
            store,simulator=expected_files[file.name]
            stamp=read_json(file.with_suffix('.json'))
            if stamp['file']!=digest(file):raise ValueError('Decision file changed')
            identity={'policies':{kind:digest(output/'pricing'/f'seed_{seed}'/kind/'run.json') for kind in ('song','arimax')},
                      'data':digest(output/'data/m5'/f'{store}.json'),'runner':digest(Path(__file__).with_name('pricing.py')),
                      'dependencies':dependencies}
            if stamp['identity']!=identity:raise ValueError('Decision policy/data identity differs')
            frame=pd.read_csv(file,dtype={'sku':str})
            check_decision_frame(frame,expected_by_store[store],seed,simulator)
            totals.append(frame.groupby(['simulator','method','seed'],as_index=False).agg(revenue=('revenue','sum'),decisions=('revenue','size')))
            decisions+=len(frame)
        print(f'[verify] pricing seed {seed}: raw decisions and model hashes checked',flush=True)
    recomputed=pd.concat(totals,ignore_index=True).groupby(['simulator','method','seed'],as_index=False)[['revenue','decisions']].sum()
    match=runs.merge(recomputed,on=['simulator','method','seed'],validate='one_to_one',suffixes=('_reported','_raw'))
    if len(match)!=len(runs) or not np.allclose(match['revenue_reported'],match['revenue_raw'],rtol=1e-10,atol=1e-5):raise ValueError('Pricing totals differ from raw decisions')
    if not match['decisions_reported'].eq(match['decisions_raw']).all():raise ValueError('Decision count differs from raw files')
    tests=subprocess.run([sys.executable,'-m','unittest','experiments.test_comparison_v3','-q'],cwd=ROOT,capture_output=True,text=True)
    (output/'contract_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode:raise RuntimeError('Scientific contract tests failed')
    count=re.search(r'Ran (\d+) tests',tests.stdout+tests.stderr)
    snapshot=output/'source_snapshot';copied={}
    source_files=[]
    for directory in (ROOT/'experiments',ROOT/'experiments/comparison_v3',ROOT/'final_system',ROOT/'final_system/pipeline'):
        source_files.extend(directory.glob('*.py'))
    source_files.extend((ROOT/'final_system/configs').glob('*.json'))
    source_files.extend([ROOT/'experiments/protocol_v2.json',ROOT/'experiments/phase5_settings_v2.json',
                         Path(__file__).with_name('protocol.json'),Path(__file__).with_name('README.md'),
                         Path(__file__).with_name('METHOD.md'),
                         Path(__file__).with_name('requirements.txt'),Path(__file__).with_name('run.ps1')])
    for source in sorted(set(source_files)):
        relative=source.relative_to(ROOT);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);copied[str(relative)]=digest(source)
    write_json(output/'source_manifest.json',copied)
    result={'complete':True,'forecast_test_keys':forecast_keys,'forecast_seeds':list(SEEDS),
            'pricing_series':scope['series_count'],'pricing_raw_decisions':decisions,'raw_revenue_totals_match':True,
            'feature_files_hashed':checked,'contract_tests':int(count.group(1)) if count else None,
            'environment':environment(),'evidence_status':data.config()['evidence_status'],
            'inference':'Comparison against project Song reimplementation; conditional simulator results, not causal business uplift'}
    hardware={'processor':platform.processor(),'logical_cpus':os.cpu_count(),
              'torch_cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
              'cuda_available_at_verification':torch.cuda.is_available()}
    if torch.cuda.is_available():
        device=torch.cuda.get_device_properties(0)
        hardware.update(gpu=device.name,gpu_total_bytes=device.total_memory)
    try:
        import psutil
        hardware.update(physical_cpus=psutil.cpu_count(logical=False),ram_total_bytes=psutil.virtual_memory().total)
    except ImportError:pass
    result['hardware']=hardware
    write_json(output/'verification.json',result)
    print('[verify] full comparison evidence verified',flush=True)
