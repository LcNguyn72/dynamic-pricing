"""Train the fixed Song architecture with validation selection and saved curves."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiments.song_forecast import SongTransformer
from . import data
from .common import SEEDS
from .common import digest,forecast_metrics,read_json,write_json,source_identity


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


def _tensor_shard(output,dataset,shard,split,device):
    values=data.arrays(output,dataset,shard,split)
    return tuple(torch.as_tensor(np.array(values[name],copy=True),device=device)
                 for name in ('history','current','target'))


def predict(model,h,c,device,batch=1024):
    output=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(h),batch):
            history=torch.as_tensor(np.array(h[start:start+batch],copy=True),device=device)
            current=torch.as_tensor(np.array(c[start:start+batch],copy=True),device=device)
            value=model(history,current).detach().cpu().numpy()
            if not np.isfinite(value).all():raise ValueError('Non-finite Song predictions')
            output.append(value)
    return np.concatenate(output) if output else np.empty(0)


def evaluate_validation(model,output,dataset,device):
    sums=0.;count=0;z_sums=0.
    for shard in data.shards(output,dataset):
        values=data.arrays(output,dataset,shard,'validation')
        z=predict(model,values['history'],values['current'],device)
        y=values['target'].astype(np.float64)
        actual=y*values['std']+values['mean']
        forecast=np.maximum(0,z.astype(np.float64)*values['std']+values['mean'])
        sums+=float(np.square(forecast-actual).sum());count+=len(y)
        z_sums+=float(np.square(z-y).sum())
    return {'validation_rmse':math.sqrt(sums/count),'validation_mse_z':z_sums/count,'validation_n':count}


def train_unit(output,dataset,seed,lr,device):
    settings=data.config()['song'];dest=Path(output)/'forecast'/dataset/f'seed_{seed}'/f'lr_{lr:g}'
    dest.mkdir(parents=True,exist_ok=True)
    identity={'dataset':dataset,'seed':seed,'lr':lr,'settings':settings,
              'data':{s:digest(Path(output)/'data'/dataset/f'{s}.json') for s in data.shards(output,dataset)},
              'trainer':digest(Path(__file__)),'model':digest(Path(data.ROOT)/'experiments/song_forecast.py')}
    complete=dest/'run.json';checkpoint=dest/'progress.pt'
    if complete.exists():
        record=read_json(complete)
        if record['identity']!=identity:raise ValueError(f'Run identity changed at {dest}')
        if digest(dest/'best.pt')!=record['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        return record
    torch.manual_seed(seed);np_rng=np.random.default_rng(seed)
    model=SongTransformer().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    history=[];best=float('inf');best_epoch=0;start_epoch=1;elapsed=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Partial checkpoint identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(device)
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];start_epoch=saved['epoch']+1
        np_rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        elapsed=saved['elapsed_seconds']
    started=perf_counter();shards=data.shards(output,dataset)
    for epoch in range(start_epoch,settings['max_epochs']+1):
        model.train();squared=0.;count=0;epoch_start=perf_counter()
        for shard_index in np_rng.permutation(len(shards)):
            shard=shards[int(shard_index)]
            h,c,y=_tensor_shard(output,dataset,shard,'train',device)
            permutation=torch.as_tensor(np_rng.permutation(len(y)),device=device)
            for start in range(0,len(y),settings['batch_size']):
                ix=permutation[start:start+settings['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(model(h[ix],c[ix]),y[ix])
                if not torch.isfinite(loss):raise ValueError(f'Non-finite loss {dataset}/{seed}/{epoch}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);optimizer.step()
                squared+=float(loss.detach())*len(ix);count+=len(ix)
            del h,c,y,permutation
        metrics=evaluate_validation(model,output,dataset,device)
        history.append({'epoch':epoch,'train_mse_z':squared/count,**metrics,'epoch_seconds':perf_counter()-epoch_start})
        pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False)
        value=metrics['validation_rmse']
        if value<best:
            best=value;best_epoch=epoch
            _save(dest/'best.pt',{'state_dict':model.state_dict(),'identity':identity,'epoch':epoch,'validation':metrics})
        _save(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'identity':identity,
              'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch,
              'numpy_rng':np_rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
              'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
              'elapsed_seconds':elapsed+perf_counter()-started})
        print(f'[song] {dataset} seed={seed} lr={lr:g} epoch={epoch} train={squared/count:.5f} val_RMSE={value:.5f} best={best_epoch} {perf_counter()-epoch_start:.1f}s',flush=True)
        if epoch>=settings['min_epochs'] and epoch-best_epoch>=settings['patience']:break
    record={'identity':identity,'best_epoch':best_epoch,'validation_rmse':best,'epochs_run':len(history),
            'budget_cap_reached':len(history)>=settings['max_epochs'],
            'elapsed_seconds':elapsed+perf_counter()-started,'checkpoint_sha256':digest(dest/'best.pt'),
            'checkpoint':str((dest/'best.pt').relative_to(output)),'fit_scope':'train_only',
            'device':str(device),'tf32_enabled':bool(torch.backends.cuda.matmul.allow_tf32)}
    write_json(complete,record);return record


def train(output,dataset,seeds=SEEDS,device='auto'):
    torch.set_num_threads(2)
    device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
    if device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=True
    output=Path(output);dest=output/'forecast'/dataset;dest.mkdir(parents=True,exist_ok=True)
    selector=dest/'selection.json'
    if selector.exists():selection=read_json(selector)
    else:
        candidates=[train_unit(output,dataset,42,lr,device) for lr in data.config()['song']['learning_rates']]
        best=min(candidates,key=lambda r:r['validation_rmse'])
        selection={'learning_rate':best['identity']['lr'],'criterion':'validation micro RMSE in raw units',
                   'selection_seed':42,'candidates':candidates,'test_used':False}
        write_json(selector,selection)
    for seed in seeds:
        record=train_unit(output,dataset,seed,selection['learning_rate'],device)
        path=dest/f'seed_{seed}'/'selected.json';write_json(path,record)
        infer(output,dataset,seed,record,device)


def infer(output,dataset,seed,record,device):
    output=Path(output);dest=output/'forecast'/dataset/f'seed_{seed}'/'predictions';dest.mkdir(exist_ok=True)
    checkpoint=output/record['checkpoint'];model=SongTransformer().to(device)
    saved=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(saved['state_dict'])
    for split in ('validation','test'):
        for shard in data.shards(output,dataset):
            target=dest/f'{shard}_{split}.csv.gz'
            if target.exists() and target.with_suffix('.json').exists():
                stamp=read_json(target.with_suffix('.json'))
                if stamp['checkpoint']==record['checkpoint_sha256'] and stamp['file']==digest(target):continue
            values=data.arrays(output,dataset,shard,split)
            z=predict(model,values['history'],values['current'],device)
            forecast=np.maximum(0,z.astype(np.float64)*values['std']+values['mean'])
            frame=data.rows(output,dataset,shard,split);frame['prediction']=forecast
            frame['method']='song_transformer';frame['seed']=seed
            frame.to_csv(target,index=False,compression='gzip')
            write_json(target.with_suffix('.json'),{'checkpoint':record['checkpoint_sha256'],'file':digest(target),
                       'metrics':forecast_metrics(frame['qty'],forecast)})
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
