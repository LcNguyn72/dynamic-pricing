"""Full M5 training: bounded GPU memory, BF16 autocast, FP32 checkpoints/metrics."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from experiments.song_forecast import SongTransformer
from . import data
from .common import SEEDS,digest,read_json,write_json
from .forecast import _save,_tensor_shard,evaluate_validation,infer

BATCH=4096


def train_unit(output,seed,lr,device):
    dataset='m5';settings=dict(data.config()['song'],batch_size=BATCH,mixed_precision='bfloat16_train_fp32_validation')
    dest=Path(output)/'forecast/m5'/f'seed_{seed}'/f'lr_{lr:g}';dest.mkdir(parents=True,exist_ok=True)
    identity={'dataset':dataset,'seed':seed,'lr':lr,'settings':settings,
              'data':{s:digest(Path(output)/'data/m5'/f'{s}.json') for s in data.shards(output,dataset)},
              'trainer':digest(Path(__file__)),'model':digest(data.ROOT/'experiments/song_forecast.py')}
    if (dest/'run.json').exists():
        record=read_json(dest/'run.json')
        if record['identity']!=identity or digest(dest/'best.pt')!=record['checkpoint_sha256']:raise ValueError('M5 saved run identity/hash differs')
        return record
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SongTransformer().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);history=[];best=float('inf');best_epoch=0;first=1;elapsed=0.
    progress=dest/'progress.pt';use_amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
    if progress.exists():
        saved=torch.load(progress,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('M5 partial identity differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(device)
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];first=saved['epoch']+1
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
        if use_amp:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        elapsed=saved['elapsed_seconds']
    started=perf_counter();shards=data.shards(output,dataset)
    for epoch in range(first,settings['max_epochs']+1):
        model.train();sse=0.;n=0;epoch_started=perf_counter()
        for shard_index in rng.permutation(len(shards)):
            shard=shards[int(shard_index)]
            h,c,y=_tensor_shard(output,dataset,shard,'train',device)
            order=torch.as_tensor(rng.permutation(len(y)),device=device)
            for begin in range(0,len(y),BATCH):
                ix=order[begin:begin+BATCH];optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=use_amp):
                    loss=F.mse_loss(model(h[ix],c[ix]),y[ix])
                if not torch.isfinite(loss):raise ValueError('Non-finite M5 training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                sse+=float(loss.detach())*len(ix);n+=len(ix)
            del h,c,y,order
            print(f'[full M5] seed={seed} lr={lr:g} epoch={epoch} shard={shard} {perf_counter()-epoch_started:.1f}s',flush=True)
        metrics=evaluate_validation(model,output,dataset,device)
        history.append({'epoch':epoch,'train_mse_z':sse/n,**metrics,'epoch_seconds':perf_counter()-epoch_started})
        pd.DataFrame(history).to_csv(dest/'learning_curve.csv',index=False)
        if metrics['validation_rmse']<best:
            best=metrics['validation_rmse'];best_epoch=epoch
            _save(dest/'best.pt',{'state_dict':model.state_dict(),'identity':identity,'epoch':epoch,'validation':metrics})
        _save(progress,{'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
              'epoch':epoch,'history':history,'best':best,'best_epoch':best_epoch,'numpy_rng':rng.bit_generator.state,
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if use_amp else None,
              'elapsed_seconds':elapsed+perf_counter()-started})
        print(f'[full M5] seed={seed} epoch={epoch} train={sse/n:.6f} val_RMSE={best:.6f} best_epoch={best_epoch}',flush=True)
        if epoch>=settings['min_epochs'] and epoch-best_epoch>=settings['patience']:break
    record={'identity':identity,'best_epoch':best_epoch,'validation_rmse':best,'epochs_run':len(history),
            'budget_cap_reached':len(history)>=settings['max_epochs'],'elapsed_seconds':elapsed+perf_counter()-started,
            'checkpoint_sha256':digest(dest/'best.pt'),'checkpoint':str((dest/'best.pt').relative_to(output)),
            'fit_scope':'train_only','device':str(device),'mixed_precision_training':use_amp,'evaluation_precision':'float32'}
    write_json(dest/'run.json',record);return record


def train(output,dataset='m5',seeds=SEEDS,device='auto'):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
    output=Path(output);dest=output/'forecast/m5';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'selection.json'
    if path.exists():selection=read_json(path)
    else:
        candidates=[train_unit(output,42,lr,device) for lr in data.config()['song']['learning_rates']]
        winner=min(candidates,key=lambda r:r['validation_rmse'])
        selection={'learning_rate':winner['identity']['lr'],'criterion':'validation micro RMSE in raw units',
                   'selection_seed':42,'candidates':candidates,'test_used':False}
        write_json(path,selection)
    for seed in seeds:
        record=train_unit(output,seed,selection['learning_rate'],device)
        write_json(dest/f'seed_{seed}'/'selected.json',record)
        infer(output,'m5',seed,record,device)
