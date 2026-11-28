"""Matched-budget pricing with causal simulated feedback on every eligible M5 series."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import hashlib
import math

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from threadpoolctl import threadpool_limits

from . import data
from .common import ROOT,SEEDS,digest,read_json,write_json,roll_history
from .pricing_models import FastKernel,LagAdjustedKernelSimulator,ArimaxAdapter,SongAdapter,load_store


def dependency_identity(output):
    files=[Path(__file__),Path(__file__).with_name('pricing_models.py'),Path(__file__).with_name('common.py'),
           Path(__file__).with_name('data.py'),Path(__file__).with_name('protocol.json'),
           ROOT/'experiments/pricing_simulator.py',ROOT/'experiments/song_forecast.py',
           ROOT/'final_system/engine.py',ROOT/'final_system/feedback.py',ROOT/'final_system/ppo_policy.py']
    manifest=Path(output)/'data/manifest.json'
    return {'code':{str(p.relative_to(ROOT)):digest(p) for p in files},
            'arimax':{p.name:digest(p) for p in sorted((Path(output)/'arimax').glob('*.json'))},
            'prepared_manifest':digest(manifest) if manifest.exists() else None}


def is_full_result(rows,series_count):
    cfg=data.config()['pricing']
    expected={(sim,method,seed) for sim in cfg['simulators'] for method in cfg['policies'] for seed in SEEDS}
    keys=['simulator','method','seed']
    if rows.duplicated(keys).any():return False
    actual=set(rows[keys].itertuples(index=False,name=None))
    return actual==expected and bool(rows['series'].eq(series_count).all()) and bool(rows['decisions'].eq(series_count*cfg['horizon_weeks']).all())


class PricingEnvironment(gym.Env):
    metadata={}

    def __init__(self,series,simulator,adapter,kind,reward_scale,seed):
        self.series=series;self.keys=sorted(series);self.simulator=simulator;self.adapter=adapter;self.kind=kind
        self.reward_scale=reward_scale;self.seed_value=seed;self.seen=set();self.total_steps=0
        self.noise=np.random.default_rng(seed+6000007)
        count=31 if kind=='song' else 9+len(adapter.roster)
        self.action_space=gym.spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(count,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        sku=self.keys[int(self.np_random.integers(len(self.keys)))];self.item=self.series[sku];self.seen.add(sku)
        self.index=int(self.np_random.choice(self.item.train_starts));self.end=self.index+20
        self.q=self.item.qty[self.index-12:self.index][::-1].copy();self.p=self.item.prices[self.index-12:self.index][::-1].copy()
        self.adapter.reset(sku);surface=self.simulator.surfaces[sku]
        self.current=float(np.clip(self.p[0],surface.price_min,surface.price_max))
        return self.observation(),{}

    def observation(self):
        row=self.item.row(self.index,self.q,self.p)
        return self.adapter.observation(row,self.current,self.simulator.surfaces[self.item.sku])

    def step(self,action):
        surface=self.simulator.surfaces[self.item.sku]
        price=surface.price_min+(float(np.clip(action[0],-1,1))+1)*(surface.price_max-surface.price_min)/2
        row=self.item.row(self.index,self.q,self.p)
        qty=self.simulator.sample(self.item.sku,price,row['week'],self.noise,row)
        self.adapter.feedback(row,price,qty)
        self.q,self.p=roll_history(self.q,self.p,qty,price);self.current=price;self.index+=1;self.total_steps+=1
        done=self.index==self.end
        observation=np.zeros(self.observation_space.shape,dtype=np.float32) if done else self.observation()
        return observation,price*qty/self.reward_scale,done,False,{'revenue':price*qty}


def rollout(item,simulator,method,adapter,policy,seed,split='test',horizon=20,log=True):
    start=item.test_start if split=='test' else item.validation_start
    q=item.qty[start-12:start][::-1].copy();p=item.prices[start-12:start][::-1].copy()
    surface=simulator.surfaces[item.sku]
    current=float(np.clip(p[0],surface.price_min,surface.price_max))
    if adapter is not None:adapter.reset(item.sku)
    key_seed=int.from_bytes(hashlib.sha256(item.sku.encode()).digest()[:4],'little')
    noise=np.random.default_rng(np.random.SeedSequence([seed,key_seed,0 if split=='test' else 1]))
    records=[];total=0.
    for index in range(start,start+horizon):
        row=item.row(index,q,p)
        if method=='fixed_mean':price=float(np.mean(np.exp(surface.log_prices)))
        elif method=='fixed_upper':price=surface.price_max
        elif method=='arimax_grid':price=adapter.grid(row,current)
        else:
            obs=adapter.observation(row,current,surface)
            action=policy.predict(obs,deterministic=True)[0]
            price=surface.price_min+(float(np.clip(action[0],-1,1))+1)*(surface.price_max-surface.price_min)/2
        qty=simulator.sample(item.sku,price,row['week'],noise,row)
        revenue=price*qty;total+=revenue
        if not np.isfinite(revenue) or not surface.price_min-1e-10<=price<=surface.price_max+1e-10:
            raise ValueError('Invalid pricing decision')
        if log:records.append({'sku':item.sku,'week':str(pd.Timestamp(row['week']).date()),'method':method,'seed':seed,
            'price':price,'simulated_qty':qty,'revenue':revenue,'previous_qty':float(q[0]),'previous_price':float(p[0]),
            'train_price_min':surface.price_min,'train_price_max':surface.price_max})
        if adapter is not None:adapter.feedback(row,price,qty)
        q,p=roll_history(q,p,qty,price);current=price
    return total,records


def _load_all(output):
    series={};surfaces={};excluded=[]
    for store in data.shards(output,'m5'):
        items,sim,bad=load_store(output,store);series.update(items);excluded.extend(bad)
        if sim is not None:surfaces.update(sim.surfaces)
    sim=FastKernel();sim.surfaces=surfaces
    return series,sim,excluded


def _song_weights(output,seed):
    selected=read_json(Path(output)/'forecast/m5'/f'seed_{seed}'/'selected.json')
    path=Path(output)/selected['checkpoint']
    if digest(path)!=selected['checkpoint_sha256']:raise ValueError('Song pretrain hash differs')
    return torch.load(path,map_location='cpu',weights_only=False)['state_dict'],selected['checkpoint_sha256']


def _scalers(output):
    combined={}
    for store in data.shards(output,'m5'):combined.update(read_json(Path(output)/'data/m5'/f'{store}_scalers.json'))
    return combined


def _training_unit(output,seed,series,simulator,scalers,roster,kind):
    dest=Path(output)/'pricing'/f'seed_{seed}'/kind;dest.mkdir(parents=True,exist_ok=True)
    cfg=data.config()['pricing'];weights,pretrain_hash=_song_weights(output,seed)
    identity={'seed':seed,'kind':kind,'pricing':cfg,'roster_sha256':hashlib.sha256('\n'.join(roster).encode()).hexdigest(),
              'pretrain':pretrain_hash,'dependencies':dependency_identity(output)}
    if (dest/'run.json').exists():
        saved=read_json(dest/'run.json')
        if saved['identity']!=identity:raise ValueError('Pricing run identity changed')
        for name,hash_ in saved['files'].items():
            if digest(dest/name)!=hash_:raise ValueError('Pricing checkpoint changed')
        return saved
    torch.manual_seed(seed)
    states={k:series[k].state for k in roster}
    adapter=SongAdapter(weights,scalers,seed) if kind=='song' else ArimaxAdapter(states,roster)
    scale=max(1.,float(np.mean([np.mean(s.quantities*np.exp(s.log_prices)) for s in simulator.surfaces.values()])))
    env=PricingEnvironment(series,simulator,adapter,kind,scale,seed)
    architecture=[256,256,256] if kind=='song' else [32,32]
    model=PPO('MlpPolicy',env,seed=seed,device='cpu',learning_rate=3e-4,
              gamma=.99,gae_lambda=.95,clip_range=.2,batch_size=64,n_steps=256,n_epochs=6,
              ent_coef=.005 if kind=='arimax' else 0.,
              policy_kwargs={'net_arch':{'pi':architecture,'vf':architecture},
                             'activation_fn':torch.nn.ReLU if kind=='song' else torch.nn.Tanh})
    # Validation representatives are chosen by train volume, one per store/category.
    val_keys=[]
    grouped={}
    for key in roster:
        group=(key.rsplit('@',1)[1],key.split('_',1)[0])
        if group not in grouped or series[key].qty[:series[key].n_train].sum()>series[grouped[group]].qty[:series[grouped[group]].n_train].sum():grouped[group]=key
    val_keys=sorted(grouped.values());curve=[];best=-float('inf');best_step=0;started=perf_counter()

    def evaluate_checkpoint():
        nonlocal best,best_step
        values=[]
        snapshot=deepcopy(adapter.model.state_dict()) if kind=='song' else None
        for key in val_keys:
            agent=SongAdapter(snapshot,scalers,seed) if kind=='song' else ArimaxAdapter(states,roster)
            value,_=rollout(series[key],simulator,'song_ppo_feedback' if kind=='song' else 'arimax_ppo',agent,model,seed,split='validation',log=False)
            values.append(value)
        score=float(sum(values));curve.append({'timesteps':model.num_timesteps,'validation_revenue':score,
            'validation_series':len(val_keys),'train_series_seen':len(env.seen),'seconds':perf_counter()-started})
        pd.DataFrame(curve).to_csv(dest/'learning_curve.csv',index=False)
        if score>best:
            best=score;best_step=model.num_timesteps;model.save(dest/'policy')
            if kind=='song':torch.save({'state_dict':snapshot,'seed':seed,'pretrain_sha256':pretrain_hash},dest/'song.pt')
        print(f'[pricing train] {kind} seed={seed} steps={model.num_timesteps} validation_R={score:.2f} seen={len(env.seen)}/{len(roster)}',flush=True)

    class Checkpoints(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_end(self):
            if self.num_timesteps%2048==0:evaluate_checkpoint()

    model.learn(total_timesteps=cfg['matched_training_steps'],callback=Checkpoints())
    evaluate_checkpoint()
    files=['policy.zip']+(['song.pt'] if kind=='song' else [])
    record={'identity':identity,'selected_step':best_step,'validation_revenue':best,
            'train_series_seen':len(env.seen),'train_series_total':len(roster),
            'training_steps':model.num_timesteps,'reward_scale':scale,'seconds':perf_counter()-started,
            'validation_selection_scope':'one train-volume-ranked representative per store/category',
            'validation_keys':val_keys,'files':{name:digest(dest/name) for name in files},
            'budget_warning':'Fixed matched steps do not establish policy convergence; report roster coverage'}
    write_json(dest/'run.json',record);return record


def _evaluate_store(args):
    output,store,seed,roster=args;output=Path(output)
    torch.set_num_threads(1);threadpool_limits(limits=1)
    series,gaussian,excluded=load_store(output,store)
    if not series:return []
    lag=LagAdjustedKernelSimulator(.75,ratio_limit=2.);lag.surfaces=gaussian.surfaces
    states={k:v.state for k,v in series.items()};scalers=_scalers(output)
    policy_dirs={kind:output/'pricing'/f'seed_{seed}'/kind for kind in ('song','arimax')}
    policies={k:PPO.load(path/'policy.zip',device='cpu') for k,path in policy_dirs.items()}
    weights=torch.load(policy_dirs['song']/'song.pt',map_location='cpu',weights_only=False)['state_dict']
    adapters={'arimax_grid':ArimaxAdapter(states,roster),'arimax_ppo':ArimaxAdapter(states,roster),
              'song_ppo_feedback':SongAdapter(weights,scalers,seed)}
    totals=[];dest=output/'pricing'/f'seed_{seed}'/'decisions';dest.mkdir(exist_ok=True)
    for sim_name,sim in (('gaussian',gaussian),('lag_0.75',lag)):
        file=dest/f'{store}_{sim_name}.csv.gz';stamp=file.with_suffix('.json')
        identity={'policies':{k:digest(p/'run.json') for k,p in policy_dirs.items()},
                  'data':digest(output/'data/m5'/f'{store}.json'),'runner':digest(Path(__file__)),
                  'dependencies':dependency_identity(output)}
        if stamp.exists() and file.exists():
            saved=read_json(stamp)
            if saved['identity']!=identity:raise ValueError('Pricing evaluation identity differs')
            if saved['file']!=digest(file):raise ValueError('Pricing decisions changed')
            totals.extend(saved['totals']);continue
        parts=[];scores={k:0. for k in data.config()['pricing']['policies']};start=perf_counter()
        for index,(key,item) in enumerate(sorted(series.items())):
            for method in scores:
                adapter=adapters.get(method)
                if method.startswith('song'):adapter.reset_evaluation(weights,seed)
                policy=policies['song'] if method.startswith('song') else policies['arimax'] if method=='arimax_ppo' else None
                value,records=rollout(item,sim,method,adapter,policy,seed)
                scores[method]+=value;parts.extend(records)
            if (index+1)%200==0:print(f'[pricing test] {store} seed={seed} {sim_name}: {index+1}/{len(series)} series',flush=True)
        frame=pd.DataFrame(parts);frame['simulator']=sim_name
        if frame.duplicated(['sku','week','method','seed']).any():raise ValueError('Duplicate pricing keys')
        frame.to_csv(file,index=False,compression='gzip')
        rows=[{'store':store,'seed':seed,'simulator':sim_name,'method':method,'revenue':score,
               'series':len(series),'decisions':len(series)*20,'seconds':perf_counter()-start} for method,score in scores.items()]
        totals.extend(rows);write_json(stamp,{'identity':identity,'file':digest(file),'totals':rows})
        print(f'[pricing done] {store} seed={seed} {sim_name}: {len(series)} series, {perf_counter()-start:.1f}s',flush=True)
    return totals


def run(output,seeds=SEEDS,workers=6):
    output=Path(output);torch.set_num_threads(1);threadpool_limits(limits=1)
    series,simulator,excluded=_load_all(output);roster=sorted(series)
    if not roster:raise ValueError('No valid full-M5 pricing series')
    dest=output/'pricing';dest.mkdir(exist_ok=True)
    write_json(dest/'scope.json',{'series':roster,'series_count':len(roster),'excluded_context':excluded,
        'fit_scope':'train_only','eligibility':'ARIMAX fit + variable-price kernel + 12-week pre-period history',
        'test_scope':'first 20 V2 test weeks for every eligible series','outcomes_used_for_selection':False})
    scalers=_scalers(output)
    for seed in seeds:
        for kind in ('arimax','song'):_training_unit(output,seed,series,simulator,scalers,roster,kind)
    jobs=[(str(output),store,seed,roster) for seed in seeds for store in data.shards(output,'m5')]
    records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_evaluate_store,jobs,chunksize=1):
            records.extend(result);pd.DataFrame(records).to_csv(dest/'store_runs_partial.csv',index=False)
    pd.DataFrame(records).to_csv(dest/'store_runs.csv',index=False)
    totals=pd.DataFrame(records).groupby(['simulator','method','seed'],as_index=False).agg(revenue=('revenue','sum'),decisions=('decisions','sum'),series=('series','sum'))
    totals.to_csv(dest/'runs.csv',index=False)
    write_json(dest/'complete.json',{'seeds':list(seeds),'series':len(roster),'decisions_per_seed_method':len(roster)*20,
               'runs_sha256':digest(dest/'runs.csv'),'scope_sha256':digest(dest/'scope.json'),
               'full_comparison_complete':is_full_result(totals,len(roster)),
               'evidence_status':data.config()['evidence_status']})
