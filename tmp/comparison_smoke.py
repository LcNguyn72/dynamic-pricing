"""Real SB3/feedback/reload smoke; isolated artifacts, never used as results."""
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO
from experiments.comparison_v3 import data,pricing
from experiments.comparison_v3.pricing_models import load_store,SongAdapter,ArimaxAdapter
from experiments.comparison_v3.common import DEFAULT_OUTPUT,read_json,write_json,digest

if __name__=='__main__':
    torch.set_num_threads(1);threadpool_limits(limits=1)
    source=DEFAULT_OUTPUT;dest=Path('tmp/comparison_smoke_artifacts');dest.mkdir(exist_ok=True)
    start=perf_counter();series,sim,_=load_store(source,'CA_1')
    roster=sorted(series)[:2];series={s:series[s] for s in roster};sim.surfaces={s:sim.surfaces[s] for s in roster}
    scalers=read_json(source/'data/m5/CA_1_scalers.json')
    checkpoint=source/'forecast/m5/seed_42/lr_0.001/best.pt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    settings=deepcopy(data.config());settings['pricing']['matched_training_steps']=256
    original_config=data.config;data.config=lambda:settings
    pricing._song_weights=lambda *_:(weights,digest(checkpoint))
    pricing.dependency_identity=lambda *_:{'smoke':True}
    result={}
    for kind in ('arimax','song'):
        record=pricing._training_unit(dest,42,series,sim,scalers,roster,kind)
        directory=dest/'pricing/seed_42'/kind
        policy=PPO.load(directory/'policy.zip',device='cpu')
        if kind=='song':
            fitted=torch.load(directory/'song.pt',map_location='cpu',weights_only=False)['state_dict']
            adapter=SongAdapter(fitted,scalers,42)
        else:adapter=ArimaxAdapter({k:v.state for k,v in series.items()},roster)
        revenue,rows=pricing.rollout(series[roster[0]],sim,'song_ppo_feedback' if kind=='song' else 'arimax_ppo',adapter,policy,42)
        assert len(rows)==20 and revenue>0
        result[kind]={'training_steps':record['training_steps'],'revenue':revenue,'decisions':len(rows)}
    result['seconds']=perf_counter()-start
    write_json(dest/'smoke.json',result);print(result,flush=True)
