"""Serving-equivalent ARIMAX, paper-aligned Song and a verified faster kernel fit."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiments.pricing_simulator import KernelDemandSimulator,LagAdjustedKernelSimulator,ProductSurface,_silverman
from experiments.song_forecast import SongTransformer,_season
from final_system.engine import _price_point
from final_system.feedback import update_arimax
from .common import read_json,roll_history,choose_grid
from . import data


class FastKernel(KernelDemandSimulator):
    def fit(self,weekly):
        rows=weekly.loc[weekly['split'].eq('train') & weekly['observed_sales'].astype(bool)]
        surfaces={}
        for sku,g in rows.groupby('sku',sort=True):
            if len(g)<12 or g['price'].nunique()<2:continue
            p=g['price'].to_numpy(dtype=float);q=g['qty'].to_numpy(dtype=float);lp=np.log(p)
            w=g['week'].dt.isocalendar().week.to_numpy(dtype=int)
            e=g['event_days'].to_numpy(dtype=float);s=g['snap_days'].to_numpy(dtype=float)
            band=_silverman(lp);distance=np.abs(w[:,None]-w[None,:]);distance=np.minimum(distance,52-distance)
            weights=np.exp(-.5*((lp[:,None]-lp[None,:])/band)**2)
            weights*=.15+.85*np.exp(-.5*(distance/self.seasonal_bandwidth_weeks)**2)
            weights*=.2+.8*np.exp(-.5*((e[:,None]-e[None,:])/2.)**2)
            weights*=.2+.8*np.exp(-.5*((s[:,None]-s[None,:])/2.)**2)
            np.fill_diagonal(weights,0.);total=weights.sum(axis=1)
            fitted=np.divide(weights@q,total,out=np.zeros(len(q)),where=total>=1e-10)
            base=ProductSurface(lp,q,w,e,s,float(p.min()),float(p.max()),band,0.,len(np.unique(p)))
            for i in np.flatnonzero(total<1e-10):
                fitted[i]=self._predict(base,p[i],w[i],{'event_days':e[i],'snap_days':s[i]},exclude_index=int(i))
            noise=float(np.clip(np.std(np.log1p(q)-np.log1p(fitted),ddof=1),.05,1.25))
            surfaces[str(sku)]=ProductSurface(lp,q,w,e,s,float(p.min()),float(p.max()),band,noise,len(np.unique(p)))
        if not surfaces:raise ValueError('No kernel-eligible training series')
        self.surfaces=surfaces;return self


@dataclass
class Series:
    sku:str
    dates:tuple
    qty:np.ndarray
    prices:np.ndarray
    event:np.ndarray
    snap:np.ndarray
    n_train:int
    validation_start:int
    test_start:int
    state:dict
    train_starts:np.ndarray

    def row(self,index,q,p):
        row={'sku':self.sku,'week':self.dates[index],'event_days':float(self.event[index]),'snap_days':float(self.snap[index])}
        for j in range(1,13):
            row[f'qty_lag_{j}']=float(q[j-1]);row[f'price_lag_{j}']=float(p[j-1])
            row[f'event_days_lag_{j}']=float(self.event[index-j]);row[f'snap_days_lag_{j}']=float(self.snap[index-j])
        return row


def load_store(output,store):
    panel=data.weekly(output,'m5',store)
    states=read_json(Path(output)/'arimax'/f'{store}.json')['states']
    panel=panel.loc[panel['sku'].isin([s for s,v in states.items() if v is not None])].copy()
    if panel.empty:return {},None,[]
    simulator=FastKernel().fit(panel)
    series={};excluded=[]
    for sku,g in panel.groupby('sku',sort=True):
        g=g.sort_values('week');dates=tuple(pd.to_datetime(g['week']))
        qty=g['qty'].to_numpy(dtype=float);prices=g['price'].to_numpy(dtype=float)
        n_train=int(g['split'].eq('train').sum());val=n_train;test=int(g['split'].ne('test').sum())
        starts=np.array([i for i in range(12,n_train-20+1) if np.isfinite(prices[i-12:i]).all() and (prices[i-12:i]>0).all()],dtype=int)
        reason=None
        if sku not in simulator.surfaces:reason='no_kernel_surface'
        elif not len(starts):reason='no_train_episode_with_12_past_weeks'
        elif not np.isfinite(prices[test-12:test]).all():reason='missing_pretest_history'
        elif not np.isfinite(prices[val-12:val]).all():reason='missing_prevalidation_history'
        if reason:excluded.append({'sku':sku,'reason':reason});continue
        series[sku]=Series(sku,dates,qty,prices,g['event_days'].to_numpy(dtype=float),g['snap_days'].to_numpy(dtype=float),
                           n_train,val,test,states[sku],starts)
    simulator.surfaces={sku:simulator.surfaces[sku] for sku in series}
    return series,simulator,excluded


@lru_cache(maxsize=512)
def season(week):
    dates=pd.Series(pd.date_range(end=pd.Timestamp(week),periods=13,freq='7D'))
    return np.column_stack(_season(dates)).astype(np.float32)


class ArimaxAdapter:
    def __init__(self,states,roster):
        self.states=states;self.roster=tuple(roster);self.roster_index={s:i for i,s in enumerate(roster)}

    def reset(self,sku):self.sku=sku;self.base=self.states[sku];self.state=self.base;self.samples=[]

    def predict(self,row,price):
        return _price_point(self.state,SimpleNamespace(qty=row['qty_lag_1'],price=row['price_lag_1']),float(price))['predicted_qty']

    def grid(self,row,current):
        low,high=self.base['price_min'],self.base['price_max']
        prices=np.unique(np.r_[np.linspace(low,high,41),current])
        st=self.state;c,beta,phi,_=st['params']
        q=(row['qty_lag_1']-st['qty_mean'])/st['qty_sd']
        p=(row['price_lag_1']-st['price_mean'])/st['price_sd']
        quantity=np.maximum(0,st['qty_mean']+st['qty_sd']*(c+phi*(q-beta*p)+beta*(prices-st['price_mean'])/st['price_sd']))
        return choose_grid(prices,quantity,current)

    def observation(self,row,current,surface):
        st=self.state;c,beta,phi,_=st['params'];lo,hi=st['price_min'],st['price_max']
        q=(row['qty_lag_1']-st['qty_mean'])/st['qty_sd'];p=(row['price_lag_1']-st['price_mean'])/st['price_sd']
        def relative(price):return 2*max(-3.5,min(4.5,(price-lo)/(hi-lo)))-1
        base=c+phi*(q-beta*p);phase=2*np.pi*(pd.Timestamp(row['week']).isocalendar().week-1)/52
        out=np.zeros(9+len(self.roster),dtype=np.float32)
        out[:9]=np.clip([q,relative(row['price_lag_1']),relative(current),
                        base+beta*(lo-st['price_mean'])/st['price_sd'],base+beta*(hi-st['price_mean'])/st['price_sd'],
                        beta,math.log(hi/lo),math.sin(phase),math.cos(phase)],-8,8)
        out[9+self.roster_index[self.sku]]=1
        return out

    def feedback(self,row,price,qty):
        self.samples.append(SimpleNamespace(week=pd.Timestamp(row['week']).date(),price=price,qty=qty,
                            previous_qty=row['qty_lag_1'],previous_price=row['price_lag_1']))
        self.state,_=update_arimax(self.base,self.samples)


class SongAdapter:
    def __init__(self,weights,scalers,seed=42):
        self.model=SongTransformer().cpu().eval();self.model.load_state_dict(weights)
        self.scalers=scalers;self.seed=seed;self.rng=np.random.default_rng(seed+1000003)
        self.samples=deque(maxlen=200);self.interactions=0;self.updates=0

    def reset(self,sku):self.sku=sku

    def reset_evaluation(self,weights,seed):
        self.model.load_state_dict(weights);self.model.eval();self.samples.clear()
        self.interactions=0;self.updates=0;self.seed=seed;self.rng=np.random.default_rng(seed+1000003)

    def encode(self,row,price):
        st=self.scalers[str(row['sku'])];dates=season(str(pd.Timestamp(row['week']).date()))
        h=np.zeros((1,12,9),dtype=np.float32);c=np.zeros((1,8),dtype=np.float32)
        h[0,:,0]=[(row[f'qty_lag_{j}']-st['qty_mean'])/st['qty_std'] for j in range(12,0,-1)]
        h[0,:,1]=[(row[f'price_lag_{j}']-st['price_mean'])/st['price_std'] for j in range(12,0,-1)]
        h[0,:,2:6]=dates[:-1];c[0,0]=(price-st['price_mean'])/st['price_std'];c[0,1:5]=dates[-1]
        for column,k in (('event_days',6),('snap_days',7)):
            h[0,:,k]=[row.get(f'{column}_lag_{j}',0)/7 for j in range(12,0,-1)]
            c[0,k-1]=row.get(column,0)/7
        return h,c

    def prediction_gradient(self,row,price):
        h,c=self.encode(row,price);current=torch.from_numpy(c).requires_grad_(True)
        z=self.model(torch.from_numpy(h),current)[0];st=self.scalers[str(row['sku'])]
        qty=max(0.,float(z.detach())*st['qty_std']+st['qty_mean'])
        grad=float(torch.autograd.grad(z,current)[0][0,0])*st['qty_std']/st['price_std'] if qty>0 else 0.
        return qty,grad

    def observation(self,row,current,surface):
        qm=float(np.mean(np.log1p(surface.quantities)));qs=max(float(np.std(np.log1p(surface.quantities))),.1)
        pm=float(np.mean(surface.log_prices));ps=max(float(np.std(surface.log_prices)),.05)
        q=np.array([row[f'qty_lag_{j}'] for j in range(1,13)])
        p=np.array([row[f'price_lag_{j}'] for j in range(1,13)])
        prediction,gradient=self.prediction_gradient(row,current)
        week=int(pd.Timestamp(row['week']).isocalendar().week)
        return np.r_[np.clip((np.log1p(q)-qm)/qs,-10,10),np.clip((np.log(p)-pm)/ps,-10,10),
                     np.sin(2*np.pi*week/52),np.cos(2*np.pi*week/52),row['event_days']/7,row['snap_days']/7,
                     np.clip((np.log(current)-pm)/ps,-10,10),np.clip((np.log1p(prediction)-qm)/qs,-10,10),
                     np.clip(gradient*current/max(float(np.mean(surface.quantities)),1.),-10,10)].astype(np.float32)

    def feedback(self,row,price,qty):
        self.samples.append((dict(row),float(price),float(qty)));self.interactions+=1
        if self.interactions%10:return
        ids=self.rng.choice(len(self.samples),size=min(32,len(self.samples)),replace=False)
        examples=[self.samples[int(i)] for i in ids];h=[];c=[];target=[]
        for row,price,qty in examples:
            hi,ci=self.encode(row,price);h.append(hi[0]);c.append(ci[0]);st=self.scalers[str(row['sku'])]
            target.append((qty-st['qty_mean'])/st['qty_std'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed+2000003+self.updates);self.model.train()
            optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(self.model(torch.tensor(np.array(h)),torch.tensor(np.array(c))),torch.tensor(target,dtype=torch.float32))
            if not torch.isfinite(loss):raise ValueError('Non-finite Song feedback loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);optimizer.step();self.model.eval()
        self.updates+=1
