"""Additional pricing curves, per-store comparisons and action diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from .charts import LABELS,COLORS,_save
from .common import read_json
import matplotlib.pyplot as plt


def diagnostics(output,figures):
    output=Path(output);files=sorted((output/'pricing').glob('seed_*/decisions/*.csv.gz'))
    if not files or not (output/'pricing/complete.json').exists():return
    included=set(read_json(output/'pricing/complete.json')['seeds'])
    files=[p for p in files if int(p.parent.parent.name.split('_',1)[1]) in included]
    cumulative=[];behavior=[]
    columns=['week','method','seed','simulator','price','previous_price','revenue','train_price_min','train_price_max']
    for index,file in enumerate(files):
        for chunk in pd.read_csv(file,usecols=columns,chunksize=100000):
            cumulative.append(chunk.groupby(['simulator','method','seed','week'],as_index=False)['revenue'].sum())
            chunk['at_upper']=np.isclose(chunk['price'],chunk['train_price_max'],rtol=0,atol=1e-8)
            chunk['at_lower']=np.isclose(chunk['price'],chunk['train_price_min'],rtol=0,atol=1e-8)
            chunk['absolute_change']=(chunk['price']-chunk['previous_price']).abs()
            behavior.append(chunk.groupby(['simulator','method','seed'],as_index=False).agg(
                decisions=('price','size'),upper=('at_upper','sum'),lower=('at_lower','sum'),absolute_change=('absolute_change','sum')))
        if (index+1)%10==0:print(f'[charts] pricing decisions {index+1}/{len(files)} files',flush=True)
    weekly=pd.concat(cumulative,ignore_index=True).groupby(['simulator','method','seed','week'],as_index=False)['revenue'].sum()
    weekly=weekly.sort_values(['simulator','method','seed','week'])
    weekly['cumulative_revenue']=weekly.groupby(['simulator','method','seed'])['revenue'].cumsum()
    weekly.to_csv(output/'pricing_weekly_cumulative.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,simulator in zip(axes,sorted(weekly['simulator'].unique())):
        for method in ('fixed_upper','arimax_grid','arimax_ppo','song_ppo_feedback'):
            part=weekly.loc[weekly['simulator'].eq(simulator)&weekly['method'].eq(method)]
            values=part.groupby('week')['cumulative_revenue'].agg(['mean','min','max'])
            dates=pd.to_datetime(values.index)
            ax.plot(dates,values['mean'],label=LABELS[method],color=COLORS[method])
            if method=='song_ppo_feedback':ax.fill_between(dates,values['min'],values['max'],color=COLORS[method],alpha=.12)
        ax.set_title(simulator);ax.set_ylabel('Doanh thu mô phỏng lũy kế (USD)');ax.tick_params(axis='x',rotation=25)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.suptitle('Trung bình qua seed · vùng tô Song là min–max, không phải khoảng tin cậy',fontsize=11)
    fig.tight_layout();_save(fig,figures/'pricing_cumulative')
    stats=pd.concat(behavior,ignore_index=True).groupby(['simulator','method','seed'],as_index=False)[['decisions','upper','lower','absolute_change']].sum()
    stats['upper_pct']=100*stats['upper']/stats['decisions'];stats['lower_pct']=100*stats['lower']/stats['decisions']
    stats['mean_absolute_change']=stats['absolute_change']/stats['decisions']
    stats.to_csv(output/'pricing_action_behavior.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,simulator in zip(axes,sorted(stats['simulator'].unique())):
        values=stats.loc[stats['simulator'].eq(simulator)].groupby('method')[['lower_pct','upper_pct']].mean()
        x=np.arange(len(values));ax.bar(x,values['lower_pct'],label='Cận dưới',color='#91aab8')
        ax.bar(x,values['upper_pct'],bottom=values['lower_pct'],label='Cận trên',color='#315f78')
        ax.set_xticks(x,[LABELS[m] for m in values.index],rotation=25,ha='right');ax.set_ylim(0,105)
        ax.set_ylabel('Tỷ lệ quyết định chạm biên (%)');ax.set_title(simulator)
    axes[0].legend(fontsize=8);fig.tight_layout();_save(fig,figures/'pricing_price_bounds')
    store_path=output/'pricing/store_runs.csv'
    if store_path.exists():
        runs=pd.read_csv(store_path);parts=[]
        for simulator,g in runs.groupby('simulator'):
            reference=g.loc[g['method'].eq('song_ppo_feedback'),['store','seed','revenue']].rename(columns={'revenue':'reference'})
            for method in ('arimax_grid','arimax_ppo','fixed_upper'):
                actual=g.loc[g['method'].eq(method),['store','seed','revenue']]
                merged=actual.merge(reference,on=['store','seed'],validate='one_to_one')
                if len(merged)!=len(actual) or len(merged)!=len(reference):raise ValueError('Store uplift has unmatched seed keys')
                merged['uplift_pct']=100*(merged['revenue']/merged['reference']-1)
                merged['method']=method;merged['simulator']=simulator;parts.append(merged)
        comparisons=pd.concat(parts,ignore_index=True);comparisons.to_csv(output/'pricing_by_store_paired.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,simulator in zip(axes,sorted(comparisons['simulator'].unique())):
            values=comparisons.loc[comparisons['simulator'].eq(simulator)].groupby(['store','method'])['uplift_pct'].mean().unstack()
            values.rename(columns=LABELS).plot.bar(ax=ax,rot=0)
            ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Chênh lệch so với Song PPO (%)');ax.set_title(simulator)
            ax.set_xlabel('Cửa hàng')
            ax.set_ylim(ax.get_ylim()[0],max(1.,float(values.max().max())*1.2))
            ax.legend(fontsize=7,ncol=3,loc='upper center')
        fig.suptitle('So sánh theo cửa hàng · trung bình uplift ghép cặp qua seed');fig.tight_layout();_save(fig,figures/'pricing_by_store')


def learning(output,figures):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));found=False
    for kind in ('song','arimax'):
        for file in sorted((Path(output)/'pricing').glob(f'seed_*/{kind}/learning_curve.csv')):
            values=pd.read_csv(file);found=True;label=f'{kind} · {file.parents[1].name}'
            axes[0].plot(values['timesteps'],values['validation_revenue'],label=label)
            axes[1].plot(values['timesteps'],values['train_series_seen'],label=label)
    if found:
        axes[0].set_ylabel('Doanh thu validation đại diện (USD mô phỏng)');axes[1].set_ylabel('Số chuỗi train đã tương tác')
        for ax in axes:ax.set_xlabel('Timestep huấn luyện');ax.grid(alpha=.15)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,fontsize=7,ncol=5,loc='lower center')
        fig.suptitle('Đường học PPO và độ phủ tương tác · không dùng test để chọn checkpoint')
        fig.tight_layout(rect=(0,.13,1,1));_save(fig,figures/'pricing_learning_curves')
    else:plt.close(fig)
