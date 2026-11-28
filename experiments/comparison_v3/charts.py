"""Build comparison tables and publication-ready PNG/SVG figures from raw runs."""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import math
import os

import numpy as np
import pandas as pd
_cache=Path(__file__).resolve().parents[2]/'tmp/comparison_mpl_cache'
_cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(_cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import data
from .common import read_json,write_json,digest,assert_same_keys,paired_interval,SEEDS

LABELS={'song_transformer':'Transformer theo Song','naive_last':'Tuần trước','mean_4':'Trung bình 4 tuần',
        'arimax_fallback':'ARIMAX + fallback','arimax_grid':'Lưới ARIMAX','arimax_ppo':'PPO + ARIMAX',
        'song_ppo_feedback':'PPO + Transformer Song','fixed_mean':'Giá trung bình train','fixed_upper':'Giá cận trên train'}
COLORS={'song_transformer':'#3268a8','naive_last':'#7d8994','mean_4':'#197b69','arimax_fallback':'#197b69',
        'arimax_grid':'#197b69','arimax_ppo':'#c58c26','song_ppo_feedback':'#3268a8','fixed_mean':'#8c959d','fixed_upper':'#353c42'}


class Accumulator:
    def __init__(self):self.n=0;self.y=0.;self.y2=0.;self.sse=0.;self.sae=0.;self.bias=0.;self.ape=0.;self.pos=0
    def add(self,y,p):
        y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
        if len(y)==0:return
        if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Nonfinite raw prediction')
        e=p-y;positive=y>0;self.n+=len(y);self.y+=float(y.sum());self.y2+=float(np.square(y).sum())
        self.sse+=float(np.square(e).sum());self.sae+=float(np.abs(e).sum());self.bias+=float(e.sum())
        self.ape+=float(np.abs(e[positive]/y[positive]).sum());self.pos+=int(positive.sum())
    def metrics(self):
        den=self.y2-self.y*self.y/self.n
        return {'n':self.n,'rmse':math.sqrt(self.sse/self.n),'mae':self.sae/self.n,
                'wape_pct':100*self.sae/self.y if self.y else None,'r2':1-self.sse/den if den>0 else None,
                'bias':self.bias/self.n,'mape_positive_pct':100*self.ape/self.pos if self.pos else None,'positive_n':self.pos}


def _save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)


def _remove_known_plot(directory,name):
    for extension in ('.png','.svg'):
        (Path(directory)/(name+extension)).unlink(missing_ok=True)


def forecast_tables(output):
    output=Path(output);acc=defaultdict(Accumulator);store_records=[]
    baseline_seen=defaultdict(set);eligible_count=defaultdict(int)
    for dataset in ('uci','m5'):
        for shard in data.shards(output,dataset):
            baseline_path=output/'forecast'/dataset/'baselines'/f'{shard}_test.csv.gz'
            if not baseline_path.exists():continue
            baseline=pd.read_csv(baseline_path,dtype={'sku':str},parse_dates=['week'])
            baseline_seen[dataset].add(shard)
            methods=['naive_last','mean_4']+(['arimax_fallback'] if dataset=='m5' else [])
            masks={'all_v2':np.ones(len(baseline),dtype=bool)}
            if dataset=='m5':
                masks['arimax_available']=baseline['arimax_available'].to_numpy(dtype=bool)
                eligible_count[dataset]+=int(masks['arimax_available'].sum())
            for scope,mask in masks.items():
                for method in methods:
                    part=baseline.loc[mask];acc[(dataset,scope,method,-1)].add(part['qty'],part[method])
                    row=Accumulator();row.add(part['qty'],part[method]);store_records.append({'dataset':dataset,'scope':scope,'store':shard,'method':method,'seed':-1,**row.metrics()})
            for seed in SEEDS:
                file=output/'forecast'/dataset/f'seed_{seed}'/'predictions'/f'{shard}_test.csv.gz'
                stamp_path=file.with_suffix('.json')
                if not file.exists() or not stamp_path.exists():continue
                stamp=read_json(stamp_path)
                if stamp['file']!=digest(file):raise ValueError('Forecast prediction hash differs from completion stamp')
                selected=read_json(file.parent.parent/'selected.json')
                if stamp['checkpoint']!=selected['checkpoint_sha256']:raise ValueError('Forecast prediction checkpoint differs from selected checkpoint')
                frame=pd.read_csv(file,dtype={'sku':str},parse_dates=['week'])
                assert_same_keys(frame,baseline)
                if not np.allclose(frame['qty'],baseline['qty'],rtol=0,atol=1e-8):raise ValueError('Forecast test labels differ')
                merged=baseline[['sku','week','qty']+(['arimax_available'] if dataset=='m5' else [])].merge(
                    frame[['sku','week','prediction']],on=['sku','week'],validate='one_to_one',sort=False)
                for scope,mask in masks.items():
                    part=merged.loc[mask];acc[(dataset,scope,'song_transformer',seed)].add(part['qty'],part['prediction'])
                    row=Accumulator();row.add(part['qty'],part['prediction']);store_records.append({'dataset':dataset,'scope':scope,'store':shard,'method':'song_transformer','seed':seed,**row.metrics()})
    records=[]
    expected=read_json(output/'data/manifest.json')['frozen_test_keys']
    for (dataset,scope,method,seed),value in acc.items():
        metrics=value.metrics()
        complete=(metrics['n']==expected[dataset]['count']) if scope=='all_v2' else (
            len(baseline_seen[dataset])==len(data.shards(output,dataset)) and metrics['n']==eligible_count[dataset])
        records.append({'dataset':dataset,'scope':scope,'method':method,'seed':seed,'complete_key_scope':complete,**metrics})
    frame=pd.DataFrame(records);pd.DataFrame(store_records).to_csv(output/'forecast_by_store.csv',index=False)
    frame.to_csv(output/'forecast_metrics.csv',index=False)
    return frame


def forecast_figures(output,metrics,figures):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,dataset in zip(axes,('uci','m5')):
        rows=metrics.loc[metrics['dataset'].eq(dataset)&metrics['scope'].eq('all_v2')&metrics['complete_key_scope']]
        if rows.empty:ax.text(.5,.5,'Chưa có kết quả đầy đủ',ha='center');continue
        groups=list(rows.groupby('method',sort=False))
        for i,(method,g) in enumerate(groups):
            ax.barh(i,g['rmse'].mean(),color=COLORS.get(method,'gray'),alpha=.85)
            if method=='song_transformer':ax.scatter(g['rmse'],np.full(len(g),i),color='#172633',s=18,zorder=3)
            label_x=g['rmse'].max()+rows['rmse'].max()*.015
            ax.text(label_x,i,f'{g["rmse"].mean():.2f}',va='center',fontsize=9)
        ax.set_yticks(range(len(groups)),[LABELS.get(m,m) for m,_ in groups]);ax.invert_yaxis()
        ax.set_xlim(0,rows['rmse'].max()*1.14)
        nseed=rows.loc[rows['method'].eq('song_transformer'),'seed'].nunique()
        ax.set_title(f'{dataset.upper()} · Song {nseed}/5 seed');ax.set_xlabel('RMSE (sản phẩm/tuần)')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Đối sánh dự báo trên toàn bộ khóa test V2 · khảo sát bổ sung',fontsize=12)
    fig.tight_layout();_save(fig,figures/'forecast_rmse_full')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    plotted=False
    for col,dataset in enumerate(('uci','m5')):
        for seed in SEEDS:
            selected=Path(output)/'forecast'/dataset/f'seed_{seed}'/'selected.json'
            if selected.exists():
                record=read_json(selected);curve=Path(output)/record['checkpoint'];curve=curve.parent/'learning_curve.csv'
                values=pd.read_csv(curve);label=f'seed {seed}'
            else:
                curves=sorted((Path(output)/'forecast'/dataset/f'seed_{seed}').glob('lr_*/learning_curve.csv'))
                if not curves:continue
                values=pd.read_csv(curves[0]);label=f'seed {seed}, đang chọn LR'
            plotted=True
            axes[0,col].plot(values['epoch'],values['train_mse_z'],label=label)
            axes[1,col].plot(values['epoch'],values['validation_rmse'],label=label)
        axes[0,col].set_title(dataset.upper());axes[0,col].set_ylabel('Train MSE (z-score)')
        axes[1,col].set_ylabel('Validation RMSE (sản phẩm/tuần)');axes[1,col].set_xlabel('Epoch')
        for ax in axes[:,col]:ax.grid(alpha=.15)
    if plotted:
        axes[0,0].legend(fontsize=8);fig.suptitle('Đường học và lựa chọn checkpoint · không dùng test để chọn epoch')
        fig.tight_layout();_save(fig,figures/'forecast_learning_curves')
    else:plt.close(fig)
    stores=pd.read_csv(Path(output)/'forecast_by_store.csv')
    m5=stores.loc[stores['dataset'].eq('m5')&stores['scope'].eq('all_v2')]
    if not m5.empty:
        pivot=m5.groupby(['store','method'])['rmse'].mean().unstack()
        ax=pivot.rename(columns=LABELS).plot.bar(figsize=(11,4),rot=0)
        ax.set_ylabel('RMSE (sản phẩm/tuần)');ax.set_xlabel('Cửa hàng');ax.set_title('M5: sai số theo cửa hàng · toàn bộ khóa V2')
        ax.legend(fontsize=8);ax.figure.tight_layout();_save(ax.figure,figures/'forecast_by_store')


def forecast_intervals(output,metrics,figures):
    records=[]
    for (dataset,scope),g in metrics.loc[metrics['complete_key_scope']].groupby(['dataset','scope']):
        reference=g.loc[g['method'].eq('song_transformer')].sort_values('seed')
        if len(reference)!=len(SEEDS):continue
        for method,ours in g.loc[g['seed'].eq(-1)].groupby('method'):
            if len(ours)!=1:raise ValueError('Duplicate deterministic forecast metrics')
            ref=reference['rmse'].to_numpy();value=float(ours['rmse'].iloc[0]);delta=100*(ref-value)/ref
            rng=np.random.default_rng(20261126)
            means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            records.append({'dataset':dataset,'scope':scope,'method':method,'reference':'song_transformer',
                'pairs':len(delta),'mean_rmse_reduction_pct':float(delta.mean()),'ci_low_pct':float(lo),'ci_high_pct':float(hi),
                'wins':int((delta>0).sum()),'uncertainty_scope':'training seeds; conditional on fixed V2 labels and keys'})
    frame=pd.DataFrame(records,columns=['dataset','scope','method','reference','pairs','mean_rmse_reduction_pct',
                                        'ci_low_pct','ci_high_pct','wins','uncertainty_scope'])
    frame.to_csv(Path(output)/'forecast_paired_intervals.csv',index=False)
    if frame.empty:
        _remove_known_plot(figures,'forecast_rmse_reduction_ci');return
    fig,ax=plt.subplots(figsize=(9,4.8));rows=frame.loc[frame['scope'].eq('all_v2')].reset_index(drop=True)
    for i,row in rows.iterrows():
        color=COLORS.get(row['method'],'gray')
        ax.plot([row['ci_low_pct'],row['ci_high_pct']],[i,i],color=color,lw=2)
        ax.scatter(row['mean_rmse_reduction_pct'],i,color=color,zorder=3)
    ax.set_yticks(range(len(rows)),[f'{r.dataset.upper()} · {LABELS[r.method]}' for r in rows.itertuples()])
    ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Giảm RMSE so với bản Song tái dựng (%) · dương là tốt hơn')
    ax.set_title('Khoảng bootstrap 95% qua 5 seed · cùng toàn bộ khóa V2');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();_save(fig,figures/'forecast_rmse_reduction_ci')


def coverage_and_audit(output,figures):
    records=[]
    for file in sorted((Path(output)/'arimax').glob('*.json')):
        entry=read_json(file)
        for reason,count in pd.Series(entry['status']).value_counts().items():records.append({'store':file.stem,'status':reason,'series':int(count)})
    if records:
        frame=pd.DataFrame(records);frame.to_csv(Path(output)/'arimax_fit_coverage.csv',index=False)
        pivot=frame.pivot(index='store',columns='status',values='series').fillna(0)
        ax=pivot.plot.bar(stacked=True,figsize=(10,4.5),rot=0)
        ax.set_title('Độ phủ ARIMAX train-only trên toàn bộ 29.069 chuỗi M5');ax.set_ylabel('Số chuỗi');ax.set_xlabel('Cửa hàng')
        ax.legend(title='Trạng thái fit',fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1.))
        ax.figure.tight_layout();_save(ax.figure,figures/'arimax_coverage')
    file=Path(output)/'simulator_audit/metrics.csv'
    if file.exists():
        frame=pd.read_csv(file);rows=frame.loc[frame['scope'].eq('in_train_support')]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(rows['method'],rows['rmse'],color=['#3268a8','#197b69','#7d8994'][:len(rows)])
        ax.set_ylabel('RMSE tại giá đã quan sát (sản phẩm/tuần)')
        ax.set_title('Kiểm tra simulator trên validation trong miền giá train')
        ax.text(.5,-.25,'Kiểm tra dự báo cần thiết; chưa xác nhận đáp ứng tại giá phản thực tế',transform=ax.transAxes,ha='center',fontsize=9)
        fig.tight_layout();_save(fig,figures/'simulator_validation')


def pricing_figures(output,figures):
    path=Path(output)/'pricing/runs.csv'
    if not path.exists():
        _remove_known_plot(figures,'pricing_uplift_ci');_remove_known_plot(figures,'pricing_revenue_seeds');return None
    runs=pd.read_csv(path);intervals=[]
    summary=runs.groupby(['simulator','method'],as_index=False).agg(
        seeds=('seed','nunique'),revenue_mean=('revenue','mean'),revenue_sample_std=('revenue','std'),
        revenue_min=('revenue','min'),revenue_max=('revenue','max'))
    summary.to_csv(Path(output)/'pricing_summary.csv',index=False)
    for simulator,rows in runs.groupby('simulator'):
        if rows['seed'].nunique()<2:continue
        for reference in ('song_ppo_feedback','fixed_upper'):
            for method in rows['method'].unique():
                if method==reference:continue
                result=paired_interval(rows,method,reference);result['simulator']=simulator;intervals.append(result)
    frame=pd.DataFrame(intervals,columns=['method','reference','pairs','mean_uplift_pct','ci_low_pct','ci_high_pct',
        'wins','losses','ties','uncertainty_scope','few_seed_warning','simulator'])
    frame.to_csv(Path(output)/'pricing_paired_intervals.csv',index=False)
    if not frame.empty:
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,simulator in zip(axes,sorted(runs['simulator'].unique())):
            values=frame.loc[frame['simulator'].eq(simulator)&frame['reference'].eq('song_ppo_feedback')].reset_index(drop=True)
            for i,row in values.iterrows():
                ax.plot([row['ci_low_pct'],row['ci_high_pct']],[i,i],color=COLORS.get(row['method'],'gray'),lw=2)
                ax.scatter(row['mean_uplift_pct'],i,color=COLORS.get(row['method'],'gray'),zorder=3)
            ax.set_yticks(range(len(values)),[LABELS.get(x,x) for x in values['method']]);ax.axvline(0,color='black',lw=.8)
            ax.set_title(simulator);ax.set_xlabel('Chênh lệch doanh thu so với Song PPO (%)');ax.grid(axis='x',alpha=.15)
        count=runs['seed'].nunique()
        fig.suptitle(f'Khoảng bootstrap 95% ghép cặp qua {count} seed · điều kiện theo cohort và simulator',fontsize=11)
        fig.tight_layout();_save(fig,figures/'pricing_uplift_ci')
    else:_remove_known_plot(figures,'pricing_uplift_ci')
    normalized=[]
    for simulator,g in runs.groupby('simulator'):
        reference=g.loc[g['method'].eq('fixed_upper')].set_index('seed')['revenue']
        for row in g.itertuples():normalized.append({'simulator':simulator,'method':row.method,'seed':row.seed,'index':100*row.revenue/reference[row.seed]})
    normalized=pd.DataFrame(normalized);normalized.to_csv(Path(output)/'pricing_revenue_index.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,sim in zip(axes,sorted(runs['simulator'].unique())):
        groups=list(normalized.loc[normalized['simulator'].eq(sim)].groupby('method',sort=False))
        for i,(method,g) in enumerate(groups):
            ax.scatter(np.full(len(g),i),g['index'],color=COLORS.get(method,'gray'),alpha=.7,s=28)
            ax.plot([i-.2,i+.2],[g['index'].mean()]*2,color='black',lw=2)
        ax.axhline(100,color='gray',ls='--',lw=.8);ax.set_xticks(range(len(groups)),[LABELS[m] for m,_ in groups],rotation=25,ha='right')
        ax.set_title(sim);ax.set_ylabel('Chỉ số doanh thu (giá cận trên = 100)');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Doanh thu mô phỏng · từng chấm là một lượt seed');fig.tight_layout();_save(fig,figures/'pricing_revenue_seeds')
    return None if frame.empty else frame


def build(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=forecast_tables(output)
    if not metrics.empty:
        forecast_figures(output,metrics,figures)
        forecast_intervals(output,metrics,figures)
    coverage_and_audit(output,figures)
    intervals=pricing_figures(output,figures)
    from .pricing_charts import diagnostics,learning
    learning(output,figures);diagnostics(output,figures)
    statuses=[]
    for dataset in ('uci','m5'):
        selected=[seed for seed in SEEDS if (output/'forecast'/dataset/f'seed_{seed}'/'selected.json').exists()]
        complete=[]
        for seed in selected:
            completed=metrics.loc[metrics['dataset'].eq(dataset)&metrics['scope'].eq('all_v2')&
                                  metrics['method'].eq('song_transformer')&metrics['seed'].eq(seed)&metrics['complete_key_scope']]
            if len(completed)==1:complete.append(seed)
        statuses.append({'dataset':dataset,'selected_seeds':selected,'fully_predicted_seeds':complete,
                         'expected_seeds':list(SEEDS),'full_comparison_complete':complete==list(SEEDS)})
    pricing_marker=output/'pricing/complete.json'
    pricing_complete=read_json(pricing_marker).get('full_comparison_complete',False) if pricing_marker.exists() else False
    write_json(output/'comparison_status.json',{'forecast':statuses,'pricing_complete':pricing_complete,
              'evidence_status':data.config()['evidence_status'],'plots':[str(p.relative_to(output)) for p in sorted(figures.glob('*.png'))]})
    lines=['# Đối sánh Song và mô hình dự án','',
        'Kết quả là khảo sát bổ sung trên test V2 đã từng được xem. Song là bản tái dựng, không phải mã nguồn tác giả. Không sửa báo cáo Word.','',
        '## Trạng thái','']
    for status in statuses:lines.append(f'- {status["dataset"].upper()}: {len(status["fully_predicted_seeds"])}/5 seed có đủ dự báo trên toàn bộ khóa.')
    lines+=['','## Dự báo','', '| Dataset | Phạm vi | Mô hình | Seed | Số khóa | RMSE | MAE | WAPE % |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.dataset} | {row.scope} | {LABELS.get(row.method,row.method)} | {row.seed} | {row.n} | {row.rmse:.4f} | {row.mae:.4f} | {row.wape_pct:.4f} |')
    lines+=['','Seed -1 là mô hình xác định, không được nhân bản để tạo độ lệch chuẩn. ARIMAX + fallback dùng tuần trước ở chuỗi không fit được; bảng arimax_available dùng cùng các khóa đủ mô hình giá cho tất cả phương pháp.','',
            '## Định giá','']
    lines.append('Đủ toàn bộ seed/phương pháp/phạm vi: '+('có' if pricing_complete else 'chưa'))
    if intervals is None:lines.append('Chưa có khoảng bất định định giá đủ điều kiện. Không suy ra từ bảng dự báo.')
    else:
        lines+=['| Simulator | Phương pháp | Đối chứng | Uplift TB % | Khoảng 95% | Thắng / seed |','|---|---|---|---:|---|---:|']
        for row in intervals.itertuples():lines.append(f'| {row.simulator} | {LABELS.get(row.method,row.method)} | {LABELS.get(row.reference,row.reference)} | {row.mean_uplift_pct:.3f} | [{row.ci_low_pct:.3f}; {row.ci_high_pct:.3f}] | {row.wins}/{row.pairs} |')
    lines+=['','## Giới hạn','',
        '- Năm seed cho khoảng bất định hạn chế; khoảng này không bao gồm sai số do simulator hoặc chọn dataset.',
        '- Doanh thu do simulator tạo ra không chứng minh tăng doanh thu thị trường.',
        '- Các kết quả baseline mới và Song train-only không trộn với bảng Phase 5 fit train+validation.',
        '- PPO có cùng ngân sách tương tác cố định; kiểm tra learning_curve và độ phủ chuỗi train trước khi diễn giải thứ hạng.',
        '- Chỉ kết luận cho các seed, khóa và model đã hoàn tất.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(output/'README.md',flush=True)
