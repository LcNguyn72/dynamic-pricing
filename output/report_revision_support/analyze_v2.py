"""Reproducible descriptive analysis and causal one-step baselines for the report."""
from pathlib import Path
import sys, json, hashlib, time, platform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(Path(__file__).parent / 'deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from final_system.pipeline.verified_weekly import load_contract
from final_system.data_pipeline import _verify_existing_panels

PREP = ROOT/'final_system/runtime/prepared'
OUT = ROOT/'output/report_revision_support'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':180})
MODELS = ['previous_week','mean_4','mean_8','positive_median_8']
records=[]; summaries=[]; aggregate={}; baseline_samples=[]

def metric(y,p):
    e=p-y; sst=float(np.square(y-y.mean()).sum()); positive=y>0
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))), 'mae':float(np.abs(e).mean()),
      'r2':float(1-np.square(e).sum()/sst) if sst else None,
      'wape_pct':float(100*np.abs(e).sum()/y.sum()) if y.sum() else None,
      'mape_positive_pct':float(100*np.mean(np.abs(e[positive])/y[positive])) if positive.any() else None,
      'positive_count':int(positive.sum()), 'bias':float(e.mean())}

def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',facecolor='white');plt.close(fig)

def run():
    start=time.perf_counter()
    contract=load_contract(PREP);_verify_existing_panels(PREP,contract)
    all_test={d:pd.read_csv(PREP/f'test_keys_{d}.csv',dtype={'sku':str},parse_dates=['week']) for d in ['uci','m5']}
    proof={}; stats=[]
    for dataset in ['uci','m5']:
        shards=[PREP/'uci/panel/weekly.csv'] if dataset=='uci' else [PREP/'m5'/f'weekly_{s}.csv' for s in contract['m5_stores']]
        collected={split:{m:[] for m in MODELS} for split in ['validation','test']}
        targets={split:[] for split in ['validation','test']}
        histogram_q=np.zeros(40,dtype=np.int64);histogram_p=np.zeros(40,dtype=np.int64)
        binsq=np.linspace(0,12,41);binsp=np.linspace(0,8,41)
        train_nprices=[];train_nobs=[];weekly=[];test_keys=[]
        nrows=nobs=nzero=0
        split_counts={}
        for path in shards:
            frame=pd.read_csv(path,usecols=['sku','week','qty','price','observed_sales','split'],dtype={'sku':str},parse_dates=['week'])
            frame=frame.sort_values(['sku','week']).reset_index(drop=True)
            assert not frame.duplicated(['sku','week']).any()
            obs=frame.observed_sales.astype(str).str.lower().isin(['true','1'])
            q=frame.qty.astype(float);g=q.groupby(frame.sku,sort=False)
            lag=pd.concat({i:g.shift(i) for i in range(1,13)},axis=1)
            methods={'previous_week':lag[1], 'mean_4':lag.loc[:,1:4].mean(axis=1),
              'mean_8':lag.loc[:,1:8].mean(axis=1),
              'positive_median_8':lag.loc[:,1:8].where(lag.loc[:,1:8]>0).median(axis=1).fillna(lag[1])}
            # All evaluation rows must have 12 complete historical calendar weeks,
            # matching the frozen V2 horizon; test key equality is checked below.
            price_lags=pd.concat({i:frame.price.groupby(frame.sku,sort=False).shift(i) for i in range(1,13)},axis=1)
            complete=lag.notna().all(axis=1) & price_lags.notna().all(axis=1) & price_lags.gt(0).all(axis=1)
            for split in ['validation','test']:
                mask=obs & frame.split.eq(split)&complete
                targets[split].append(q[mask].to_numpy())
                for m,pred in methods.items(): collected[split][m].append(pred[mask].to_numpy())
                if split=='test': test_keys.append(frame.loc[mask,['sku','week']])
            nrows+=len(frame);nobs+=int(obs.sum());nzero+=int((obs&q.eq(0)).sum())
            for split,part in frame.groupby('split'):
                split_counts[split]=split_counts.get(split,0)+int(obs.loc[part.index].sum())
            train=frame[frame.split.eq('train')&obs]
            train_nprices.extend(train.groupby('sku').price.nunique().tolist())
            train_nobs.extend(train.groupby('sku').size().tolist())
            histogram_q+=np.histogram(np.log1p(q[obs]),binsq)[0]
            histogram_p+=np.histogram(np.log1p(frame.loc[obs,'price']),binsp)[0]
            weekly.append(frame[obs].groupby('week').qty.sum())
            if dataset=='uci':
                sample=frame.loc[obs&frame.split.eq('test')&complete,['sku','week','qty']].copy()
                for m,pred in methods.items():sample[m]=pred.loc[sample.index]
                baseline_samples.append(sample)
            print(dataset,path.name,len(frame),flush=True)
        got=pd.concat(test_keys).sort_values(['sku','week']).reset_index(drop=True)
        wanted=all_test[dataset][['sku','week']].sort_values(['sku','week']).reset_index(drop=True)
        pd.testing.assert_frame_equal(got,wanted)
        proof[dataset]={'test_keys_matched':len(got),'sha256':contract['phase1'][dataset]['test_keys']['sha256']}
        for split in ['validation','test']:
            y=np.concatenate(targets[split])
            for m in MODELS: records.append({'dataset':dataset,'split':split,'model':m,**metric(y,np.concatenate(collected[split][m]))})
        stats.append({'dataset':dataset,'panel_rows':nrows,'observed_rows':nobs,'observed_zero_rows':nzero,
          'observed_zero_pct':100*nzero/nobs,'train_observed_rows':split_counts.get('train',0),
          'validation_observed_rows':split_counts.get('validation',0),'test_observed_rows':split_counts.get('test',0),
          'test_scored_keys':len(got),'train_series':len(train_nobs),
          'train_constant_price_series':int((np.array(train_nprices)<=1).sum())})
        assert int(histogram_q.sum())==nobs, 'Quantity histogram omitted observations'
        assert int(histogram_p.sum())==nobs, 'Price histogram omitted observations'
        pd.DataFrame({'left':binsq[:-1],'right':binsq[1:],'count':histogram_q}).to_csv(OUT/f'histogram_quantity_{dataset}.csv',index=False)
        pd.DataFrame({'left':binsp[:-1],'right':binsp[1:],'count':histogram_p}).to_csv(OUT/f'histogram_price_{dataset}.csv',index=False)
        aggregate[dataset]={'q':histogram_q,'p':histogram_p,'binsq':binsq,'binsp':binsp,
          'nprices':train_nprices,'nobs':train_nobs,'weekly':pd.concat(weekly,axis=1).sum(axis=1).sort_index()}
    metrics=pd.DataFrame(records);metrics.to_csv(OUT/'baseline_metrics.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'data_summary.csv',index=False)
    # Independent checks against supplied Phase 6 deterministic baseline scores.
    for dataset in ['uci','m5']:
        source=pd.read_csv(ROOT/f'experiments/artifacts/phase6_v2/{dataset}_forecast_test.csv')
        expected=float(source.loc[source.model.eq('naive_last'),'rmse'].iloc[0])
        actual=float(metrics.loc[metrics.dataset.eq(dataset)&metrics.split.eq('test')&metrics.model.eq('previous_week'),'rmse'].iloc[0])
        assert np.isclose(actual,expected,rtol=1e-10),(dataset,actual,expected)
    fig,axes=plt.subplots(2,2,figsize=(11,6.3),layout='constrained')
    for j,d in enumerate(['uci','m5']):
        v=aggregate[d]
        for i,(key,bins,label) in enumerate([('q','binsq','log(1 + lượng bán tuần)'),('p','binsp','log(1 + giá tuần)')]):
            ax=axes[i,j]; edges=v[bins];ax.bar(edges[:-1],v[key],width=np.diff(edges),align='edge',color=['#176b65','#375f92'][j],edgecolor='white',linewidth=.3)
            ax.set(title=d.upper(),xlabel=label,ylabel='Số quan sát hợp lệ');ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    savefig(fig,'eda_quantity_price')
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for ax,d in zip(axes,['uci','m5']):
        data=np.asarray(aggregate[d]['nprices']);ax.hist(data,bins=np.arange(.5,data.max()+1.5,1),color='#176b65',edgecolor='white')
        ax.set(title=f'{d.upper()} — giá trong train',xlabel='Số mức giá khác nhau',ylabel='Số chuỗi');ax.grid(axis='y',alpha=.2)
        ax.text(.96,.94,f'Tổng: {len(data):,} chuỗi\n1 mức giá: {(data<=1).sum():,}',transform=ax.transAxes,ha='right',va='top')
    savefig(fig,'eda_train_price_support')
    fig,axes=plt.subplots(1,2,figsize=(11,3.4),layout='constrained')
    for ax,d in zip(axes,['uci','m5']):
        series=aggregate[d]['weekly'];series.to_csv(OUT/f'weekly_total_{d}.csv',header=['qty'])
        ax.plot(series.index,series.values,color='#176b65',lw=1.3)
        for label,key,color in [('Validation','validation_start','#b87918'),('Test','test_start','#9c3c42')]:
            date=pd.Timestamp(json.loads((ROOT/'final_system/configs/serving_v2.json').read_text())['datasets'][d][key])
            ax.axvline(date,color=color,ls='--',label=label)
        ax.set(title=d.upper(),ylabel='Tổng lượng bán tuần');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',rotation=20)
    savefig(fig,'eda_weekly_sales')
    selected={}
    for d in ['uci','m5']:
        val=metrics[metrics.dataset.eq(d)&metrics.split.eq('validation')]
        selected[d]=str(val.sort_values(['rmse','model']).iloc[0].model)
    names={'previous_week':'Tuần trước','mean_4':'Trung bình 4 tuần','mean_8':'Trung bình 8 tuần','positive_median_8':'Trung vị dương 8 tuần'}
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,d in zip(axes,['uci','m5']):
        part=metrics[metrics.dataset.eq(d)&metrics.split.eq('test')].sort_values('rmse',ascending=False)
        ax.barh(part.model.map(names),part.rmse,color=['#176b65' if m==selected[d] else '#7791ae' for m in part.model])
        for i,v in enumerate(part.rmse):ax.text(v,i,f' {v:.2f}',va='center',fontsize=9)
        ax.set(title=d.upper(),xlabel='RMSE test (sản phẩm/tuần)',xlim=(0,part.rmse.max()*1.22));ax.grid(axis='x',alpha=.2)
    savefig(fig,'simple_baselines_rmse')
    proof.update({'selected_by_validation_rmse':selected,'purpose':'supplementary exploratory analysis; fixed V2 keys',
      'models_predeclared':MODELS,'seconds':time.perf_counter()-start,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
      'source_protocol_sha256':contract['protocol_sha256'],'no_model_retraining':True})
    (OUT/'analysis_manifest.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'data_summary.txt').write_text(pd.DataFrame(stats).to_string(index=False)+'\n\n'+metrics.to_string(index=False),encoding='utf-8')
    print(metrics.to_string(index=False),flush=True);print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':run()
