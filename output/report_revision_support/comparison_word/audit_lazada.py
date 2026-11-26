from pathlib import Path
from hashlib import sha256
import json,math
import pandas as pd
import numpy as np
import word_figures as fig
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

BASE=Path('D:/sendhelp/BI/baocao');OUT=fig.OUT
manifest=json.loads((BASE/'manifest.json').read_text('utf-8'))
given=json.loads((BASE/'metrics.json').read_text('utf-8'))
panel=pd.read_csv(BASE/'panel.csv',dtype={'sku':str},parse_dates=['week'])
test=pd.read_csv(BASE/'test_predictions.csv',dtype={'sku':str},parse_dates=['week'])
assert not panel.duplicated(['sku','week']).any() and not test.duplicated(['sku','week']).any()
joined=test.merge(panel,on=['sku','week'],suffixes=('_prediction','_panel'),validate='one_to_one')
assert len(joined)==len(test)
for col in ['qty','price']:
    assert np.allclose(joined[col+'_prediction'],joined[col+'_panel'])
assert joined['split_panel'].eq('test').all()
assert np.isfinite(panel[['qty','price']]).all().all() and panel.qty.ge(0).all() and panel.price.gt(0).all()
assert np.isfinite(test.predicted_qty).all() and test.predicted_qty.ge(0).all()
def metrics(pred):
    y=test.qty.to_numpy();p=np.asarray(pred);e=p-y
    return {'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),
            'mae_log1p':float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
            'wape_pct':float(100*np.abs(e).sum()/y.sum()),
            'r2':float(1-np.square(e).sum()/np.square(y-y.mean()).sum())}
scores={'provided_ols':metrics(test.predicted_qty)}
for key,value in [('test_mae_qty','mae'),('test_rmse_qty','rmse'),('test_mae_log1p_qty','mae_log1p')]:
    assert np.isclose(given[key],scores['provided_ols'][value],rtol=1e-10,atol=1e-9)
train=panel.loc[panel.split.eq('train')&panel.observed_sales].sort_values(['sku','week'])
for method,k in [('last_train',1),('mean_last_3_train',3)]:
    by=train.groupby('sku').tail(k).groupby('sku').qty.mean()
    prediction=test.sku.map(by);assert prediction.notna().all()
    scores[method]=metrics(prediction)
    test[method]=prediction
errors=(test.predicted_qty-test.qty).to_numpy();order=np.argsort(errors**2)[::-1]
top=test.iloc[order[:5]].copy();top['error']=(top.predicted_qty-top.qty);top['sse_share_pct']=100*top.error**2/np.square(errors).sum()
top.to_csv(OUT/'lazada_largest_errors.csv',index=False)
pd.DataFrame([{'method':k,**v} for k,v in scores.items()]).to_csv(OUT/'lazada_metrics_verified.csv',index=False)
test.to_csv(OUT/'lazada_predictions_and_baselines.csv',index=False)
hashes={name:{'actual':sha256((BASE/name).read_bytes()).hexdigest(),'expected':h,
              'matches':sha256((BASE/name).read_bytes()).hexdigest()==h}
        for name,h in manifest['files'].items() if (BASE/name).exists()}
dates=sorted(panel.week.unique());gaps=sorted(set(int((dates[i]-dates[i-1])/np.timedelta64(1,'D')) for i in range(1,len(dates))))
summary={'source':'Lazada','label_definition':'Daily sold units according to user; delivered manifest aggregates into three-day periods',
 'price_currency':'VND according to user','panel_rows':len(panel),'panel_skus':panel.sku.nunique(),'periods':len(dates),'period_gaps_days':gaps,
 'train_periods':panel.loc[panel.split.eq('train'),'week'].nunique(),'test_periods':panel.loc[panel.split.eq('test'),'week'].nunique(),
 'period_label_first':str(panel.week.min().date()),'period_label_last':str(panel.week.max().date()),
 'train_last_period_label':str(panel.loc[panel.split.eq('train'),'week'].max().date()),
 'test_first_period_label':str(panel.loc[panel.split.eq('test'),'week'].min().date()),
 'test_panel_rows':int(panel.split.eq('test').sum()),'test_prediction_rows':len(test),'test_prediction_skus':test.sku.nunique(),
 'missing_qty':int(panel.qty.isna().sum()),'zero_qty_pct':float(100*panel.qty.eq(0).mean()),
 'quantity':panel.qty.describe(percentiles=[.5,.9,.95,.99]).to_dict(),'price':panel.price.describe(percentiles=[.5,.9,.95,.99]).to_dict(),
 'observed_sales_counts':{str(k):int(v) for k,v in panel.observed_sales.value_counts().items()},
 'file_hashes':hashes,'all_delivered_hashes_match':all(x['matches'] for x in hashes.values()),
 'model_file_available':(BASE/'model.json').exists(),'metrics_recomputed_match':True,'prediction_keys_and_labels_match_panel':True,
 'scores':scores,'largest_error':{'sku':str(top.iloc[0].sku),'week':str(top.iloc[0].week.date()),
 'actual':float(top.iloc[0].qty),'prediction':float(top.iloc[0].predicted_qty),'sse_share_pct':float(top.iloc[0].sse_share_pct)},
 'baseline_protocol':'Freeze last observed train qty or mean of last three observed train periods at the train cutoff; no test-label feedback.',
 'manifest_reported_source_audit':manifest['source_audit'],'raw_daily_and_product_files_independently_verified':False}
(OUT/'lazada_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n',encoding='utf-8')

# Histograms of all delivered panel rows with explicitly transformed axes.
name='word_lazada_distributions';c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(450,395))
for values,y,title,tickfmt,axis in [
    (np.log1p(panel.qty.to_numpy()),245,'Lượng bán mỗi kỳ 3 ngày',lambda t:f'{t:.1f}','log(1 + lượng bán)'),
    (np.log10(panel.price.to_numpy()),63,'Giá niêm yết theo VNĐ',lambda t:f'{t:.1f}','log10(giá VNĐ)')]:
    counts,edges=np.histogram(values,bins=10);x=54.;w=373.;h=112.;maxy=max(counts)*1.15
    fig.txt(c,x+w/2,y+h+16,title,11,'center')
    for val in [0,max(counts)/2,max(counts)]:
        yy=y+val/maxy*h;fig.line(c,x,yy,x+w,yy,'#e2e5e7');fig.txt(c,x-7,yy-3,f'{val:.0f}',9,'right')
    for i,count in enumerate(counts):
        xx=x+i*w/len(counts);c.setFillColor(HexColor(fig.TEAL if y>100 else fig.BLUE))
        c.rect(xx+.7,y,w/len(counts)-1.4,count/maxy*h,stroke=0,fill=1)
    fig.line(c,x,y,x+w,y,'#222222');fig.line(c,x,y,x,y+h,'#222222')
    for i in range(0,len(edges),2):
        xx=x+i/len(counts)*w;fig.txt(c,xx,y-14,tickfmt(edges[i]),9,'center')
    fig.txt(c,x+w/2,y-32,axis,10,'center')
    c.saveState();c.translate(12,y+h/2);c.rotate(90);fig.txt(c,0,0,'Số dòng panel',9,'center');c.restoreState()
fig.finish(c,name)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item()))
