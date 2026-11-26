"""Word-sized scientific figures from verified CSVs, using bundled ReportLab/PDFium."""
from pathlib import Path
import csv, json, math
from statistics import mean
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'experiments/artifacts/comparison_v3'
OUT=ROOT/'output/report_revision_support/comparison_word';OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('FigureArial','C:/Windows/Fonts/arial.ttf'))
FONT='FigureArial';BLUE='#3268a8';TEAL='#197b69';GRAY='#7d8994';ORANGE='#c58c26';DARK='#25333d'
LABEL={'naive_last':'Tuần trước','mean_4':'Trung bình 4 tuần','arimax_fallback':'ARIMAX + fallback',
       'song_transformer':'Transformer theo Song','arimax_grid':'Lưới ARIMAX','arimax_ppo':'PPO + ARIMAX',
       'fixed_mean':'Giá trung bình train','fixed_upper':'Giá cận trên train'}
def read(name):
    with (DATA/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def txt(c,x,y,text,size=10,align='left',color='#000000'):
    c.setFillColor(HexColor(color));c.setFont(FONT,size)
    getattr(c,{'left':'drawString','center':'drawCentredString','right':'drawRightString'}[align])(x,y,str(text))
def line(c,x1,y1,x2,y2,color='#cccccc',width=.5):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x1,y1,x2,y2)
def finish(c,name):
    c.save();pdf=pdfium.PdfDocument(str(OUT/(name+'.pdf')))
    pdf[0].render(scale=3).to_pil().save(OUT/(name+'.png'));pdf.close()
def fmt(x,n=3):return f'{x:.{n}f}'.replace('.',',')

def forecast():
    rows=read('forecast_metrics.csv');name='word_forecast';c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(450,410))
    for dataset,y,h,maxx,ticks in [('uci',260,112,180,[0,50,100,150]),('m5',58,127,10.8,[0,2,4,6,8,10])]:
        x=145.;w=270.;methods=['naive_last','mean_4']+(['arimax_fallback'] if dataset=='m5' else [])+['song_transformer']
        txt(c,x+w/2,y+h+23,f'{dataset.upper()}   5 seed cho Transformer',11,'center')
        for tick in ticks:
            xx=x+tick/maxx*w;line(c,xx,y,xx,y+h,'#e6e8e9');txt(c,xx,y-15,tick,9,'center')
        line(c,x,y,x+w,y,'#2b2b2b');line(c,x,y,x,y+h,'#2b2b2b')
        for i,method in enumerate(methods):
            values=[float(r['rmse']) for r in rows if r['dataset']==dataset and r['scope']=='all_v2' and r['method']==method and r['complete_key_scope']=='True']
            assert len(values)==(5 if method=='song_transformer' else 1)
            value=mean(values);yy=y+h-(i+.5)*h/len(methods)
            color=BLUE if method=='song_transformer' else TEAL if method in ('mean_4','arimax_fallback') else GRAY
            c.setFillColor(HexColor(color));c.rect(x,yy-9,value/maxx*w,18,fill=1,stroke=0)
            txt(c,x-8,yy-3,LABEL[method],10,'right')
            if method=='song_transformer':
                c.setFillColor(HexColor(DARK))
                for val in values:c.circle(x+val/maxx*w,yy,2.3,fill=1,stroke=0)
            txt(c,x+max(values)/maxx*w+5,yy-3,fmt(value),9.5)
        txt(c,x+w/2,y-33,'RMSE (sản phẩm mỗi tuần)',10,'center')
    finish(c,name)

def uplift():
    rows=read('pricing_paired_intervals.csv');name='word_pricing_ci';c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(450,428))
    methods=['arimax_grid','arimax_ppo','fixed_mean','fixed_upper'];x=141.;w=271.;low=-1.;high=17.
    xpos=lambda v:x+(v-low)/(high-low)*w
    for sim,y in [('gaussian',263),('lag_0.75',57)]:
        h=121.;txt(c,x+w/2,y+h+23,'Gaussian' if sim=='gaussian' else 'Lag với trọng số 0,75',11,'center')
        for tick in (0,4,8,12,16):
            xx=xpos(tick);line(c,xx,y,xx,y+h,'#dfe3e6');txt(c,xx,y-15,tick,9,'center')
        line(c,xpos(0),y,xpos(0),y+h,'#222222',.8);line(c,x,y,x+w,y,'#222222')
        for i,method in enumerate(methods):
            row=next(r for r in rows if r['simulator']==sim and r['method']==method and r['reference']=='song_ppo_feedback')
            assert int(row['pairs'])==5
            value,lo,hi=(float(row[k]) for k in ('mean_uplift_pct','ci_low_pct','ci_high_pct'))
            yy=y+h-(i+.5)*h/4;color=TEAL if method=='arimax_grid' else ORANGE if method=='arimax_ppo' else GRAY
            line(c,xpos(lo),yy,xpos(hi),yy,color,1.8)
            for edge in (lo,hi):line(c,xpos(edge),yy-3,xpos(edge),yy+3,color,1.)
            c.setFillColor(HexColor(color));c.circle(xpos(value),yy,3.1,fill=1,stroke=0)
            txt(c,x-8,yy-3,LABEL[method],10,'right')
        txt(c,x+w/2,y-34,'Chênh lệch doanh thu so với Song (%)',10,'center')
    finish(c,name)

def stores():
    rows=read('pricing_by_store_paired.csv');name='word_pricing_stores';c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(450,260))
    stores=sorted({r['store'] for r in rows});x=43.;y=52.;w=393.;h=158.
    for val in (0,5,10,15,20):
        yy=y+val/20*h;line(c,x,yy,x+w,yy,'#e0e4e7');txt(c,x-7,yy-3,val,9,'right')
    line(c,x,y,x+w,y,'#222222');line(c,x,y,x,y+h,'#222222')
    for j,sim in enumerate(('gaussian','lag_0.75')):
        color=TEAL if j==0 else BLUE
        for i,store in enumerate(stores):
            values=[float(r['uplift_pct']) for r in rows if r['simulator']==sim and r['method']=='arimax_grid' and r['store']==store]
            assert len(values)==5
            v=mean(values);assert 0<v<20
            xx=x+(i+.5)*w/10+(j-.5)*12
            c.setFillColor(HexColor(color));c.rect(xx-5.5,y,11,v/20*h,fill=1,stroke=0)
        xx=139+j*141;c.setFillColor(HexColor(color));c.rect(xx,235,11,7,fill=1,stroke=0)
        txt(c,xx+17,234,'Gaussian' if j==0 else 'Lag 0,75',10)
    for i,store in enumerate(stores):txt(c,x+(i+.5)*w/10,y-17,store,8.8,'center')
    c.saveState();c.translate(13,y+h/2);c.rotate(90);txt(c,0,0,'Chênh lệch so với Song (%)',10,'center');c.restoreState()
    txt(c,x+w/2,12,'Trung bình năm seed theo từng cửa hàng',10,'center')
    finish(c,name)

if __name__=='__main__':
    evidence=json.loads((DATA/'verification.json').read_text('utf-8'));assert evidence['complete']
    forecast();uplift();stores();print(OUT)
