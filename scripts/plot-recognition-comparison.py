#!/usr/bin/env python3
"""Export paired recognition metrics, figures, spatial maps and error galleries."""
import argparse
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

NAMES={'efficientnet':'EfficientNet-B3','qwen':'Qwen3-VL'}
COLORS={'efficientnet':'#2475b0','qwen':'#8c52b8'}
SHORT={'arquus_vt4':'VT4','renault_ccfm':'CCFM','renault_gbc180':'GBC180','vltt':'VLTT'}
STATUS={'correct':'Acerto','wrong':'Classe incorreta','missed':'Não detectado'}
STATUS_COLOR={'correct':'#258c54','wrong':'#ce4147','missed':'#d98b18'}


def save(fig,folder,name,pdf=True):
    fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight')
    if pdf:fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def bars(ax,values,labels,title):
    x=np.arange(len(labels));width=.36
    for i,(name,ys) in enumerate(values.items()):
        b=ax.bar(x+(i-.5)*width,ys,width,label=NAMES[name],color=COLORS[name])
        ax.bar_label(b,labels=[f'{v:.1f}%' for v in ys],padding=3,fontsize=8)
    ax.set(xticks=x,xticklabels=labels,ylim=(0,112),ylabel='%',title=title)
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def load(run_paths):
    runs={};metrics={};images={};oracle={};rows=[]
    for name,path in run_paths.items():
        runs[name]=json.loads((path/'run.json').read_text())
        metrics[name]=json.loads((path/'metrics.json').read_text())
        images[name]={r['image']:r for r in json.loads((path/'predictions.json').read_text())['images']}
        oracle[name]={r['image']:r for r in json.loads((path/'gt_crops/predictions.json').read_text())['images']}
    a,b=runs.values()
    assert a['dataset_sha256']==b['dataset_sha256'],'Different evaluation data'
    assert a['reference_index']['gallery']==b['reference_index']['gallery'],'Different reference galleries'
    assert a['configuration']['detector']==b['configuration']['detector'],'Different detector settings'
    assert a['configuration']['detector']['crop_margin_pixels']>=0
    assert all(r['configuration']['recognition']['acceptance_threshold']==-1 for r in runs.values())
    assert all(m['failed_images']==0 for m in metrics.values())
    assert images['efficientnet'].keys()==images['qwen'].keys()
    labels=a['reference_index']['gallery']['labels']
    for identifier,reference in images['efficientnet'].items():
        other=images['qwen'][identifier]
        assert reference['image_sha256']==other['image_sha256']
        assert reference['ground_truth']==other['ground_truth']
        assert reference['matches']==other['matches']
        assert len(reference['detections'])==len(other['detections'])
        for p,q in zip(reference['detections'],other['detections']):
            assert np.allclose(p['bbox'],q['bbox'],atol=1e-5)
            assert abs(p['detection_confidence']-q['detection_confidence'])<1e-5
        for name in NAMES:
            record=images[name][identifier]
            mapping={m['truth_index']:m['prediction_index'] for m in record['matches']}
            assert len(oracle[name][identifier]['detections'])==len(record['ground_truth'])
            for i,truth in enumerate(record['ground_truth']):
                pred=record['detections'][mapping[i]] if i in mapping else None
                gt_pred=oracle[name][identifier]['detections'][i]
                box=truth['bbox'];correct=pred is not None and pred['label']==truth['label']
                rows.append({'model':name,'image':identifier,'truth_index':i,'true_label':truth['label'],
                             'predicted_label':pred['label'] if pred else 'not_detected',
                             'status':'correct' if correct else 'wrong' if pred else 'missed',
                             'correct':correct,'matched':pred is not None,
                             'top3_correct':pred is not None and truth['label'] in [c['label'] for c in pred['top_candidates']],
                             'score':pred['similarity'] if pred else np.nan,
                             'gt_top1_correct':gt_pred['candidate_label']==truth['label'],
                             'gt_top3_correct':truth['label'] in [c['label'] for c in gt_pred['top_candidates']],
                             'x':(box[0]+box[2])/(2*record['width']),
                             'y':(box[1]+box[3])/(2*record['height']),
                             'top3':json.dumps(pred['top_candidates']) if pred else '[]'})
    df=pd.DataFrame(rows)
    assert all(m['predictions']==m['similarity_on_matched_detections']['accepted'] for m in metrics.values())
    for name,m in metrics.items():
        part=df[df.model==name]
        assert int(part.correct.sum())==m['end_to_end']['true_positives']
        assert int(part.top3_correct.sum())==m['similarity_on_matched_detections']['known_topk_correct']
        assert int(part.gt_top1_correct.sum())==m['similarity_on_ground_truth_crops']['known_top1_correct']
    return runs,metrics,images,labels,df


def metric_plots(metrics,df,labels,folder):
    summaries=[]
    values={};prf={}
    for name,m in metrics.items():
        gt=m['similarity_on_ground_truth_crops'];matched=m['similarity_on_matched_detections']
        values[name]=[100*gt['raw_top1_accuracy_known'],100*gt['raw_topk_accuracy_known'],
                      100*matched['raw_top1_accuracy_known'],100*matched['raw_topk_accuracy_known'],
                      100*m['end_to_end']['recall']]
        prf[name]=[100*m[stage][stat] for stage in ['detection','end_to_end'] for stat in ['precision','recall','f1']]
        summaries.append({'model':name,'images':m['images'],'annotated_vehicles':m['ground_truth_objects'],
                          'detected':m['detection']['true_positives'],'missed':m['detection']['false_negatives'],
                          'false_positive_detections':m['detection']['false_positives'],
                          'gt_top1':gt['raw_top1_accuracy_known'],'gt_top3':gt['raw_topk_accuracy_known'],
                          'detector_crop_top1':matched['raw_top1_accuracy_known'],'detector_crop_top3':matched['raw_topk_accuracy_known'],
                          'final_correct':m['end_to_end']['true_positives'],'identification_recall':m['end_to_end']['recall'],
                          'detection_recall':m['detection']['recall'],'elapsed_seconds':m['elapsed_seconds']})
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    bars(ax,values,['GT Top-1\n(n=51)','GT Top-3\n(n=51)','YOLO Top-1\n(n=44)','YOLO Top-3\n(n=44)','Fluxo completo\n(n=51)'],'Reconhecimento')
    save(fig,folder,'01_recognition_metrics')
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    bars(ax,prf,['Localização\nPrecisão','Localização\nRecall','Localização\nF1','Identificação\nPrecisão','Identificação\nRecall','Identificação\nF1'],'Detecção e identificação')
    save(fig,folder,'02_pipeline_precision_recall')
    supported=[c for c in labels if (df.true_label==c).any()]
    per=[];classvalues={}
    for name in NAMES:
        classvalues[name]=[]
        for label in supported:
            sub=df[(df.model==name)&(df.true_label==label)]
            per.append({'model':name,'class':label,'support':len(sub),'detected':int(sub.matched.sum()),
                        'correct':int(sub.correct.sum()),'identification_recall':float(sub.correct.mean()),
                        'conditional_top1':float(sub[sub.matched].correct.mean())})
            classvalues[name].append(100*sub.correct.mean())
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    bars(ax,classvalues,[f'{SHORT[c]}\n(n={len(df[(df.model=="qwen")&(df.true_label==c)])})' for c in supported],'Recall por classe')
    save(fig,folder,'03_class_recall')
    return summaries,pd.DataFrame(per)


def confusion_plots(df,labels,folder):
    fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained')
    for name,ax in zip(NAMES,axes):
        part=df[df.model==name];columns=labels+['not_detected']
        counts=pd.crosstab(part.true_label,part.predicted_label).reindex(index=labels,columns=columns,fill_value=0)
        assert int(counts.to_numpy().sum())==len(part)
        counts.to_csv(folder.parent/f'confusion_{name}_counts.csv')
        norm=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0)
        norm.to_csv(folder.parent/f'confusion_{name}_normalized.csv')
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#e4e4e4')
        im=ax.imshow(np.ma.masked_invalid(norm.to_numpy()),vmin=0,vmax=1,cmap=cmap,aspect='auto')
        for i in range(len(labels)):
            for j in range(len(columns)):
                value=norm.iloc[i,j]
                text='—' if pd.isna(value) else f'{counts.iloc[i,j]}\n{value:.0%}'
                ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if pd.notna(value) and value>.55 else 'black')
        ax.set(xticks=range(len(columns)),xticklabels=[SHORT.get(c,'Não\ndetectado') for c in columns],
               yticks=range(len(labels)),yticklabels=[f'{SHORT[c]} (n={int(counts.loc[c].sum())})' for c in labels],
               xlabel='Classe prevista / perda de localização',ylabel='Classe anotada',title=NAMES[name])
        fig.colorbar(im,ax=ax,label='Fração da classe anotada',shrink=.7)
    fig.suptitle('Matrizes de confusão')
    save(fig,folder,'04_confusion_matrices')


def score_plots(df,folder):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    curves=[]
    for name,ax in zip(NAMES,axes):
        sub=df[(df.model==name)&df.matched]
        for correct,color,label in [(True,'#258c54','Acertos'),(False,'#ce4147','Erros')]:
            scores=sub.loc[sub.correct==correct,'score']
            ax.hist(scores,bins=np.linspace(0,1,26),alpha=.65,color=color,label=f'{label} (n={len(scores)})')
        ax.set(title=NAMES[name],xlabel='Similaridade máxima por classe (cosseno)',ylabel='Veículos',xlim=(0,1));ax.legend()
        for threshold in sorted(sub.score.unique(),reverse=True):
            kept=sub[sub.score>=threshold]
            curves.append({'model':name,'threshold':threshold,'accepted':len(kept),'coverage':len(kept)/len(sub),
                           'accepted_accuracy':float(kept.correct.mean()),'recognition_recall':kept.correct.sum()/51})
    fig.suptitle('Distribuição dos scores')
    save(fig,folder,'05_scores')
    curve=pd.DataFrame(curves);curve.to_csv(folder.parent/'score_coverage.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for name in NAMES:
        sub=curve[curve.model==name].sort_values('coverage')
        ax.plot(sub.coverage*100,sub.accepted_accuracy*100,marker='.',label=NAMES[name],color=COLORS[name])
    ax.set(xlabel='Cobertura entre veículos detectados (%)',ylabel='Acurácia entre aceitos (%)',ylim=(0,105),xlim=(0,105),title='Acurácia × cobertura')
    ax.legend();ax.grid(alpha=.2)
    save(fig,folder,'06_score_coverage')


def spatial_plots(df,folder,bins=6):
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout='constrained');csv=[]
    for column,name in enumerate(NAMES):
        sub=df[df.model==name]
        support=np.histogram2d(sub.y,sub.x,bins=bins,range=[[0,1],[0,1]])[0]
        correct=np.histogram2d(sub[sub.correct].y,sub[sub.correct].x,bins=bins,range=[[0,1],[0,1]])[0]
        wrong=support-correct
        rates=np.divide(correct,support,out=np.full_like(correct,np.nan),where=support>0)
        assert support.sum()==51 and correct.sum()==sub.correct.sum()
        for row,(values,cmap,title,maxvalue) in enumerate([(rates,'RdYlGn','Taxa de acerto',1),(np.where(support>0,wrong,np.nan),'Reds','Erros de classe + perdas YOLO',max(1,wrong.max()))]):
            ax=axes[row,column];palette=plt.get_cmap(cmap).copy();palette.set_bad('#e4e4e4')
            im=ax.imshow(np.ma.masked_invalid(values),origin='upper',extent=[0,1,1,0],vmin=0,vmax=maxvalue,cmap=palette)
            for y,x in zip(*np.nonzero(support)):
                caption=f'{int(correct[y,x])}/{int(support[y,x])}' if row==0 else str(int(wrong[y,x]))
                dark = rates[y,x] >= .8 or rates[y,x] <= .15 if row==0 else wrong[y,x]/maxvalue>.55
                ax.text((x+.5)/bins,(y+.5)/bins,caption,ha='center',va='center',fontsize=9,color='white' if dark else 'black')
            ax.set(title=f'{NAMES[name]} — {title}',xlabel='x / largura',ylabel='y / altura')
            fig.colorbar(im,ax=ax,shrink=.75)
        for y in range(bins):
            for x in range(bins):csv.append({'model':name,'x_bin':x,'y_bin':y,'support':int(support[y,x]),'correct':int(correct[y,x]),'errors_and_misses':int(wrong[y,x]),'accuracy':rates[y,x]})
    fig.suptitle('Acertos por posição')
    save(fig,folder,'07_spatial_success_heatmaps');pd.DataFrame(csv).to_csv(folder.parent/'spatial_support.csv',index=False)


def failure_heatmaps(df,folder,bins=6):
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    data={};rows=[]
    for name in NAMES:
        sub=df[df.model==name]
        support=np.histogram2d(sub.y,sub.x,bins=bins,range=[[0,1],[0,1]])[0]
        for status in ['wrong','missed']:
            subset=sub[sub.status==status]
            counts=np.histogram2d(subset.y,subset.x,bins=bins,range=[[0,1],[0,1]])[0]
            assert counts.sum()==len(subset)
            data[name,status]=(counts,support)
            for y in range(bins):
                for x in range(bins):rows.append({'model':name,'status':status,'x_bin':x,'y_bin':y,'support':int(support[y,x]),'failures':int(counts[y,x])})
    for row,status in enumerate(['wrong','missed']):
        maximum=max(1,max(data[name,status][0].max() for name in NAMES))
        for col,name in enumerate(NAMES):
            counts,support=data[name,status];ax=axes[row,col]
            cmap=plt.get_cmap('Reds' if status=='wrong' else 'YlOrBr').copy();cmap.set_bad('#e4e4e4')
            im=ax.imshow(np.ma.masked_where(support==0,counts),origin='upper',extent=[0,1,1,0],vmin=0,vmax=maximum,cmap=cmap)
            for y,x in zip(*np.nonzero(support)):
                ax.text((x+.5)/bins,(y+.5)/bins,str(int(counts[y,x])),ha='center',va='center',color='white' if counts[y,x]/maximum>.55 else 'black')
            title='Classe incorreta' if status=='wrong' else 'Perda do YOLO'
            ax.set(title=f'{NAMES[name]} · {title}',xlabel='x / largura',ylabel='y / altura')
            fig.colorbar(im,ax=ax,shrink=.75,label='Veículos')
    fig.suptitle('Erros e perdas')
    save(fig,folder,'08_failure_heatmaps')
    pd.DataFrame(rows).to_csv(folder.parent/'failure_spatial_counts.csv',index=False)


def overview(metrics,folder):
    fig=plt.figure(figsize=(15,7),layout='constrained');grid=fig.add_gridspec(2,2,width_ratios=[1.1,1.4])
    ax=fig.add_subplot(grid[:,0]);positions=np.arange(2);bottom=np.zeros(2)
    totals=[metrics[name]['ground_truth_objects'] for name in NAMES]
    states={
        'correct':[metrics[name]['end_to_end']['true_positives'] for name in NAMES],
        'wrong':[metrics[name]['detection']['true_positives']-metrics[name]['end_to_end']['true_positives'] for name in NAMES],
        'missed':[metrics[name]['detection']['false_negatives'] for name in NAMES],
    }
    for status,counts in states.items():
        percentages=np.array(counts)/totals*100
        ax.bar(positions,percentages,bottom=bottom,color=STATUS_COLOR[status],label=STATUS[status])
        for i,(n,value) in enumerate(zip(counts,percentages)):
            if n:ax.text(i,bottom[i]+value/2,f'{n}\n{value:.1f}%',ha='center',va='center',color='white',weight='bold')
        bottom+=percentages
    ax.set(xticks=positions,xticklabels=list(NAMES.values()),ylim=(0,100),ylabel='% dos 51 veículos anotados',title='Resultado por veículo')
    ax=fig.add_subplot(grid[0,1]);values={}
    for name,m in metrics.items():
        r=m['similarity_on_matched_detections'];values[name]=[100*m['detection']['recall'],100*r['raw_top1_accuracy_known'],100*r['raw_topk_accuracy_known']]
    bars(ax,values,['Detecção\n(n=51)','Top-1\n(n=44)','Top-3\n(n=44)'],'Detecção e classificação')
    ax=fig.add_subplot(grid[1,1]);ax.axis('off');ax.set_title('Identificação no fluxo completo')
    cells=[[NAMES[name]]+[f'{m["end_to_end"][key]:.2%}' for key in ['precision','recall','f1']] for name,m in metrics.items()]
    table=ax.table(cellText=cells,colLabels=['Extrator','Precisão','Recall','F1'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
    for (r,c),cell in table.get_celld().items():
        if r==0:cell.set_facecolor('#e6edf3');cell.set_text_props(weight='bold')
    fig.suptitle('Desempenho geral')
    fig.legend(handles=[Patch(color=STATUS_COLOR[s],label=STATUS[s]) for s in states],loc='outside lower center',ncol=3)
    save(fig,folder,'09_overall_performance')


def topk_table(df,metrics,folder):
    fig,axes=plt.subplots(3,1,figsize=(15,10),layout='constrained',gridspec_kw={'height_ratios':[1,1,1.7]})
    global_rows=[];score_rows=[];class_rows=[];score_csv=[]
    for name,m in metrics.items():
        gt=m['similarity_on_ground_truth_crops'];det=m['similarity_on_matched_detections']
        global_rows.append([NAMES[name],f'{gt["known_top1_correct"]}/51 ({gt["raw_top1_accuracy_known"]:.2%})',f'{gt["known_topk_correct"]}/51 ({gt["raw_topk_accuracy_known"]:.2%})',f'{det["known_top1_correct"]}/44 ({det["raw_top1_accuracy_known"]:.2%})',f'{det["known_topk_correct"]}/44 ({det["raw_topk_accuracy_known"]:.2%})'])
        candidates=[json.loads(value) for value in df.loc[(df.model==name)&df.matched,'top3']]
        means=[float(np.mean([items[i]['similarity'] for items in candidates])) for i in range(3)]
        score_rows.append([NAMES[name]]+[f'{value:.4f}' for value in means])
        score_csv.append({'model':name,**{f'mean_rank_{i+1}_cosine':v for i,v in enumerate(means)}})
    for label in df.true_label.unique():
        sub=df[(df.true_label==label)&df.matched];n=int((sub.model=='qwen').sum())
        values=[SHORT[label],str(n)]
        for name in NAMES:
            part=sub[sub.model==name];values += [f'{int(part.correct.sum())}/{n}',f'{int(part.top3_correct.sum())}/{n}']
        class_rows.append(values)
    tables=[('Acertos',[ 'Extrator','GT Top-1','GT Top-3','YOLO Top-1','YOLO Top-3'],global_rows),
            ('Cosseno médio por posição',[ 'Extrator','1º candidato','2º candidato','3º candidato'],score_rows),
            ('Acertos por classe — veículos detectados',['Classe','n','Eff. Top-1','Eff. Top-3','Qwen Top-1','Qwen Top-3'],class_rows)]
    for ax,(title,headers,rows) in zip(axes,tables):
        ax.axis('off');ax.set_title(title)
        table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='center')
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
        for (r,c),cell in table.get_celld().items():
            if r==0:cell.set_facecolor('#e6edf3');cell.set_text_props(weight='bold')
            elif r%2==0:cell.set_facecolor('#f5f7f9')
    fig.suptitle('Top-1 e Top-3')
    fig.text(.5,0,'GT: 51 recortes anotados · YOLO: 44 recortes detectados · Cosseno não é probabilidade e tem escalas diferentes entre extratores.',ha='center',fontsize=9)
    save(fig,folder,'10_topk_score_table')
    pd.DataFrame(score_csv).to_csv(folder.parent/'rank_score_means.csv',index=False)


def crop_view(ax,image,truth,pred,margin=5):
    boxes=[truth['bbox']]+([pred['bbox']] if pred else [])
    x1=min(b[0] for b in boxes);y1=min(b[1] for b in boxes);x2=max(b[2] for b in boxes);y2=max(b[3] for b in boxes)
    pad=max(20,.15*max(x2-x1,y2-y1));region=(max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(image.width,math.ceil(x2+pad)),min(image.height,math.ceil(y2+pad)))
    ax.imshow(image.crop(region));ax.set_axis_off()
    def rectangle(box,color,style='-'):
        a,b,c,d=box;ax.add_patch(Rectangle((a-region[0],b-region[1]),c-a,d-b,fill=False,color=color,linewidth=1.5,linestyle=style))
    rectangle(truth['bbox'],'#258c54')
    if pred:
        rectangle(pred['bbox'],'#2475b0','--')
        a,b,c,d=pred['bbox'];rectangle((max(0,math.floor(a-margin)),max(0,math.floor(b-margin)),min(image.width,math.ceil(c+margin)),min(image.height,math.ceil(d+margin))),'#b04ab0',':')


def error_grids(root,images,df,folder,margin):
    counts={}
    for name in NAMES:
        errors=df[(df.model==name)&~df.correct].sort_values(['status','image','truth_index'])
        counts[name]=len(errors);columns=4;nrows=max(1,math.ceil(len(errors)/columns))
        fig,axes=plt.subplots(nrows,columns,figsize=(16,nrows*3.7),squeeze=False,layout='constrained')
        for ax in axes.flat:ax.set_axis_off()
        for ax,row in zip(axes.flat,errors.itertuples()):
            record=images[name][row.image];truth=record['ground_truth'][row.truth_index]
            match=next((m for m in record['matches'] if m['truth_index']==row.truth_index),None)
            pred=record['detections'][match['prediction_index']] if match else None
            with Image.open(root/'data/forest_vehicles'/row.image) as opened:crop_view(ax,opened.convert('RGB'),truth,pred,margin)
            short=Path(row.image).stem.split('_JPG')[0]
            candidates=json.loads(row.top3)
            ranking=' > '.join(SHORT[c['label']] for c in candidates)
            caption=f'{short} #{row.truth_index+1} | GT: {SHORT[row.true_label]}\n'
            caption+=f'Prev.: {SHORT[row.predicted_label]} | sim={row.score:.3f}\nTop-3: {ranking}' if row.matched else 'NÃO DETECTADO PELO YOLO'
            ax.set_title(caption,fontsize=9,color=STATUS_COLOR[row.status])
        if errors.empty:axes[0,0].text(.5,.5,'Nenhum erro',ha='center')
        fig.suptitle(f'{NAMES[name]} — erros',fontsize=13)
        fig.legend(handles=[Line2D([],[],color='#258c54',label='Anotação'),Line2D([],[],color='#2475b0',linestyle='--',label='Caixa YOLO'),Line2D([],[],color='#b04ab0',linestyle=':',label=f'Recorte (+{margin} px)')],loc='outside lower center',ncol=3)
        save(fig,folder,f'errors_{name}',pdf=False)
    return counts


def annotate(ax,image,record,ground_truth=False):
    ax.imshow(image);ax.set_axis_off()
    matched={m['prediction_index']:m['truth_index'] for m in record['matches']}
    detected={m['truth_index'] for m in record['matches']}
    items=record['ground_truth'] if ground_truth else record['detections']
    for i,item in enumerate(items):
        box=item['bbox']
        if ground_truth:color='#258c54';caption=SHORT[item['label']]
        else:
            correct=i in matched and item['label']==record['ground_truth'][matched[i]]['label']
            color=STATUS_COLOR['correct' if correct else 'wrong'];caption=f"{SHORT[item['label']]} {item['similarity']:.2f}"
        ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,linewidth=1.8))
        ax.text(box[0],box[1],caption,color='white',fontsize=8,bbox={'facecolor':color,'alpha':.8,'pad':1})
    if not ground_truth:
        for i,item in enumerate(record['ground_truth']):
            if i not in detected:
                a,b,c,d=item['bbox'];ax.add_patch(Rectangle((a,b),c-a,d-b,fill=False,color=STATUS_COLOR['missed'],linestyle='--',linewidth=2))


def frame_pairs(root,images,folder):
    manifest=[]
    for identifier,reference in images['efficientnet'].items():
        if not reference['ground_truth'] and not reference['detections']:continue
        with Image.open(root/'data/forest_vehicles'/identifier) as opened:image=opened.convert('RGB')
        fig,axes=plt.subplots(1,3,figsize=(18,4.4),layout='constrained')
        for column,name in enumerate(['truth','efficientnet','qwen']):
            record=reference if name=='truth' else images[name][identifier]
            annotate(axes[column],image,record,ground_truth=name=='truth')
            axes[column].set_title('Anotações' if name=='truth' else NAMES[name])
        short=Path(identifier).stem.split('_JPG')[0];filename=f'frame_{len(manifest)+1:03d}.png'
        fig.suptitle(short)
        fig.legend(handles=[Patch(color=color,label=STATUS[status]) for status,color in STATUS_COLOR.items()],loc='outside lower center',ncol=3,fontsize=9)
        fig.savefig(folder/filename,dpi=120,bbox_inches='tight');plt.close(fig)
        manifest.append({'image':identifier,'figure':filename})
    pd.DataFrame(manifest).to_csv(folder.parent/'frame_manifest.csv',index=False)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--efficientnet',type=Path,default=Path('outputs/recognition_comparison_margin5/runs_82/efficientnet'))
    parser.add_argument('--qwen',type=Path,default=Path('outputs/recognition_comparison_margin5/runs_82/qwen'))
    parser.add_argument('--output',type=Path,default=Path('outputs/recognition_comparison_margin5/analysis'))
    args=parser.parse_args();root=args.root.resolve();out=root/args.output
    for sub in ['figures','errors','frames']:(out/sub).mkdir(parents=True,exist_ok=True)
    runs,metrics,images,labels,df=load({'efficientnet':root/args.efficientnet,'qwen':root/args.qwen})
    margin=runs['efficientnet']['configuration']['detector']['crop_margin_pixels']
    references=len(runs['efficientnet']['reference_index']['gallery']['entries'])
    df.to_csv(out/'objects.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summaries,per_class=metric_plots(metrics,df,labels,out/'figures')
    pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False);per_class.to_csv(out/'per_class.csv',index=False)
    confusion_plots(df,labels,out/'figures');score_plots(df,out/'figures');spatial_plots(df,out/'figures')
    failure_heatmaps(df,out/'figures')
    overview(metrics,out/'figures')
    topk_table(df,metrics,out/'figures')
    print('Metrics, confusion matrices and spatial maps saved.',flush=True)
    errors=error_grids(root,images,df,out/'errors',margin)
    frames=frame_pairs(root,images,out/'frames')
    paired=df.pivot(index=['image','truth_index'],columns='model',values='correct')
    outcomes={'both_correct':int((paired.efficientnet&paired.qwen).sum()),'only_efficientnet':int((paired.efficientnet&~paired.qwen).sum()),'only_qwen':int((~paired.efficientnet&paired.qwen).sum()),'both_wrong_or_missed':int((~paired.efficientnet&~paired.qwen).sum())}
    result={'references':references,'classes':labels,'metrics':summaries,'paired_outcomes':outcomes,'error_grid_objects':errors,'paired_frames':len(frames),
            'checks':{'same_dataset':True,'same_gallery':True,'same_detector_settings_and_boxes':True,'margin':margin,'no_rejection':True},
            'limitations':['No unknown-vehicle ground truth','Scores are not calibrated probabilities and are not directly comparable between encoders','Localization metrics use fixed confidence/IoU; mAP is not computed','Exploratory comparison on previously inspected data']}
    (out/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    headers='| Extrator | GT Top-1 | GT Top-3 | YOLO Top-1 | YOLO Top-3 | Acertos finais / 51 |'
    lines=[f'# EfficientNet-B3 × Qwen — margem de {margin} px','',f'Mesmas 252 imagens, galeria de {references} referências e detector YOLO26s. Rejeição desativada. EfficientNet usa o recorte inteiro com bordas; Qwen usa a instrução genérica. As caixas de localização foram conferidas e são idênticas.','','[Abrir relatório visual](index.html)','',''+headers,'| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:lines.append(f"| {NAMES[r['model']]} | {r['gt_top1']:.2%} | {r['gt_top3']:.2%} | {r['detector_crop_top1']:.2%} | {r['detector_crop_top3']:.2%} | {r['final_correct']}/51 ({r['identification_recall']:.2%}) |")
    lines+=['','GT usa os 51 recortes definidos pelas anotações. YOLO usa os 44 veículos localizados; o fluxo completo inclui os sete perdidos. A margem só afeta os recortes do detector. O modelo de localização é compartilhado; as diferenças são na identificação.','',f"Comparação pareada: {outcomes['both_correct']} acertos comuns, {outcomes['only_efficientnet']} somente EfficientNet, {outcomes['only_qwen']} somente Qwen, {outcomes['both_wrong_or_missed']} erros/perdas comuns.",'','## Figuras','']
    for path in sorted((out/'figures').glob('*.png')):lines += [f'![{path.stem}](figures/{path.name})','']
    lines+=['## Grades de erros','','As grades incluem erros de classe e perdas de localização, identificados separadamente. Verde marca a anotação, azul a caixa YOLO e pontilhado o recorte com margem.','']
    for name in NAMES:lines += [f'![{NAMES[name]}](errors/errors_{name}.png)','']
    lines+=['## Reprodução e limites','','Regenerar com `python scripts/plot-recognition-comparison.py`. PNG e PDF estão em figures/, as duas grades em errors/, e todos os frames positivos pareados em frames/, sem ampliação. CSVs preservam métricas, suporte espacial, matrizes, scores e resultados por objeto.','','O catálogo contém as quatro classes anotadas. Não há exemplos anotados de veículos desconhecidos. Os mapas espaciais são posições normalizadas no enquadramento, não coordenadas geográficas; cada célula mostra seu suporte. Curvas de score/cobertura são calculadas das predições salvas em veículos conhecidos e não validam rejeição de desconhecidos. A seleção e esta comparação são exploratórias.','']
    (out/'report.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>EfficientNet × Qwen</title><style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px;background:#fafafa;color:#20252b}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc}nav a{margin-right:18px}.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}figure{margin:12px 0;background:white;padding:12px}figcaption{font-size:13px}</style><h1>EfficientNet-B3 × Qwen</h1><p>Margem 5 px · __REFERENCE_COUNT__ referências · sem rejeição · mesmas 252 imagens · localização YOLO compartilhada.</p><nav><a href="#metrics">Métricas</a><a href="#errors">Erros</a><a href="#frames">Mesmos frames</a><a href="report.md">Relatório</a><a href="metrics.csv">CSV</a></nav><h2 id="metrics">Métricas e mapas</h2>']
    page[0]=page[0].replace('__REFERENCE_COUNT__',str(references))
    page.append('<table><tr><th>Extrator</th><th>Top-1 GT</th><th>Top-1 YOLO</th><th>Acertos / 51</th></tr>')
    for r in summaries:page.append(f"<tr><td>{NAMES[r['model']]}</td><td>{r['gt_top1']:.2%}</td><td>{r['detector_crop_top1']:.2%}</td><td>{r['final_correct']}/51</td></tr>")
    page.append('</table><p>GT: 51 veículos; YOLO: 44 localizados. Não há suporte para avaliar veículos desconhecidos. Scores não são probabilidades calibradas. Os mapas mostram posição relativa na imagem, com suporte por célula.</p>')
    for path in sorted((out/'figures').glob('*.png')):page.append(f'<figure><a href="figures/{path.name}"><img loading="lazy" src="figures/{path.name}"></a><figcaption>{html.escape(path.stem)} · <a href="figures/{path.stem}.pdf">PDF</a></figcaption></figure>')
    page.append('<h2 id="errors">Grades de erros</h2><p>Erros de classe e veículos não detectados estão identificados separadamente.</p>')
    for name in NAMES:page.append(f'<h3>{NAMES[name]}</h3><a href="errors/errors_{name}.png"><img loading="lazy" src="errors/errors_{name}.png"></a>')
    paired_rows=[]
    page.append('<details><summary>Top-1, Top-3 e scores por veículo</summary><table><tr><th>Imagem / anotação</th><th>GT</th><th>EfficientNet</th><th>Qwen</th></tr>')
    for (identifier,index),group in df.groupby(['image','truth_index'],sort=True):
        item={'image':identifier,'truth_index':index,'true_label':group.iloc[0].true_label}
        cells=[]
        for name in NAMES:
            row=group[group.model==name].iloc[0]
            ranking=' > '.join(f"{SHORT[c['label']]} {c['similarity']:.3f}" for c in json.loads(row.top3))
            text=f"{SHORT[row.predicted_label]} ({row.score:.3f})" if row.matched else 'Não detectado'
            cells.append(html.escape(text)+'<br>'+html.escape(ranking))
            item.update({name+'_prediction':row.predicted_label,name+'_score':row.score,name+'_top3':row.top3,name+'_status':row.status})
        paired_rows.append(item)
        page.append('<tr><td>'+html.escape(Path(identifier).stem.split('_JPG')[0])+f' #{index+1}</td><td>'+html.escape(SHORT[item['true_label']])+'</td><td>'+cells[0]+'</td><td>'+cells[1]+'</td></tr>')
    page.append('</table></details>')
    pd.DataFrame(paired_rows).to_csv(out/'paired_objects.csv',index=False)
    page.append('<h2 id="frames">Mesmos frames</h2><div class="frames">')
    for item in frames:page.append(f'<figure><a href="frames/{item["figure"]}"><img loading="lazy" src="frames/{item["figure"]}"></a><figcaption>{html.escape(item["image"])}</figcaption></figure>')
    page.append('</div></html>');(out/'index.html').write_text('\n'.join(page))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
