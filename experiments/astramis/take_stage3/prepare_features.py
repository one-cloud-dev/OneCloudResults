from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path('/home/jovyan/work')
OUT = ROOT/'runs/take_stage3'
VOL = ROOT/'runs/rgbd_volume_stage2'
MAT = ROOT/'runs/dinov2_material_stage1'
OLD = ROOT/'runs/material_final_m_20261121_7f35d05_oof4'
DATA = ROOT/'datasets/dataset'
read = lambda p: json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
sys.path.insert(0,str(ROOT/'datasets'))
import score
CLASSES = sorted(score.MATERIALS)
COLS = ['log_volume','log_mass','mass_over_limit',*['p_'+c for c in CLASSES],'solid','projected_m2']

def prepare():
    stage2 = read(VOL/'manifest.json'); stage1=read(MAT/'manifest.json')
    for directory, checks in [(ROOT,stage1['inputs']), (OLD.parent.parent,read(OLD/'protocol.json')['input_sha256']), (VOL,stage2['files']), (MAT,stage1['files'])]:
        for name,h in checks.items():
            if name != 'head_all40.npz':
                assert sha(directory/name)==h,name
    assert sha(ROOT/'datasets/score.py')==stage2['official_metric_sha256']
    split=read(ROOT/'baseline_v0/split_v0.json'); train=set(split['train']['object_ids']); val=set(split['val']['object_ids'])
    assert (len(train),len(val))==(32,8) and train.isdisjoint(val)
    stage1res=read(MAT/'results.json'); foldinfo={d['fold']:d for d in stage1res['folds']}; blocks={k:set(d['held']) for k,d in foldinfo.items()}
    assert {str(k):sorted(v) for k,v in blocks.items()}=={k:sorted(v) for k,v in read(VOL/'configuration.json')['held'].items()}
    for k,d in foldinfo.items():
        assert set(d['fit_objects'])==train-blocks[k]
    assert {k:sorted(v) for k,v in blocks.items()}=={f['fold']:sorted(f['held_objects']) for f in read(OLD/'results.json')['folds']}
    folds={o:k for k,b in blocks.items() for o in b}; assert len(folds)==sum(map(len,blocks.values()))==32
    cache=read(OLD/'oof_predictions.json')['A']; crops=read(MAT/'crops.json'); emb=np.load(MAT/'embeddings.npy'); assert len(crops)==len(emb)
    dets={(r['frame'],r['index']):i for i,r in enumerate(crops) if r['kind']=='det'}
    boxedgt={r['frame']:i for i,r in enumerate(crops) if r['kind']=='gt' and r['object_id'] in train}
    iv=read(MAT/'heldout_detector_indices.json'); dv={tuple(x):i for i,x in enumerate(iv)}; Xv=np.load(MAT/'heldout_detector_embeddings.npy'); assert len(dv)==len(Xv)==57
    head=np.load(MAT/'head_train32.npz'); assert set(head['fit_objects'])==train and list(head['classes'])==CLASSES
    probs_val=softmax(Xv@head['weight'].T+head['bias'],axis=1)
    prev_train={r['frame']:r['det_DINO'] for r in read(MAT/'oof_rows.json')}; prev_val={r['frame']:r['DINO'] for r in read(MAT/'heldout_detector.json')['rows']}
    tables={}; dino_x={}; gt_rf={}; det_rf={}; gt_folds={}; labels={}
    runner=ROOT/'onecloud_v2_20261120T220121/verified_candidate_v1/run_candidate_exact.py'; tree=ast.parse(runner.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='material_features'); scope={'np':np,'cv2':cv2}; exec(compile(ast.Module([fn],type_ignores=[]),str(runner),'exec'),scope); extract=scope['material_features']
    locations={Path(i['file_name']).parent.name:DATA/Path(i['file_name']).parent for i in read(DATA/'labels/real_train.json')['images']}
    def rf_features(rgb,box):
        x,y,xx,yy=box;h,w=rgb.shape[:2];return extract(rgb[max(0,int(y)):min(h,math.ceil(yy)),max(0,int(x)):min(w,math.ceil(xx))])
    for part in ('train','val'):
        pack=read(VOL/f'{part}_frames.json'); ds=pd.read_csv(VOL/f'{part}_predictions.csv').set_index(['frame','det'],verify_integrity=True); actual=cache if part=='train' else pack['pred']; records=[]; joined=set()
        assert set(actual)==set(pack['truth'])==set(split[part]['image_ids'])
        for f,gt in pack['truth'].items():
            ps=actual[f]; geo=pack['pred'][f]; assert len(ps)==len(geo) and len(gt)<=1
            obj=gt[0]['object_id'] if gt else None; group=folds.get(obj,1) if part=='train' else 0
            if gt: assert gt[0]['take']==int(gt[0]['mass']<=500)
            matched={id(p):g for g,p in zip(gt,score._match(gt,ps),strict=True) if p is not None}
            if part=='train' and (gt or ps):
                path=locations[f]/'color.png'; path=path if path.exists() else path.with_suffix('.jpg'); rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
                if gt:
                    idx=boxedgt[f]; assert crops[idx]['object_id']==obj and np.allclose(crops[idx]['box'],gt[0]['box']); gt_rf[f]=rf_features(rgb,gt[0]['box']); gt_folds[f]=group; labels[f]=gt[0]['material']
            for j,(p,gp) in enumerate(zip(ps,geo,strict=True)):
                key=(f,j); joined.add(key); row=ds.loc[key]; assert np.allclose(p['box'],gp['box'],atol=.001,rtol=0) and abs(p['score']-gp['score'])<=1e-6 and p['form']==gp['form'] and np.isclose(p['volume'],row.base_ml)
                assert p['take']==int(p['mass']<=500); truth=matched.get(id(p)); detfold=group
                if part=='train':
                    idx=dets[key];assert np.allclose(crops[idx]['box'],p['box']);dino_x[key]=emb[idx];det_rf[key]=rf_features(rgb,p['box']); assert not gt or row['fold']==group
                records.append(dict(frame=f,det=j,group=detfold,object_id=obj,y=truth['take'] if truth else np.nan,base_ml=row.base_ml,corrected_ml=row.corrected_ml,mass=p['mass'],A=p['take'],solid=row.solid,form=p['form'],projected_m2=row.projected_m2))
        df=pd.DataFrame(records); assert joined==set(ds.index);tables[part]=df
    tr,va=tables['train'],tables['val']; assert set(dets)==set(dino_x) and set(dv)==set(zip(va.frame,va.det)); den=pd.read_csv(DATA/'densities.csv').set_index(['form','material']).density_g_per_ml.to_dict(); params=read(OLD/'protocol.json')['rf_params']; params['n_jobs']=1
    def light_outputs(fit_groups,subset,C,with_rf):
        fs=[f for f in boxedgt if gt_folds[f] in fit_groups]; xs=emb[[boxedgt[f] for f in fs]]; ys=np.asarray([labels[f] for f in fs]); keys=list(zip(subset.frame,subset.det)); assert set(subset.group).isdisjoint(fit_groups)
        head_lr=LogisticRegression(C=C,max_iter=2000,class_weight='balanced',solver='lbfgs',tol=1e-6).fit(xs,ys); proba=np.zeros((len(keys),len(CLASSES))); proba[:,[CLASSES.index(c) for c in head_lr.classes_]]=head_lr.predict_proba(np.stack([dino_x[k] for k in keys])); assert np.allclose(proba.sum(axis=1),1)
        result=subset.copy();result[['p_'+c for c in CLASSES]]=proba
        if with_rf:
            rf=RandomForestClassifier(**params).fit([gt_rf[f] for f in fs],ys); cats=rf.predict([det_rf[k] for k in keys]); result['mass']=[r.base_ml*den.get((r.form,c),1.0) for r,c in zip(result.itertuples(),cats,strict=True)]
        return result
    with threadpool_limits(limits=1):
        chunks=[]
        for k in blocks:
            pred=light_outputs(set(blocks)-{k},tr[tr.group==k],foldinfo[k]['C'],False);chunks.append(pred)
            inner=[]
            for h in sorted(set(blocks)-{k}):
                inner.append(light_outputs(set(blocks)-{k,h},tr[(tr.group==h)&tr.y.notna()],foldinfo[k]['C'],True))
            pd.concat(inner).to_csv(OUT/f'nested_fit_{k}.csv',index=False)
        tr=pd.concat(chunks).sort_index(); va[['p_'+c for c in CLASSES]]=np.array([probs_val[dv[k]] for k in zip(va.frame,va.det)])
    for frame,ref,name in [(tr,prev_train,'train'),(va,prev_val,'val')]:
        hits=frame[frame.y.notna()]; winners=hits[['p_'+c for c in CLASSES]].idxmax(axis=1).str[2:]; assert all(w==ref[f] for f,w in zip(hits.frame,winners)), (name,[(f,w,ref[f]) for f,w in zip(hits.frame,winners) if w!=ref[f]][:15]); assert all(ref[f]=='miss' for f in set(ref)-set(hits.frame)); frame.to_csv(OUT/f'{name}_features.csv',index=False)
    print('verified detector rows',len(tr),len(va),'matched',tr.y.notna().sum(),va.y.notna().sum(),flush=True)

if __name__=='__main__':
    prepare()
