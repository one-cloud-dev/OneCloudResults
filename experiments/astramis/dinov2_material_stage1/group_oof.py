from pathlib import Path
from collections import defaultdict, Counter
import copy, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits
from extract_embeddings import ROOT, OUT, DATA, OLD, read, save
sys.path.insert(0,str(ROOT/'datasets'))
import score as official

CLASSES=["metal","wood","plastic","glass","paper","mineral"]
assert set(CLASSES)==set(official.MATERIALS)
GRID=[.0001,.001,.01,.1,1.0]

def train(X,y,C):
    return LogisticRegression(C=C, max_iter=2000, class_weight='balanced', solver='lbfgs',tol=1e-6).fit(X,y)

def tune(X,y,g):
    # Split the distinct physical objects within each material. Singletons stay in inner-fit.
    folds=[[] for _ in range(3)]; rng=np.random.default_rng(2718)
    for label in sorted(set(y)):
        ids=np.unique(g[y==label]);rng.shuffle(ids)
        if len(ids)>1:
            for j,obj in enumerate(ids): folds[j%3].append(str(obj))
    scores={}
    for c in GRID:
        yt=[];yp=[]
        for held in folds:
            test=np.isin(g,held)
            if not test.any():continue
            assert set(g[test]).isdisjoint(g[~test]); assert set(y[~test])==set(y)
            pred=train(X[~test],y[~test],c).predict(X[test]);yt.extend(y[test]);yp.extend(pred)
        scores[c]=float(recall_score(yt,yp,labels=sorted(set(yt)),average='macro',zero_division=0))
    best=max(GRID,key=lambda c:scores[c]);return best,dict(C=best,scores=scores,inner_validation_objects=folds)

def metrics(yt,yp):
    return dict(n=len(yt),accuracy=float(accuracy_score(yt,yp)),macro_f1=float(f1_score(yt,yp,labels=CLASSES,average='macro',zero_division=0)),confusion=confusion_matrix(yt,yp,labels=CLASSES+['miss']).tolist()[:len(CLASSES)],recall={m:float(np.mean(np.array(yp)[np.array(yt)==m]==m)) for m in CLASSES if m in yt})

def head(name,model,objects,c):
    np.savez(OUT/name,weight=model.coef_,bias=model.intercept_,classes=model.classes_,fit_objects=np.unique(objects),C=c)
    with np.load(OUT/name) as z:
        assert np.array_equal(model.predict(X),z['classes'][(X@z['weight'].T+z['bias']).argmax(axis=1)])

def run():
    global X
    rows=read(OUT/'crops.json');X=np.load(OUT/'embeddings.npy'); split=read(ROOT/'baseline_v0/split_v0.json')
    anns=read(DATA/'labels/real_train.json'); ims={i['id']:Path(i['file_name']).parent.name for i in anns['images']}
    truth={f:[] for f in split['train']['image_ids']}; validation=set(split['val']['image_ids'])
    for a in anns['annotations']:
        f=ims[a['image_id']]; x,y,w,h=a['bbox']; v=a['attributes']
        if f in truth:truth[f].append(dict(box=[x,y,x+w,y+h],form=v['form'],material=v['material'],volume=v['volume_ml'],take=v['take']))
    for f in truth: assert len(truth[f])<=1
    base=read(OLD/'oof_predictions.json')['A']; assert set(base)==set(truth)
    candidate=copy.deepcopy(base); gtrows=[i for i,r in enumerate(rows) if r['kind']=='gt']; tidx=np.array([i for i in gtrows if rows[i]['frame'] in truth]); vidx=np.array([i for i in gtrows if rows[i]['frame'] in validation]); assert len(tidx)+len(vidx)==len(gtrows)
    fn=np.array([rows[i]['frame'] for i in tidx]); y=np.array([rows[i]['material'] for i in tidx]); g=np.array([rows[i]['object_id'] for i in tidx]); frame_obj=dict(zip(fn,g));detidx={(r['frame'],r['index']):i for i,r in enumerate(rows) if r['kind']=='det'}
    heldfolds=read(ROOT/'runs/take_probability_20261121_e4374505/train.json')['folds']; gtout={}; foldinfo=[]; done=set()
    for d in heldfolds:
        fold=d['fold']; held=d['held_object_ids']; valid=np.isin(g,held); fit=~valid
        assert set(g[valid])==set(held) and set(g[valid]).isdisjoint(done) and set(y[fit])==set(CLASSES);done.update(held)
        C,info=tune(X[tidx[fit]],y[fit],g[fit]); model=train(X[tidx[fit]],y[fit],C)
        gtout.update(zip(fn[valid],model.predict(X[tidx[valid]])))
        frames=set(fn[valid]); frames.update(f for f in truth if not truth[f] and fold==1)
        for f in frames:
            for j,p in enumerate(candidate[f]):p['material']=str(model.predict(X[[detidx[f,j]]])[0])
        targets={f:truth[f] for f in frames}; info.update(fold=fold,held=held,fit_objects=sorted(set(g[fit])),support=dict(Counter(y[valid])),M_baseline=official.components(targets,{f:base[f] for f in frames})['M'],M_dino=official.components(targets,{f:candidate[f] for f in frames})['M']);foldinfo.append(info)
    assert done==set(g) and set(gtout)==set(fn) and set(g).isdisjoint({rows[i]['object_id'] for i in vidx})
    records=[]
    for f,t,obj in zip(fn,y,g):
        matched=lambda p:official._match(truth[f],p[f])[0]
        b=matched(base); c=matched(candidate)
        records.append(dict(frame=str(f),object_id=str(obj),truth=str(t),GT_DINO=str(gtout[f]),det_DINO=c['material'] if c else 'miss',det_RF=b['material'] if b else 'miss'))
    summaries={k:metrics(list(y),[r[k] for r in records]) for k in ['GT_DINO','det_DINO','det_RF']}
    for key,p in [('det_DINO',candidate),('det_RF',base)]:summaries[key]['M']=official.components(truth,p)['M']
    assert abs(summaries['det_RF']['M']-read(OLD/'results.json')['metrics']['A']['M'])<1e-10
    by_object={}
    for obj in sorted(set(g)):
        sub=[r for r in records if r['object_id']==obj]; by_object[obj]={k:sum(r[k]==r['truth'] for r in sub)/len(sub) for k in summaries};by_object[obj]['n']=len(sub)
    C,train_info=tune(X[tidx],y,g); model=train(X[tidx],y,C);head('head_train32.npz',model,g,C)
    yv=[rows[i]['material'] for i in vidx]; pv=list(model.predict(X[vidx])); hv=dict(n=len(yv),accuracy=float(accuracy_score(yv,pv)),support=dict(Counter(yv)),recall={m:np.mean(np.array(pv)[np.array(yv)==m]==m) for m in sorted(set(yv))},macro_f1_present=float(f1_score(yv,pv,labels=sorted(set(yv)),average='macro',zero_division=0)))
    all_y=np.array([rows[i]['material'] for i in gtrows]); all_g=np.array([rows[i]['object_id'] for i in gtrows]); finalC,final_info=tune(X[gtrows],all_y,all_g); final=train(X[gtrows],all_y,finalC);head('head_all40.npz',final,all_g,finalC)
    result=dict(classes=CLASSES,confusion_columns=CLASSES+['miss'],folds=foldinfo,OOF=summaries,objects=by_object,heldout_GT=hv,train32=train_info,final40=final_info)
    save(OUT/'results.json',result);save(OUT/'oof_rows.json',records);save(OUT/'heldout_GT_rows.json',[dict(frame=rows[i]['frame'],object_id=rows[i]['object_id'],truth=t,predicted=p) for i,t,p in zip(vidx,yv,pv)])
    print('classes',CLASSES,'OOF',summaries,'folds',[(i['M_baseline'],i['M_dino']) for i in foldinfo], 'heldout',hv,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): run()
