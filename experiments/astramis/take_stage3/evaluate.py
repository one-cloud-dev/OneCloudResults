from prepare_features import OUT, VOL, OLD, ROOT, CLASSES, COLS, read, score
import copy
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def matrix(df, volume):
    x=df.assign(log_volume=np.log1p(df[volume]),log_mass=np.log1p(df.mass),mass_over_limit=df.mass/500)[COLS]
    assert np.isfinite(x).all().all()
    return x

def fit(df,volume):
    d=df[df.y.notna()]; assert set(d.y)=={0,1}
    return make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-7)).fit(matrix(d,volume),d.y.astype(int))

def evaluate(df,part,label):
    name=('oof' if part=='train' else 'val')+'_'+label
    pack=read(VOL/f'{part}_frames.json'); truth=pack['truth']; src=read(OLD/'oof_predictions.json')['A'] if part=='train' else pack['pred']; a=copy.deepcopy(src); b=copy.deepcopy(src)
    assert not df.duplicated(['frame','det']).any() and len(df)==sum(map(len,src.values())); assert df.prob.between(0,1).all()
    for r in df.itertuples():
        x=a[r.frame][r.det]; assert x['take']==r.A; x.update(material=getattr(r,'predicted_material'),volume=r.corrected_ml); b[r.frame][r.det]={**x,'take':int(r.prob>0.5)}
    ma=score.components(truth,a);mb=score.components(truth,b);assert all(ma[k]==mb[k] for k in ('D','M','V'))
    rows=[]
    for f,gt in truth.items():
        for g,pa,pb in zip(gt,score._match(gt,a[f]),score._match(gt,b[f]),strict=True):
            x=pa['take'] if pa else -1; y=pb['take'] if pb else -1; rows.append(dict(frame=f,object_id=g['object_id'],true_mass=g['mass'],y=g['take'],A=x,B=y,matched=(pa is not None),hit_A=int(x==g['take']),hit_B=int(y==g['take'])))
    z=pd.DataFrame(rows);z.to_csv(OUT/(name+'_annotations.csv'),index=False); df.assign(B=(df.prob>.5).astype(int)).to_csv(OUT/(name+'_predictions.csv'),index=False)
    groups=[]
    for obj,s in z.groupby('object_id'):
        ids=set(s.frame);left={f:g for f,g in truth.items() if f not in ids};la=score.components(left,a)['S'];lb=score.components(left,b)['S']; assert s.y.nunique()==1
        row=dict(object_id=obj,mass_g=s.true_mass.iloc[0],take=int(s.y.iloc[0]),n=len(s),miss=int((~s.matched).sum()),correct_A=int(s.hit_A.sum()),correct_B=int(s.hit_B.sum()),change=int(s.hit_B.sum()-s.hit_A.sum()),leave_out_A=la,leave_out_B=lb,leave_out_delta=lb-la)
        for k in ('A','B'):
            row['FP_'+k]=int(((s.y==0)&(s[k]==1)).sum());row['FN_'+k]=int(((s.y==1)&(s[k]==0)).sum())
        groups.append(row)
    objects=pd.DataFrame(groups); objects.to_csv(OUT/(name+'_objects.csv'),index=False); matched=z[z.matched]; heavy=z[z.y==0];light=z[z.y==1]; result=dict(A=ma,B=mb,delta=mb['S']-ma['S'],objects=[int((objects.change>0).sum()),int((objects.change<0).sum()),int((objects.change==0).sum())],class_frames=z.y.value_counts().to_dict(),class_objects=objects['take'].value_counts().to_dict(),miss=int((~z.matched).sum()))
    # Contribution to the pooled S with original class denominators.
    weights={i:1/(2*sum(z.y==i)) for i in [0,1]}; best=objects.assign(contribution=objects.change*objects['take'].map(weights)).sort_values(['contribution','object_id'],ascending=[False,True]).iloc[0];result.update(best_object=best.object_id,best_contribution=float(best.contribution),without_best=float(best.leave_out_delta),minimum_leave_out=float(objects.leave_out_delta.min()))
    for k in ('A','B'):
        result[k].update(accuracy_matched=accuracy_score(matched.y,matched[k]),balanced_accuracy_matched=balanced_accuracy_score(matched.y,matched[k]),accuracy_all=float(z['hit_'+k].mean()),FP=int(((matched.y==0)&(matched[k]==1)).sum()),FN=int(((matched.y==1)&(matched[k]==0)).sum()),heavy_hits=int(heavy['hit_'+k].sum()),light_hits=int(light['hit_'+k].sum()))
    return result

def main():
    tr=pd.read_csv(OUT/'train_features.csv');va=pd.read_csv(OUT/'val_features.csv');result={}
    for label in ('strict','stack'):
        strict=label=='strict';volume='base_ml' if strict else 'corrected_ml'; pred=tr.copy();pred['prob']=np.nan
        for k in sorted(tr.group.unique()):
            train=pd.read_csv(OUT/f'nested_fit_{k}.csv') if strict else tr[tr.group!=k]; held=tr[tr.group==k];assert set(train.object_id.dropna()).isdisjoint(held.object_id.dropna());model=fit(train,volume);pred.loc[held.index,'prob']=model.predict_proba(matrix(held,'corrected_ml'))[:,1]
        final=fit(tr,volume);joblib.dump(final,OUT/f'take_train32_{label}.joblib'); independent=va.assign(prob=final.predict_proba(matrix(va,'corrected_ml'))[:,1]);assert set(tr.object_id.dropna()).isdisjoint(va.object_id.dropna())
        for part,df in [('train',pred),('val',independent)]:
            df['predicted_material']=df[['p_'+c for c in CLASSES]].idxmax(axis=1).str[2:];result[part+'_'+label]=evaluate(df,part,label)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print({k:[r['A']['S'],r['B']['S'],r['objects'],r['without_best']] for k,r in result.items()})

if __name__=='__main__':
    main()
