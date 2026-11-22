from common import *
from catboost import CatBoostRegressor
import pandas as pd
from collections import Counter

CONFIG=dict(iterations=120,depth=3,learning_rate=.04,l2_leaf_reg=8,random_seed=2026,loss_function='RMSE',thread_count=2,task_type='CPU',allow_writing_files=False,verbose=False)

def fit(rows):
    rows=rows.loc[(rows.base_fallback==0)&(rows.target>0)].copy();counts=rows.object_id.value_counts(); weights=1/rows.object_id.map(counts);weights/=weights.mean();model=CatBoostRegressor(**CONFIG);model.fit(rows[FEATURES],np.log(rows.target/rows.base_ml),sample_weight=weights);return model

def predict(model,df):
    if df.empty:return np.empty(0)
    log_c=model.predict(df[FEATURES]);base=df.base_ml.to_numpy();good=(df.base_fallback==0)&np.isfinite(log_c)&np.isfinite(base)&(base>0); result=np.where(good,base*np.exp(np.clip(log_c,-8,8)),np.where(np.isfinite(base)&(base>0),base,420.));assert np.isfinite(result).all() and (result>0).all();return result

def score_set(frames, estimates, ids):
    gt={f:frames['truth'][f] for f in ids};a={f:frames['pred'][f] for f in ids};b={f:[dict(p,volume=float(estimates[(f,j)])) for j,p in enumerate(a[f])] for f in ids}; A=metric.components(gt,a);B=metric.components(gt,b);assert all(A[k]==B[k] for k in ['D','M','S']);return A['V'],B['V']

def quality(v,t):return np.maximum(0.,1-np.abs(np.log(v/t))/np.log(3))

def evaluate(part,frames,ds,matched,fold_info):
    indexed=ds.set_index(['frame','det']);ests=indexed.corrected_ml.to_dict(); ids=list(frames['truth']);a,b=score_set(frames,ests,ids);r=matched.merge(ds[['frame','det','base_ml','corrected_ml','missing_fraction','outlier_fraction','compactness']],on=['frame','det'],how='left',validate='one_to_one'); r['ratio_A']=r.base_ml/r.target;r['ratio_B']=r.corrected_ml/r.target
    for name in ['A','B']:
        r['V_'+name]=quality(r['ratio_'+name],1).fillna(0);r['error_'+name]=(r['ratio_'+name]-1).abs()
    assert np.allclose([a,b],[r.V_A.mean(),r.V_B.mean()]);r['delta']=r.V_B-r.V_A
    g=r.groupby('object_id').agg(frames=('frame','size'),matches=('base_ml','count'),material=('material','first'),mass_g=('mass','first'),V_A=('V_A','mean'),V_B=('V_B','mean'),ratio_A=('ratio_A','median'),ratio_B=('ratio_B','median'),error_A=('error_A','median'),error_B=('error_B','median'),sum_delta=('delta','sum'));g['delta']=g.V_B-g.V_A;g['leave_one_out_delta']=(r.delta.sum()-g.sum_delta)/(len(r)-g.frames);g.to_csv(HERE/f'{part}_objects.csv');r.to_csv(HERE/f'{part}_annotations.csv',index=False);ds.to_csv(HERE/f'{part}_predictions.csv',index=False)
    rng=np.random.default_rng(2026); sample=rng.integers(len(g),size=(10000,len(g))); boot=g.sum_delta.to_numpy()[sample].sum(1)/g.frames.to_numpy()[sample].sum(1)
    return dict(A=a,B=b,delta=b-a,median_relative_error=[float(r.error_A.median()),float(r.error_B.median())],improved=int((g.delta>1e-9).sum()),worse=int((g.delta< -1e-9).sum()),ties=int((g.delta.abs()<=1e-9).sum()),macro_delta=float(g.delta.mean()),leave_one_out_range=[float(g.leave_one_out_delta.min()),float(g.leave_one_out_delta.max())],cluster_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),folds=fold_info)

def main(part):
    frames=load(HERE/f'{part}_frames.json');ds=pd.read_csv(HERE/f'{part}_features.csv');matched=pd.read_csv(HERE/f'{part}_matches.csv'); fold_info=[]; splits=load(SPLIT);assert set(splits['train']['object_ids']).isdisjoint(splits['val']['object_ids']); held=load(FOLDS)['folds'];oidfold={o:f['fold'] for f in held for o in f['held']};assert len(held)==4 and sum(len(f['held']) for f in held)==len(oidfold)==32 and set(oidfold)==set(splits['train']['object_ids'])
    if part=='train':
        save(HERE/'configuration.json',dict(catboost=CONFIG,features=FEATURES,held={str(f['fold']):f['held'] for f in held},fold_source_sha256=sha(FOLDS),weighting='equal total weight per matched physical object',target='log(true_volume/base_ml)'))
        rows=matched.merge(ds,on=['frame','det'],validate='one_to_one');framefold={}
        for fid,grp in matched.groupby('frame'):assert grp.object_id.map(oidfold).nunique()==1;framefold[fid]=oidfold[grp.object_id.iloc[0]]
        for j,fid in enumerate(sorted(set(frames['truth'])-set(framefold))):framefold[fid]=held[j%4]['fold']
        ds['fold']=ds.frame.map(framefold); ds['corrected_ml']=np.nan
        for f in held:
            train=rows[~rows.object_id.isin(f['held'])];assert set(train.object_id)<=set(f['fit_objects']); model=fit(train);flag=ds.fold==f['fold'];ds.loc[flag,'corrected_ml']=predict(model,ds[flag]);ids=[i for i in frames['truth'] if framefold[i]==f['fold']];a,b=score_set(frames,ds[flag].set_index(['frame','det']).corrected_ml.to_dict(),ids);fold_info.append(dict(fold=f['fold'],fit_objects=int(train.object_id.nunique()),A=a,B=b,delta=b-a));
        assert ds.corrected_ml.notna().all();result=evaluate(part,frames,ds,matched,fold_info);model=fit(rows);model.save_model(str(HERE/'volume_train32.cbm'))
    else:
        model=CatBoostRegressor();model.load_model(str(HERE/'volume_train32.cbm'));ds['corrected_ml']=predict(model,ds);result=evaluate(part,frames,ds,matched,[]);alt=pd.read_csv(HERE/'val_mask_sensitivity.csv');sens={}
        for op,d in alt.groupby('op'):
            est=dict(zip(zip(d.frame,d.det),predict(model,d)));sens[op]=score_set(frames,est,list(frames['truth']))[1]
        result['mask_sensitivity_V']=sens
    save(HERE/f'{part}_metrics.json',result);print(part,json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
