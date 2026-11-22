from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
import csv, hashlib, importlib.util, json, math, statistics as st, sys

W=Path('/home/jovyan/work'); R=W/'runs'; T=R/'take_stage3'; V=R/'rgbd_volume_stage2'; M=R/'dinov2_material_stage1'
O=R/'diagnostic_stage7'; O.mkdir(exist_ok=True)
C=T/'submission_take_stage3_diagnostic.csv'; L=W/'datasets/dataset/labels/real_train.json'; SC=W/'datasets/score.py'
H='2c55a9bf5bab0e9118c8ea7cbee69f0d392e47712f17c288d7bee034a19dafd1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def ix(p):
    data=rows(p); index={(r['frame'],int(r['det'])):r for r in data}
    assert len(index)==len(data),p
    return index
def write(name,data):
    if isinstance(data,list):
        with (O/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in data for k in r))); w.writeheader(); w.writerows(data)
    else: (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def close(a,b): return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9)
def avg(a):
    a=list(a); return st.mean(a) if a else None
def med(a):
    a=list(a); return st.median(a) if a else None
def signs(xs):
    xs=list(xs); return dict(better=sum(x>1e-9 for x in xs),worse=sum(x< -1e-9 for x in xs),tied=sum(abs(x)<=1e-9 for x in xs))
def dv(x,y): return max(0.,1-abs(math.log(x/y))/math.log(3)) if x>0 and y>0 else 0.
assert sha(C)==H==read(T/'integrity.json')['candidate_sha256']
manifest=read(V/'manifest.json')
old_path=R/'material_final_m_20261121_7f35d05_oof4/oof_predictions.json'
old_oof=read(old_path)['A']
assert sha(SC)==manifest['official_metric_sha256']==read(R/'detection_stage6/inputs.json')['score_hash']
spec=importlib.util.spec_from_file_location('astramis_official_score',SC); score=importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
ref=read(T/'metrics.json'); split=read(W/'baseline_v0/split_v0.json'); folds=read(V/'configuration.json')['held']; groups={obj:int(f) for f,ids in folds.items() for obj in ids}
assert set(groups)==set(split['train']['object_ids']) and len(groups)==32 and set(groups).isdisjoint(split['val']['object_ids'])
sol=score.labels_to_solution(L); official={f:score._parse(cell,False,f) for f,cell in zip(sol.image_id,sol.objects)}
labels=read(L); image_paths={im['id']:Path(im['file_name']).parent for im in labels['images']}
attrs={image_paths[ann['image_id']].name:ann['attributes'] for ann in labels['annotations']}
assert len(attrs)==214
matlabels=list(score.MATERIALS)
dino={r['frame']:r for r in read(M/'oof_rows.json')}; dval={r['frame']:r for r in read(M/'heldout_detector.json')['rows']}
mat_gtval={r['frame']:r for r in read(M/'heldout_GT_rows.json')}
summary={}; frames_all=[]; objs_all=[]; results=[]; conf_all=[]; det_all=[]; breakdown=[]; stability=[]
fields={'A_form_oracle':'form','B_material_oracle':'material','C_volume_oracle':'volume','D_take_oracle':'take'}
archive=read(R/'volume_stage7/results.json'); past={(z['split'],z['frame'],int(z['detection'])):z for z in rows(R/'volume_stage7/paired_predictions.csv')}

def outcome(gt,pr,base):
    s=score.components(gt,pr)
    return s, {k:s[k]-base[k] for k in s}
def metrics_row(tag,name,bas,res,scope='',object_id=''):
    return dict(split=tag,scenario=name,scope=scope,object_id=object_id,**res,**{'delta_'+k:res[k]-bas[k] for k in res})

def vstats(rs):
    return dict(frames=len(rs),objects=len({r['object_id'] for r in rs}),V_cat=avg(r['V_cat'] for r in rs),V_geom=avg(r['V_geom'] for r in rs),zero_V=sum(r['V_cat']==0 for r in rs),over=sum(r['ratio_cat']>1 for r in rs),under=sum(r['ratio_cat']<1 for r in rs),cat_worse=sum(r['V_cat']<r['V_geom']-1e-9 for r in rs),cat_better=sum(r['V_cat']>r['V_geom']+1e-9 for r in rs),median_ratio=med(r['ratio_cat'] for r in rs),median_rel=med(r['abs_relative_error'] for r in rs),median_abs_log=med(r['abs_log_error'] for r in rs),median_missing=med(r['missing_fraction'] for r in rs))

def material_stats(rs):
    ids=defaultdict(list)
    for r in rs: ids[r['object_id']].append(r)
    return dict(frames=len(rs),objects=len(ids),correct=sum(r['material_ok'] for r in rs),wrong_objects=sum(any(not r['material_ok'] for r in rr) for rr in ids.values()),pmax=avg(r['p_material_max'] for r in rs),margin=avg(r['p_material_margin'] for r in rs),iou=avg(r['iou'] for r in rs))

for tag,part,number,nd,ng in [('OOF','train',32,193,165),('val','val',8,57,49)]:
    packpath=V/f'{part}_frames.json'; geopath=V/f'{part}_predictions.csv'; matchpath=V/f'{part}_matches.csv'
    for p in [packpath,geopath,matchpath]: assert sha(p)==manifest['files'][p.name],p
    pack=read(packpath); gt=pack['truth']; pred=pack['pred']; gindex=ix(geopath); midx=ix(matchpath)
    source=old_oof if part=='train' else deepcopy(pred)
    assert set(source)==set(pred)
    up=ix(T/('oof_strict_predictions.csv' if part=='train' else 'val_strict_predictions.csv'))
    keys={(f,j) for f,ds in pred.items() for j in range(len(ds))}
    assert set(gt)==set(pred)==set(split[part]['image_ids']) and keys==set(up)==set(gindex) and len(keys)==nd and len(midx)==ng
    match={}; objframes=defaultdict(set); rr=[]; dd=[]
    for f,gs in gt.items():
        assert len(gs)==len(official[f])<=1
        for g,q in zip(gs,official[f],strict=True):
            assert all(g[k]==q[k] for k in ('form','material','volume','take')) and all(abs(x-y)<=0.051 for x,y in zip(g['box'],q['box']))
            assert g['object_id']==attrs[f]['object_id'] and g['mass']==attrs[f]['mass_g']; objframes[g['object_id']].add(f)
        for j,p in enumerate(pred[f]):
            key=(f,j); u=up[key]; z=gindex[key]
            assert p['form']==u['form'] and close(u['corrected_ml'],z['corrected_ml']) and close(p['volume'],u['base_ml']) and close(p['volume'],z['base_ml']) and (part=='train' or close(p['mass'],u['mass']))
            src=source[f][j]
            assert p['form']==src['form'] and abs(p['score']-src['score'])<1e-6 and max(abs(x-y) for x,y in zip(p['box'],src['box']))<.001 and close(src['mass'],u['mass'])
            assert (src['take']==int(u['A'])) and (float(u['prob'])>.5)==bool(int(u['B'])) and int(float(u['mass'])<=500)==int(u['A'])
            p.update(material=u['predicted_material'],volume=float(u['corrected_ml']),take=int(u['B']))
        for g,p in zip(gs,score._match(gs,pred[f]),strict=True):
            if p is None: continue
            j=next(j for j,x in enumerate(pred[f]) if x is p); key=(f,j); u=up[key]; z=gindex[key]; o=g['object_id']
            match[f]=j; target=g['volume']; cat=p['volume']; geom=float(z['base_ml']); prob=float(u['prob'])
            assert midx[key]['object_id']==u['object_id']==o and close(midx[key]['target'],target)
            if part=='train': assert int(u['group'])==int(z['fold'])==groups[o]
            logits={k:float(u['p_'+k]) for k in matlabels}; top=sorted(logits,key=logits.get,reverse=True)
            assert p['material']==top[0] and math.isclose(sum(logits.values()),1,abs_tol=1e-6)
            prev=(dino if part=='train' else dval)[f]
            assert prev['truth']==g['material'] and prev['object_id']==o and prev['det_DINO' if part=='train' else 'DINO']==p['material']
            mgt=prev['GT_DINO'] if part=='train' else mat_gtval[f]['predicted']
            width=p['box'][2]-p['box'][0]; height=p['box'][3]-p['box'][1]
            rr.append(dict(split=tag,frame=f,det=j,object_id=o,fold=groups.get(o,0),form=g['form'],pred_form=p['form'],material=g['material'],pred_material=p['material'],gt_crop_material=mgt,material_ok=p['material']==g['material'],p_material_max=logits[top[0]],p_material_margin=logits[top[0]]-logits[top[1]],material_second=top[1],**{'p_material_'+k:v for k,v in logits.items()},crop_area=width*height,crop_width=width,crop_height=height,iou=score._iou(g['box'],p['box']),confidence=p['score'],volume_true_ml=target,volume_cat_ml=cat,volume_geom_ml=geom,ratio_cat=cat/target,ratio_geom=geom/target,relative_error=cat/target-1,abs_relative_error=abs(cat/target-1),log_error=math.log(cat/target),abs_log_error=abs(math.log(cat/target)),V_cat=dv(cat,target),V_geom=dv(geom,target),missing_fraction=float(z['missing_fraction']),outlier_fraction=float(z['outlier_fraction']),depth_noise_m=float(z['depth_noise_m']),projected_m2=float(z['projected_m2']),mask_px=float(z['mask_px']),approx_missing_mask_px=round(float(z['missing_fraction'])*float(z['mask_px'])),q50_m=float(z['median_m']),q95_m=float(z['q95_m']),q99_m=float(z['q99_m']),robust_ml=float(z['robust_ml']),compactness=float(z['compactness']),fill_ratio=float(z['fill_ratio']),geometry_fallback=bool(int(float(z['base_fallback']))),mass_true_g=g['mass'],mass_csv_g=float(u['mass']),mass_csv_minus_500_g=float(u['mass'])-500,take_mass_rule=int(float(u['mass'])<=500),take_gt=g['take'],take_B=p['take'],p_take=prob,take_margin=abs(prob-.5),take_error='heavy_pick' if g['take']==0 and p['take']==1 else ('light_refusal' if g['take']==1 and p['take']==0 else ''),lighting=read(W/'datasets/dataset'/Path('real/train')/f/'meta.json').get('lighting','')))
        for j,p in enumerate(pred[f]):
            best=max((score._iou(g['box'],p['box']) for g in gs),default=0.)
            cls_tp=any(z is p for z in score._match([g for g in gs if g['form']==p['form']],[x for x in pred[f] if x['form']==p['form']]))
            dd.append(dict(split=tag,frame=f,det=j,object_id=gs[0]['object_id'] if gs else '',form=p['form'],confidence=p['score'],iou=best,matched=match.get(f)==j,tp_D=cls_tp,kind='TP' if cls_tp else ('empty_FP' if not gs else ('localization_FP' if best<.5 else ('wrong_form' if p['form']!=gs[0]['form'] else 'duplicate_FP')))))
    assert len(objframes)==number and len(match)==len(rr)==ng and set(match)==set(attrs)&set(gt)
    assert {(f,j) for f,j in match.items()}==set(midx)
    baseline=score.components(gt,pred)
    assert all(close(v,ref[part+'_strict']['B'][k]) for k,v in baseline.items())
    gt_round={f:official[f] for f in gt}; rounded=score.components(gt_round,pred)
    assert all(close(v,rounded[k]) for k,v in baseline.items())
    results.append(metrics_row(tag,'baseline',baseline,baseline))
    all_variants={}
    for name,field in fields.items():
        q=deepcopy(pred)
        for f,j in match.items(): q[f][j][field]=gt[f][0][field]
        all_variants[name]=q
    q=deepcopy(pred)
    for d in dd:
        if d['kind']=='empty_FP': q[d['frame']]=[]
    all_variants['remove_empty_FP_oracle']=q
    # In a labelled one-object frame, only detections that cannot be AP true positives are removed.
    # Keep the class-agnostic match even if its form is wrong, to preserve M/V/S.
    all_variants['remove_extra_FP_oracle']={f:[p for j,p in enumerate(ds) if j==match.get(f) or any(d['tp_D'] for d in dd if d['frame']==f and d['det']==j)] for f,ds in pred.items()}
    q=deepcopy(pred)
    for f,j in match.items(): q[f][j]['box']=gt[f][0]['box']
    all_variants['matched_bbox_oracle']=q
    all_variants['D_perfect_oracle']={f:[dict(pred[f][match[f]],box=gs[0]['box'],form=gs[0]['form'])] if gs else [] for f,gs in gt.items()}
    all_variants['geometry_archived']={f:[dict(p,volume=float(gindex[f,j]['base_ml'])) for j,p in enumerate(ds)] for f,ds in pred.items()}
    all_variants['blend_50_archived']={f:[dict(p,volume=float(past[tag,f,j]['volume_B_ml'])) for j,p in enumerate(ds)] for f,ds in pred.items()}
    q=deepcopy(pred)
    for r in rr: q[r['frame']][r['det']]['material']=r['gt_crop_material'] or r['pred_material']
    assert all(r['gt_crop_material'] in matlabels for r in rr),[(r['frame'],r['gt_crop_material']) for r in rr if r['gt_crop_material'] not in matlabels]
    all_variants['DINO_GT_crop_archived']=q
    all_variants['take_mass_archived']={f:[dict(p,take=int(up[f,j]['A'])) for j,p in enumerate(ds)] for f,ds in pred.items()}
    q=deepcopy(pred)
    for f,j in match.items():
        for k in ('material','volume','take'): q[f][j][k]=gt[f][0][k]
        q[f][j]['form']=gt[f][0]['form']
    all_variants['ABCD_matched_oracle']=q
    for name,q in all_variants.items():
        s,delta=outcome(gt,q,baseline)
        changed={'A_form_oracle':'D','B_material_oracle':'M','C_volume_oracle':'V','D_take_oracle':'S','remove_empty_FP_oracle':'D','remove_extra_FP_oracle':'D','matched_bbox_oracle':'D','D_perfect_oracle':'D','geometry_archived':'V','blend_50_archived':'V','DINO_GT_crop_archived':'M','take_mass_archived':'S'}.get(name)
        if changed: assert all(close(delta[k],0) for k in ('D','M','V','S') if k!=changed),(name,delta)
        if name=='blend_50_archived': assert all(close(x,archive['splits'][tag]['B'][k]) for k,x in s.items())
        results.append(metrics_row(tag,name,baseline,s,'all_frames'))
        own_deltas={}; omitted={}
        for o,ff in objframes.items():
            keep={f:g for f,g in gt.items() if f not in ff}; own={f:g for f,g in gt.items() if f in ff}; b0=score.components(keep,{f:pred[f] for f in keep}); w0=score.components(own,{f:pred[f] for f in own}); wp=score.components(own,{f:q[f] for f in own})
            omitted[o]=score.components(keep,{f:q[f] for f in keep})['Score']-b0['Score']
            own_deltas[o]=wp['Score']-w0['Score']
        # For AP and macro metrics, rank influence by the full-score gain lost on removal.
        top=min(omitted,key=omitted.get)
        stability.append(dict(split=tag,scenario=name,**signs(own_deltas.values()),strongest_object=top,score_without_strongest=omitted[top],max_LOO=max(omitted.values()),single_object_delta=own_deltas[top],loo_by_object=json.dumps(omitted,sort_keys=True)))
        if name in fields or name=='ABCD_matched_oracle':
            for o,ff in sorted(objframes.items()):
                one={f:q[f] if f in ff else ds for f,ds in pred.items()}
                so=score.components(gt,one)
                results.append(metrics_row(tag,'E_'+name,baseline,so,'one_object',o))
    byobj={o:[r for r in rr if r['object_id']==o] for o in objframes}
    base_cat=baseline['V']; prior_form=Counter((r['form'],r['pred_form']) for r in rr)
    for o,rs in sorted(byobj.items()):
        z=rs[0]; vs=vstats(rs); cnt=Counter(r['pred_material'] for r in rs); tops=cnt.most_common(); gtmat=z['material']; gt_crop=[r['gt_crop_material']==gtmat for r in rs]
        own_take=Counter(r['take_error'] for r in rs if r['take_error']); own={f:g for f,g in gt.items() if f in objframes[o]}
        same=[r for r in results if r['split']==tag and r['object_id']==o]
        obj= dict(split=tag,object_id=o,fold=z['fold'],true_form=z['form'],true_material=gtmat,mass_g=z['mass_true_g'],true_volume_ml=z['volume_true_ml'],**vs,median_geom_ml=med(r['volume_geom_ml'] for r in rs),median_cat_ml=med(r['volume_cat_ml'] for r in rs),median_geom_ratio=med(r['ratio_geom'] for r in rs),V_score_loss=sum(1-r['V_cat'] for r in rs)*25/ng,volume_reversal_to_geom_gain=sum(r['V_geom']-r['V_cat'] for r in rs)*25/ng,median_iou=med(r['iou'] for r in rs),bbox_iou_min=min(r['iou'] for r in rs),form_wrong=sum(r['form']!=r['pred_form'] for r in rs),extra_D_FP=sum(d['kind'] in ('localization_FP','duplicate_FP') and d['object_id']==o for d in dd),mat_correct=sum(r['material_ok'] for r in rs),mat_gt_crop_correct=sum(gt_crop),mat_majority=tops[0][0] if len(tops)==1 or tops[0][1]>tops[1][1] else 'tie',mat_votes=json.dumps(cnt,sort_keys=True),mat_maxprob=med(r['p_material_max'] for r in rs),mat_margin=med(r['p_material_margin'] for r in rs),heavy_pick=own_take['heavy_pick'],light_refusal=own_take['light_refusal'],take_mass_disagree=sum(r['take_mass_rule']!=r['take_B'] for r in rs),median_mass_csv_minus_500=med(r['mass_csv_minus_500_g'] for r in rs),median_p_take=med(r['p_take'] for r in rs),**{r['scenario'].removeprefix('E_')+'_score':r['delta_Score'] for r in same})
        objs_all.append(obj)
    matgroups=defaultdict(list); vgroups=defaultdict(list)
    oof_crop_median=med(r['crop_area'] for r in rr) if tag=='OOF' else oof_crop_median
    for r in rr:
        for label in ['all','form:'+r['form'],'material:'+r['material'],'form/material:'+r['form']+'/'+r['material'],'crop:'+('smaller' if r['crop_area']<oof_crop_median else 'larger'),'lighting:'+str(r['lighting']),'iou:'+('below_0.75' if r['iou']<.75 else '0.75_plus'),'outcome:'+('hit' if r['material_ok'] else 'error')]: matgroups[label].append(r)
        for label in ['all','form:'+r['form'],'material:'+r['material'],'form/material:'+r['form']+'/'+r['material'],'missing:'+('<10%' if r['missing_fraction']<.1 else ('10–<50%' if r['missing_fraction']<.5 else '≥50%')),'true_volume:'+('<150' if r['volume_true_ml']<150 else ('150–499' if r['volume_true_ml']<500 else '≥500')),'geometry_error:'+('cat_worse' if r['V_cat']<r['V_geom']-1e-9 else ('cat_better' if r['V_cat']>r['V_geom']+1e-9 else 'tie')),'volume_direction:'+('over' if r['ratio_cat']>1 else ('under' if r['ratio_cat']<1 else 'exact'))]: vgroups[label].append(r)
    for label,rs in sorted(matgroups.items()): breakdown.append(dict(split=tag,component='M',group=label,**material_stats(rs)))
    for label,rs in sorted(vgroups.items()): breakdown.append(dict(split=tag,component='V',group=label,**vstats(rs)))
    objects_now=[x for x in objs_all if x['split']==tag]
    for unit, pairs in [('frame',[(r['material'],r['pred_material']) for r in rr]),('object_plurality',[(o['true_material'],o['mat_majority']) for o in objects_now])]:
        cc=Counter(pairs)
        for t in matlabels:
            for p in [*matlabels]+(['tie'] if unit=='object_plurality' else []): conf_all.append(dict(split=tag,unit=unit,truth=t,prediction=p,count=cc[t,p]))
    take_error=[r for r in rr if r['take_error']]; type_c=Counter(r['take_error'] for r in take_error)
    per_class={}
    for f in score.FORMS:
        ds=[r for r in dd if r['form']==f]; ts=[r for r in ds if r['tp_D']]; fs=[r for r in ds if not r['tp_D']]; maxfp=max((r['confidence'] for r in fs),default=0); mintp=min((r['confidence'] for r in ts),default=0)
        per_class[f]=dict(AP=score._average_precision(gt,pred,f),GT=sum(g['form']==f for gs in gt.values() for g in gs),TP=len(ts),FP=len(fs),TP_conf_min=mintp,FP_conf_max=maxfp,FP_above_lowest_TP=sum(r['confidence']>mintp for r in fs),TP_conf_median=med(r['confidence'] for r in ts),FP_conf_median=med(r['confidence'] for r in fs))
    summary[tag]=dict(frames=len(gt),detections=nd,objects=number,matched=ng,missed=ng-len(match),empty_frames=sum(not gs for gs in gt.values()),empty_FP=sum(d['kind']=='empty_FP' for d in dd),object_forms=dict(Counter(o['true_form'] for o in objects_now)),object_materials=dict(Counter(o['true_material'] for o in objects_now)),object_weights=dict(Counter('heavy' if o['mass_g']>500 else 'light' for o in objects_now)),frame_weights=dict(Counter('heavy' if r['mass_true_g']>500 else 'light' for r in rr)),D_kinds=dict(Counter(d['kind'] for d in dd)),D_by_form=per_class,agnostic_iou_min=min(r['iou'] for r in rr),agnostic_iou_median=med(r['iou'] for r in rr),form_confusions={a+'→'+b:n for (a,b),n in prior_form.items()},form_error_objects=sum(o['form_wrong']>0 for o in objects_now),material_wrong=sum(not r['material_ok'] for r in rr),material_wrong_objects=sum(o['mat_correct']<o['frames'] for o in objects_now),material_objects_wrong_in_all=sum(o['mat_correct']==0 for o in objects_now),V=vstats(rr),V_total_zero_objects=sum(o['zero_V']>0 for o in objects_now),V_median_ratio_objects=med(o['median_ratio'] for o in objects_now),V_median_signed_log=med(r['log_error'] for r in rr),V_geom_comparison_objects=signs(o['V_cat']-o['V_geom'] for o in objects_now),V_fallback_detections=sum(float(r['base_fallback'])!=0 for r in gindex.values()),worst_V_objects=[o['object_id'] for o in sorted(objects_now,key=lambda o:-o['V_score_loss'])[:8]],take_errors=dict(type_c),take_error_objects=sum(o['heavy_pick']+o['light_refusal']>0 for o in objects_now),take_error_median_probability=med(r['p_take'] for r in take_error),take_error_median_margin=med(r['take_margin'] for r in take_error),take_wrong_but_mass_right=sum(r['take_mass_rule']==r['take_gt'] for r in take_error),take_mass_disagree=sum(r['take_mass_rule']!=r['take_B'] for r in rr),take_mass_disagree_objects=sum(o['take_mass_disagree']>0 for o in objects_now),take_mass_wrong_but_lr_right=sum(r['take_mass_rule']!=r['take_gt'] and r['take_B']==r['take_gt'] for r in rr),take_errors_material_wrong=sum(not r['material_ok'] for r in take_error),take_errors_V0=sum(r['V_cat']==0 for r in take_error),take_error_median_abs_csv_mass_margin=med(abs(r['mass_csv_minus_500_g']) for r in take_error),geometry_stored_exact=True)
    frames_all.extend(rr); det_all.extend(dd)

sample=rows(W/'datasets/sample_submission.csv'); candidate=rows(C)
assert len(candidate)==137 and [r['image_id'] for r in candidate]==[r['image_id'] for r in sample]
parsed=[score._parse(r['prediction'],True,r['image_id']) for r in candidate]
assert sum(map(len,parsed))==144 and sum(not ds for ds in parsed)==19 and sha(C)==H
summary['candidate']={'sha256':H,'official_parser_rows':len(candidate),'det':sum(map(len,parsed)),'empty':sum(not ds for ds in parsed),'score_sha256':sha(SC),'public_score':78.2518,'crop_area_OOF_median':oof_crop_median}
for name,data in [('object_errors.csv',objs_all),('frame_errors.csv',frames_all),('take_errors.csv',[r for r in frames_all if r['take_error']]),('detection_errors.csv',det_all),('material_confusion.csv',conf_all),('counterfactuals.csv',results),('stability.csv',stability),('strata.csv',breakdown),('summary.json',summary)]: write(name,data)
print('OK',len(objs_all),len(frames_all),len(det_all),len(results),len(stability),sha(C))

import numpy as np, pandas as pd, joblib
from scipy.special import softmax
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(V/'deps')); from catboost import CatBoostRegressor
vc=pd.read_csv(V/'val_predictions.csv')
vv=pd.read_csv(T/'val_strict_predictions.csv')
cols=['log_volume','log_mass','mass_over_limit',*['p_'+x for x in sorted(score.MATERIALS)],'solid','projected_m2']
model=CatBoostRegressor(); model.load_model(str(V/'volume_train32.cbm'))
assert (vc[['frame','det']].to_numpy()==vv[['frame','det']].to_numpy()).all()
with threadpool_limits(limits=1):
    got=vc.base_ml*np.exp(np.clip(model.predict(vc[read(V/'configuration.json')['features']],thread_count=1),-8,8))
    caterr=float(np.max(np.abs(got-vv.corrected_ml)))
    lr=joblib.load(T/'take_train32_strict.joblib')
    x=vv.assign(log_volume=np.log1p(vv.corrected_ml),log_mass=np.log1p(vv.mass),mass_over_limit=vv.mass/500)[cols]
    lrerr=float(np.max(np.abs(lr.predict_proba(x)[:,1]-vv.prob)))
    mat_manifest=read(M/'manifest.json')['files'];
    for name in ('head_train32.npz','heldout_detector_embeddings.npy','heldout_detector_indices.json','oof_rows.json','heldout_detector.json','heldout_GT_rows.json'):
        assert sha(M/name)==mat_manifest[name],name
    h=np.load(M/'head_train32.npz'); X=np.load(M/'heldout_detector_embeddings.npy'); inds={tuple(k):j for j,k in enumerate(read(M/'heldout_detector_indices.json'))}
    assert set(h['fit_objects'])==set(groups) and X.shape[0]==len(inds)==57
    vals=softmax(X@h['weight'].T+h['bias'],axis=1)
    known=vv[['p_'+k for k in h['classes']]].to_numpy()
    dinoerr=float(np.max(np.abs(np.stack([vals[inds[r.frame,r.det]] for r in vv.itertuples()])-known)))
assert max(caterr,lrerr)<1e-8 and dinoerr<1e-6,(caterr,lrerr,dinoerr)
fit_input=read(T/'validation_upstream_replay.json'); assert set(fit_input['rf_train_objects'])==set(groups)

mass_sum=Counter()
q={(r['frame'],int(r['det'])):r for r in rows(T/'test_take_predictions.csv')}
for r in candidate:
    if r['prediction'].strip().lower()=='none': continue
    for j,line in enumerate(r['prediction'].split(';')):
        tokens=line.split(); p=q[r['image_id'],j]; on=abs(float(tokens[8])-500)<.51
        assert (int(float(p['p'])>.5)==int(tokens[9])==int(p['B']))
        mass_sum['det']+=1;mass_sum['A_vs_LR']+=int(p['A']!=p['B']);mass_sum['rounded_mass_500']+=on;mass_sum['mass500_vs_A']+=int(int(float(tokens[8])<=500)!=int(p['A']))
assert mass_sum['det']==len(q)==144
models=[W/'baseline_v0/best.pt',R/'material_dinov2_v1_20261122/dinov2_vits14_pretrain.pth',M/'head_train32.npz',M/'head_all40.npz',V/'volume_train32.cbm',T/'take_train32_strict.joblib']
assert sha(models[0])==read(R/'detection_stage6/inputs.json')['weights_sha256']
assert sha(models[4])==manifest['files'][models[4].name]
report_files=[d/'REPORT.md' for d in (M,V,T,R/'detection_stage4',R/'form_stage5',R/'detection_stage6')]
assert all(p.is_file() for p in report_files)
files=[SC,L,W/'baseline_v0/split_v0.json',old_path,V/'train_frames.json',V/'val_frames.json',V/'train_predictions.csv',V/'val_predictions.csv',V/'val_masks.jsonl.gz',R/'volume_geometry_train214_20261121_7f35d05/yolo_masks.jsonl.gz',M/'embeddings.npy',M/'heldout_detector_embeddings.npy',M/'oof_rows.json',T/'oof_strict_predictions.csv',T/'val_strict_predictions.csv',T/'test_take_predictions.csv',R/'volume_stage7/results.json',R/'volume_stage7/paired_predictions.csv',*models,*report_files]
verify=dict(inputs={str(p.relative_to(W)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},replayed_val=dict(rows=len(vv),DINO_p_max_abs_error=dinoerr,CatBoost_volume_ml_max_abs_error=caterr,LR_p_max_abs_error=lrerr),test_mass_vs_take=dict(mass_sum),test_candidate_sha256=sha(C),fold_objects=folds)
write('verification.json',verify)

# Subgroup oracles select using GT, except for the fixed archived alternatives already scored.
subcuts=[]; film=[]
for tag,part in [('OOF','train'),('val','val')]:
    rr=[r for r in frames_all if r['split']==tag]; oo={r['object_id'] for r in rr}; gt=read(V/f'{part}_frames.json')['truth']; pr=read(V/f'{part}_frames.json')['pred']; up=ix(T/('oof_strict_predictions.csv' if part=='train' else 'val_strict_predictions.csv'))
    for f,ds in pr.items():
        for j,p in enumerate(ds):
            u=up[f,j]; p.update(material=u['predicted_material'],volume=float(u['corrected_ml']),take=int(u['B']))
    b=score.components(gt,pr)
    selectors={
        'wood_volume_oracle':(lambda r:r['material']=='wood','volume_true_ml'),
        'bulk_metal_volume_oracle':(lambda r:r['material']=='metal' and r['form']=='bulk','volume_true_ml'),
        'bulk_metal_geom_if_better_oracle':(lambda r:r['material']=='metal' and r['form']=='bulk' and r['V_geom']>r['V_cat']+1e-9,'volume_geom_ml'),
        'geom_if_better_oracle':(lambda r:r['V_geom']>r['V_cat']+1e-9,'volume_geom_ml'),
        'paper_material_oracle':(lambda r:r['material']=='paper','material'),
        'wood_mineral_material_oracle':(lambda r:(r['material'],r['pred_material']) in {('wood','mineral'),('mineral','wood')},'material'),
    }
    for name,(take,field) in selectors.items():
        ss=[r for r in rr if take(r)]; new=deepcopy(pr); k='volume' if 'volume' in field else 'material'
        for r in ss: new[r['frame']][r['det']][k]=r[field]
        sc=score.components(gt,new); diffs={}; change_own={}
        for o in sorted(oo):
            ff={r['frame'] for r in rr if r['object_id']==o}; keep={f:gs for f,gs in gt.items() if f not in ff}; own={f:gs for f,gs in gt.items() if f in ff}
            diffs[o]=score.components(keep,{f:new[f] for f in keep})['Score']-score.components(keep,{f:pr[f] for f in keep})['Score']
            change_own[o]=score.components(own,{f:new[f] for f in own})['Score']-score.components(own,{f:pr[f] for f in own})['Score']
        top=min(diffs,key=diffs.get) if (ss and sc['Score']-b['Score']>1e-10) else ''
        subcuts.append(dict(metrics_row(tag,name,b,sc,'subgroup_oracle'),selected_frames=len(ss),selected_objects=len({r['object_id'] for r in ss}),**signs(change_own.values()),strongest_object=top,score_without_strongest=diffs.get(top,'')))
    for o in sorted(oo):
        rs=[r for r in rr if r['object_id']==o]
        film.append(dict(split=tag,object_id=o,form=rs[0]['form'],material=rs[0]['material'],n=len(rs),median_q50_mm=1000*med(r['q50_m'] for r in rs),median_q95_mm=1000*med(r['q95_m'] for r in rs),median_q99_mm=1000*med(r['q99_m'] for r in rs),median_area_cm2=1e4*med(r['projected_m2'] for r in rs),median_robust_over_raw=med(r['robust_ml']/r['volume_geom_ml'] for r in rs),median_cat_over_geom=med(r['volume_cat_ml']/r['volume_geom_ml'] for r in rs),median_missing_px=med(r['approx_missing_mask_px'] for r in rs),median_p_wood=med(r['p_material_wood'] for r in rs),median_p_metal=med(r['p_material_metal'] for r in rs),top2_wood=sum('wood' in (r['pred_material'],r['material_second']) for r in rs),top2_metal=sum('metal' in (r['pred_material'],r['material_second']) for r in rs)))
write('subgroup_oracles.csv',subcuts); write('object_features.csv',film)
print('replayed',verify['replayed_val'],'test',mass_sum)
for r in subcuts: print(r['split'],r['scenario'],r['selected_frames'],r['selected_objects'],round(r['delta_Score'],4),r['better'],r['strongest_object'],r['score_without_strongest'])
