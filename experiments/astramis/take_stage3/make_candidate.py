from prepare_features import ROOT,OUT,MAT,VOL,DATA,CLASSES,read,sha,score
from evaluate import matrix
import sys
import csv
import io
import json
import hashlib
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from scipy.special import softmax
import torch
sys.path.insert(0,str(VOL));sys.path.insert(0,str(MAT))
from material_classifier import MaterialClassifier
from common import read_masks, read_frame, features, polygon_mask
from train_eval import CatBoostRegressor,predict

def parse(data):
    return list(csv.reader(io.StringIO(data.decode(),newline=''),strict=True))

def render(rows):
    buff=io.StringIO(newline='');csv.writer(buff,lineterminator='\n').writerows(rows);return buff.getvalue().encode()

def main():
    torch.set_num_threads(2)
    base=VOL/'submission_dinov2_volume_diagnostic.csv';raw=base.read_bytes();vmanifest=read(VOL/'manifest.json')['files'];assert sha(base)==vmanifest[base.name]==read(VOL/'test_integrity.json')['candidate_sha256'];assert sha(VOL/'test_positive_masks.jsonl.gz')==vmanifest['test_positive_masks.jsonl.gz']
    old=ROOT/'submission_onecloud_material_simple_v1.csv';assert sha(old)==read(ROOT/'runs/dinov2_material_test_16750/integrity_report.json')['baseline_sha256'];oldrows=parse(old.read_bytes());rows=parse(raw);assert rows[0]==oldrows[0]==['image_id','prediction'] and render(rows)==raw
    model=joblib.load(OUT/'take_train32_strict.joblib');head=MaterialClassifier(MAT/'head_train32.npz'); assert list(head.classes)==CLASSES and head.fit_objects==set(read(ROOT/'baseline_v0/split_v0.json')['train']['object_ids']);vol=CatBoostRegressor();vol.load_model(str(VOL/'volume_train32.cbm'))
    ids=[x[0] for x in rows[1:]];assert ids==list(pd.read_csv(ROOT/'datasets/sample_submission.csv').image_id) and len(set(ids))==137
    masks=read_masks(VOL/'test_positive_masks.jsonl.gz');assert set(masks)=={f for f,line in rows[1:] if line!='none'};cam=read(DATA/'camera.json');den=pd.read_csv(DATA/'densities.csv').set_index(['form','material']).density_g_per_ml.to_dict(); new=[rows[0]];changes=[];empty=0;records=[]
    for (f,line),(fo,legacy) in zip(rows[1:],oldrows[1:],strict=True):
        assert fo==f and (line=='none')==(legacy=='none');tokens=[]
        if line=='none':
            new.append([f,line]);empty+=1;continue
        folder=DATA/'real/test'/f;depth,plane=read_frame(folder,cam);path=folder/'color.png';path=path if path.exists() else path.with_suffix('.jpg')
        with Image.open(path) as im:
            for j,(token,otoken,det) in enumerate(zip(line.split(';'),legacy.split(';'),masks[f]['detections'],strict=True)):
                t=token.split(' ');org=otoken.split(' ');assert len(t)==len(org)==10 and t[:6]==org[:6] and t[8:]==org[8:] and det['detection_index']==j
                assert t[:6]==[*(f'{x:.1f}' for x in det['box']),f"{det['score']:.3f}",det['form']]
                d,_=features(depth,plane,polygon_mask(depth,det),det['box'],det['form'],cam); corrected=predict(vol,pd.DataFrame([d]))[0];mass=d['base_ml']*den.get((det['form'],org[6]),1.)
                assert str(max(1,round(corrected)))==t[7] and f'{mass:.0f}'==t[8] and str(int(mass<=500))==t[9]
                probs=softmax(head.weight@head.embed(im,det['box'])+head.bias); x=pd.DataFrame([dict(corrected_ml=corrected,mass=mass,solid=d['solid'],projected_m2=d['projected_m2'],**dict(zip(['p_'+c for c in CLASSES],probs,strict=True)))])
                p=float(model.predict_proba(matrix(x,'corrected_ml'))[0,1]);b=str(int(p>.5));records.append(dict(frame=f,det=j,A=int(t[9]),B=int(b),p=p));tokens.append(' '.join(t[:9]+[b]));
                if b!=t[9]:changes.append((f,t[9]+b))
        new.append([f,';'.join(tokens)])
    blob=render(new);reread=parse(blob);back=[rows[0]]
    for (f,line),(fn,altered) in zip(rows[1:],reread[1:],strict=True):
        assert f==fn and len(score._parse(line,True,f))==len(score._parse(altered,True,f));reverse=[]
        if line!='none':
            for before,after in zip(line.split(';'),altered.split(';'),strict=True):
                x,y=before.split(' '),after.split(' ');assert x[:9]==y[:9] and y[9] in ('0','1');reverse.append(' '.join(y[:9]+[x[9]]))
        back.append([f,';'.join(reverse) if reverse else 'none'])
    assert render(back)==raw==base.read_bytes() and (len(records),empty)==(144,19)
    dst=OUT/'submission_take_stage3_diagnostic.csv';assert not dst.exists() or dst.read_bytes()==blob;dst.write_bytes(blob);assert dst.read_bytes()==blob
    pd.DataFrame(records).to_csv(OUT/'test_take_predictions.csv',index=False);audit=dict(frames=len(ids),detections=len(records),empty=empty,changed_take=len(changes),frames_changed=len(set(f for f,_ in changes)),take_0_to_1=sum(t=='01' for _,t in changes),take_1_to_0=sum(t=='10' for _,t in changes),other_token_changes=0,restored_exact=True,official_parser_rows=len(ids),baseline=base.name,baseline_sha256=sha(base),restored_sha256=hashlib.sha256(render(back)).hexdigest(),candidate_sha256=sha(dst),submitted=False);(OUT/'integrity.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
if __name__=='__main__':
    main()
