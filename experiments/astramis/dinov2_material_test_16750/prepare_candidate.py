from pathlib import Path
from collections import Counter
import csv, io, json, hashlib, math, os, sys, tempfile
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT=Path('/home/jovyan/work')
STAGE=ROOT/'runs/dinov2_material_stage1'
BUNDLE=ROOT/'onecloud_v2_20261120T220121/verified_candidate_v1'
REF=ROOT/'submission_onecloud_material_simple_v1.csv'
DEST=Path(__file__).resolve().parent
TEST=ROOT/'datasets/dataset/real/test'
EXPECTED='5ad2911ca91a485f0dc140bd5ff0aa3b34c4108f882854f2bf95ed26ad4051d6'
NAME='submission_dinov2_material_only.csv'
sha=lambda data:hashlib.sha256(data).hexdigest()

def read_csv(raw):
    rows=list(csv.reader(io.StringIO(raw.decode('utf-8'),newline=''),strict=True))
    assert rows[0]==['image_id','prediction'] and all(len(row)==2 for row in rows[1:])
    return rows[1:]

def render(rows):
    text=io.StringIO(newline='');w=csv.writer(text,lineterminator='\n');w.writerow(['image_id','prediction']);w.writerows(rows);return text.getvalue().encode('utf-8')

def raw_detections(detector,path):
    result=detector(str(path),conf=.10,imgsz=640,verbose=False)[0]
    if result.masks is None:return []
    boxes=result.boxes; forms=['bulk','solid']; masks=result.masks.xy
    assert len(boxes)==len(masks)
    return [(box,conf,forms[int(cls)]) for box,conf,cls in zip(boxes.xyxy.tolist(),boxes.conf.tolist(),boxes.cls.tolist()) if box[2]-box[0]>=2 and box[3]-box[1]>=2]

def main():
    raw=REF.read_bytes();verification=json.loads((BUNDLE/'verification.json').read_text())
    assert sha(raw)==verification['submission_sha256']==EXPECTED
    for name in ['submission.csv','reproduced_exact.csv']:assert (BUNDLE/name).read_bytes()==raw
    assert verification['test_bytes_exact_match'] and sha((BUNDLE/'run_candidate_exact.py').read_bytes())==verification['runner_sha256']
    det_hash=sha((BUNDLE/'best.pt').read_bytes());assert det_hash==json.loads((BUNDLE/'manifest.json').read_text())['input_sha256']['baseline_v0/best.pt']
    stage_manifest=json.loads((STAGE/'manifest.json').read_text());checked={}
    for name in ['head_all40.npz','material_classifier.py','encoder.json','extract_embeddings.py','group_oof.py']:
        checked[name]=sha((STAGE/name).read_bytes());assert checked[name]==stage_manifest['files'][name]
    sys.path.insert(0,str(STAGE));from material_classifier import MaterialClassifier,official
    assert sha((ROOT/'datasets/score.py').read_bytes())==stage_manifest['inputs']['datasets/score.py']
    torch.set_num_threads(2);head=MaterialClassifier(STAGE/'head_all40.npz');detector=YOLO(str(BUNDLE/'best.pt'))
    assert len(head.fit_objects)==40 and set(head.classes)==set(official.MATERIALS) and head.weight.shape==(6,384)
    assert not head.encoder.training and not any(p.requires_grad for p in head.encoder.parameters())
    base=read_csv(raw);assert render(base)==raw
    ids=[f for f,_ in base]; sample=read_csv((ROOT/'datasets/sample_submission.csv').read_bytes());assert len(ids)==len(set(ids))==137 and ids==[v[0] for v in sample]==sorted(d.name for d in TEST.iterdir() if d.is_dir())
    result=[]; inferred={};transitions=Counter();changed_frames=set();empty=0
    for index,(frame,line) in enumerate(base,1):
        path=TEST/frame/'color.png'
        if not path.is_file():path=path.with_suffix('.jpg')
        tokens=[] if line=='none' else [x.split(' ') for x in line.split(';')]
        detections=raw_detections(detector,path);assert len(tokens)==len(detections),frame
        if not tokens:empty+=1
        replaced=[]
        with Image.open(path) as rgb:
            for j,(fields,(box,confidence,form)) in enumerate(zip(tokens,detections)):
                prefix=[f'{v:.1f}' for v in box]+[f'{confidence:.3f}',form]
                assert len(fields)==10 and fields[:6]==prefix,(frame,j,'detector mismatch')
                assert fields[6] in official.MATERIALS and 0<=box[0]<box[2]<=rgb.width and 0<=box[1]<box[3]<=rgb.height
                label=head.classify(rgb,box);assert label in official.MATERIALS
                inferred[frame,j]=label; transitions[fields[6],label]+=1
                if fields[6]!=label:changed_frames.add(frame)
                replaced.append(' '.join(fields[:6]+[label]+fields[7:]))
        result.append([frame,';'.join(replaced) if replaced else 'none'])
        if index%25==0:print(f'{index}/137',flush=True)
    blob=render(result)
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        staging=Path(tmp); candidate=staging/NAME;candidate.write_bytes(blob); reread=read_csv(candidate.read_bytes()); back=[];deltas=Counter(); diff_frames=set()
        assert len(reread)==len(base)
        for (frame,old),(frame2,new) in zip(base,reread):
            assert frame==frame2 and (old=='none')==(new=='none')
            parsed=official._parse(new,True,frame); assert len(parsed)==len(official._parse(old,True,frame)); restored=[]
            if old!='none':
                for j,(old_det,new_det) in enumerate(zip(old.split(';'),new.split(';'),strict=True)):
                    a,b=old_det.split(),new_det.split();assert len(a)==len(b)==10 and b[6]==inferred[frame,j] and b[6] in official.MATERIALS
                    assert all(math.isfinite(float(b[k])) for k in [0,1,2,3,4,7,8,9]);assert 0<=float(b[4])<=1 and float(b[7])>0 and float(b[8])>=0
                    for pos,(v1,v2) in enumerate(zip(a,b)):
                        if v1!=v2:deltas[pos]+=1;diff_frames.add(frame)
                    restored.append(' '.join(b[:6]+[a[6]]+b[7:]))
            back.append([frame,';'.join(restored) if restored else 'none'])
        changed=sum(n for (a,b),n in transitions.items() if a!=b)
        assert set(deltas)<={6} and deltas[6]==changed and diff_frames==changed_frames
        assert render(back)==raw==REF.read_bytes() and sum(transitions.values())==len(inferred)
        report={'baseline_submission':16750,'frames':len(ids),'empty_prediction_rows':empty,'detections':len(inferred),'changed_materials':changed,'unchanged_materials':len(inferred)-changed,'frames_with_changes':len(changed_frames),'nonmaterial_differences':sum(v for k,v in deltas.items() if k!=6),'baseline_sha256':sha(raw),'candidate_sha256':sha(blob),'candidate_file':NAME,'baseline_reconstructed_sha256':sha(render(back)),'transitions':{a+' -> '+b:n for (a,b),n in sorted(transitions.items())},'crop':'Unrounded xyxy from frozen #16750 YOLO. Match formatted xyxy, confidence, form and detection index per frame. Stage-1 PIL RGB floor/ceil clipping, bicubic Resize 256, CenterCrop 224, ImageNet normalization.','stage1_hashes':checked,'detector_sha256':det_hash,'encoder_sha256':json.loads((STAGE/'encoder.json').read_text())['source']['dinov2_vits14_pretrain.pth']['sha256'],'checks':{'ids_equal_official_sample_in_order':True,'all_test_directories_covered_once':True,'all_detections_matched_in_order':True,'official_parser_rows':len(reread),'byte_exact_after_restoring_material':True,'baseline_and_reproduced_exact_equal':True},'official_submission_performed':False}
        report_file=staging/'integrity_report.json';report_file.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        for p in [candidate,report_file]:
            target=DEST/p.name
            if target.exists():assert target.read_bytes()==p.read_bytes(),f'Different result already exists: {target}'
        for p in [candidate,report_file]:
            if not (DEST/p.name).exists():os.replace(p,DEST/p.name)
    assert sha((DEST/NAME).read_bytes())==report['candidate_sha256'];print('VERIFIED',json.dumps({k:report[k] for k in ['frames','empty_prediction_rows','detections','changed_materials','unchanged_materials','frames_with_changes','nonmaterial_differences','baseline_sha256','candidate_sha256']}),flush=True)

if __name__=='__main__':main()
