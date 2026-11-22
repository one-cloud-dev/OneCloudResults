from pathlib import Path
import os,sys,json,gzip,hashlib
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]; DATA=ROOT/'datasets/dataset'; RUNS=ROOT/'runs'; OLD=RUNS/'volume_geometry_train214_20261121_7f35d05'; BUNDLE=ROOT/'onecloud_v2_20261120T220121/verified_candidate_v1'; MASKS=OLD/'yolo_masks.jsonl.gz'; SPLIT=ROOT/'baseline_v0/split_v0.json'; FOLDS=RUNS/'dinov2_material_stage1/results.json'
CAP={'train':RUNS/'volume_v1_train165_20261121/capture.json','val':RUNS/'diagnostic_v2_20261121/capture.json'}
sys.path.insert(0,str(HERE/'deps'));sys.path.insert(0,str(ROOT/'datasets'))
import score as metric
from features import features,polygon_mask,read_frame,FEATURES

def load(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))
def collect(ids,dirs,path):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(2); model=YOLO(str(BUNDLE/'best.pt')); rows=[]
    for fid in ids:
        p=dirs[fid]/'color.png'; p=p if p.exists() else p.with_suffix('.jpg')
        result=model(str(p),conf=.1,imgsz=640,iou=.5,device='cpu',verbose=False)[0]; dets=[]
        if result.masks is not None:
            boxes=result.boxes
            for box,sc,cl,poly in zip(boxes.xyxy.tolist(),boxes.conf.tolist(),boxes.cls.tolist(),result.masks.xy,strict=True):
                if box[2]-box[0]>=2 and box[3]-box[1]>=2:
                    dets.append(dict(detection_index=len(dets),box=box,score=sc,form=['bulk','solid'][int(cl)],contour_xy=poly.round().astype('int32').tolist()))
        rows.append(dict(frame_id=fid,height=result.orig_shape[0],width=result.orig_shape[1],detections=dets))
    temp=path.with_suffix('.partial');
    with gzip.open(temp,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    temp.replace(path)

def read_masks(p):
    with gzip.open(p,'rt') as f: rows=[json.loads(line) for line in f]
    result={r['frame_id']:r for r in rows}; assert len(result)==len(rows);return result
