from pathlib import Path
import copy, hashlib, json, math, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from extract_embeddings import DATA, ROOT, OUT, PRIOR, read, save
from group_oof import official, CLASSES

class MaterialClassifier:
    def __init__(self, head_path=OUT/'head_all40.npz'):
        weight_path=PRIOR/'dinov2_vits14_pretrain.pth'
        assert hashlib.sha256(weight_path.read_bytes()).hexdigest()==read(OUT/'encoder.json')['source'][weight_path.name]['sha256']
        sys.path.insert(0,str(PRIOR/'vendor/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'))
        from dinov2.hub.backbones import dinov2_vits14
        self.encoder=dinov2_vits14(pretrained=False)
        self.encoder.load_state_dict(torch.load(weight_path,weights_only=True,map_location='cpu'),strict=True)
        self.encoder.requires_grad_(False).eval()
        self.prep=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        with np.load(head_path) as h:
            self.classes,self.weight,self.bias=h['classes'],h['weight'],h['bias']
            self.fit_objects=set(h['fit_objects'])
    @torch.inference_mode()
    def embed(self,rgb_image,box):
        x,y,right,bottom=box
        limits=(max(0,math.floor(x)),max(0,math.floor(y)),min(rgb_image.width,math.ceil(right)),min(rgb_image.height,math.ceil(bottom)))
        crop=rgb_image.convert('RGB').crop(limits)
        return self.encoder(self.prep(crop).unsqueeze(0)).numpy()[0]
    def classify(self,rgb_image,box):
        return str(self.classes[(self.weight@self.embed(rgb_image,box)+self.bias).argmax()])

def holdout():
    torch.set_num_threads(2)
    capture=read(ROOT/'runs/diagnostic_v2_20261121/capture.json'); split=read(ROOT/'baseline_v0/split_v0.json')
    truth=capture['truth'];base=capture['predictions']['material_simple_v1'];cand=copy.deepcopy(base)
    assert set(truth)==set(base)==set(split['val']['image_ids']); model=MaterialClassifier(OUT/'head_train32.npz')
    assert model.fit_objects.isdisjoint({g['object_id'] for objs in truth.values() for g in objs}); assert len(model.fit_objects)==32
    labs=read(DATA/'labels/real_train.json'); paths={Path(i['file_name']).parent.name:DATA/i['file_name'] for i in labs['images']}
    anns={i['id']:Path(i['file_name']).parent.name for i in labs['images']}
    verified=set()
    for a in labs['annotations']:
        f=anns[a['image_id']]
        if f in truth:
            g=truth[f][0]; attr=a['attributes']; x,y,w,h=a['bbox']; assert (attr['object_id'],attr['material'])==(g['object_id'],g['material']) and np.allclose(g['box'],[x,y,x+w,y+h]); verified.add(f)
    assert verified==set(truth)
    embs=[]; indices=[];rows=[]
    for f in sorted(cand):
        path=paths[f]
        if not path.exists():path=path.with_suffix('.jpg')
        with Image.open(path) as im:
            for j,p in enumerate(cand[f]):
                z=model.embed(im,p['box']); embs.append(z); indices.append([f,j]);p['material']=str(model.classes[(model.weight@z+model.bias).argmax()])
        for g,b,c in zip(truth[f],official._match(truth[f],base[f]),official._match(truth[f],cand[f])):
            rows.append(dict(frame=f,object_id=g['object_id'],truth=g['material'],RF=b['material'] if b else 'miss',DINO=c['material'] if c else 'miss'))
    result={}
    for name,p in [('RF',base),('DINO',cand)]:
        hits=[r[name]==r['truth'] for r in rows]; recall={m:sum(r[name]==m for r in rows if r['truth']==m)/sum(r['truth']==m for r in rows) for m in sorted({r['truth'] for r in rows})}
        result[name]=dict(M=official.components(truth,p)['M'],accuracy=float(np.mean(hits)),recall=recall)
    assert abs(result['RF']['M']-capture['scores']['material_simple_v1']['M'])<1e-10
    np.save(OUT/'heldout_detector_embeddings.npy',np.stack(embs));save(OUT/'heldout_detector_indices.json',indices);save(OUT/'heldout_detector.json',dict(metrics=result,rows=rows));print('Independent holdout',len(rows),len(indices),result)

if __name__=='__main__':holdout()
