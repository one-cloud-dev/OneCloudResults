from pathlib import Path
import hashlib, json, math, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = Path('/home/jovyan/work')
OUT = Path(__file__).resolve().parent
DATA = ROOT/'datasets/dataset'
PRIOR = ROOT/'runs/material_dinov2_v1_20261122'
OLD = ROOT/'runs/material_final_m_20261121_7f35d05_oof4'
read = lambda p: json.loads(p.read_text())
save = lambda p, x: p.write_text(json.dumps(x, indent=2, ensure_ascii=False))

def main():
    params = read(PRIOR/'weights_source.json')
    weights = PRIOR/'dinov2_vits14_pretrain.pth'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == params[weights.name]['sha256'] == 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    torch.set_num_threads(2)
    source = PRIOR/'vendor/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    sys.path.insert(0,str(source))
    from dinov2.hub.backbones import dinov2_vits14
    encoder = dinov2_vits14(pretrained=False)
    encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    encoder.requires_grad_(False).eval()
    assert encoder.embed_dim == 384 and not any(p.requires_grad for p in encoder.parameters())
    prep = T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    labels=read(DATA/'labels/real_train.json'); images={im['id']:im for im in labels['images']}
    paths={Path(im['file_name']).parent.name:DATA/im['file_name'] for im in images.values()}
    rows=[]
    for a in labels['annotations']:
        frame=Path(images[a['image_id']]['file_name']).parent.name
        x,y,w,h=a['bbox']; attr=a['attributes']
        rows.append(dict(kind='gt',frame=frame,box=[x,y,x+w,y+h],object_id=attr['object_id'],material=attr['material'],ann=a['id']))
    baseline=read(OLD/'oof_predictions.json')['A']
    for frame,dets in baseline.items():
        for j,det in enumerate(dets): rows.append(dict(kind='det',frame=frame,index=j,box=det['box']))
    batches=[]; batch=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            path=paths[row['frame']]
            if not path.exists(): path=path.with_suffix('.jpg')
            with Image.open(path) as im:
                x1,y1,x2,y2=row['box']; box=(max(0,math.floor(x1)),max(0,math.floor(y1)),min(im.width,math.ceil(x2)),min(im.height,math.ceil(y2)))
                assert box[2]>box[0] and box[3]>box[1]
                batch.append(prep(im.convert('RGB').crop(box)))
            if len(batch)==8 or i+1==len(rows):
                batches.append(encoder(torch.stack(batch)).cpu().numpy()); batch.clear()
    X=np.concatenate(batches); assert X.shape==(len(rows),384) and np.isfinite(X).all()
    np.save(OUT/'embeddings.npy',X); save(OUT/'crops.json',rows); save(OUT/'encoder.json',dict(source=params,embedding='CLS after final layer normalization',transform='RGB, bbox floor/ceil clipped, Resize(256,bicubic), CenterCrop(224), ImageNet normalization',torch=torch.__version__))
    print('SAVED',X.shape,'sha256',digest,flush=True)

if __name__=='__main__': main()
