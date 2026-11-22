from common import *
import pandas as pd
from collections import defaultdict

def prepare(part):
    cv2.setNumThreads(1); split=load(SPLIT); cap=load(CAP[part]); manifest=load(OLD/'mask_manifest.json');cam=load(DATA/'camera.json'); labels=load(DATA/'labels/real_train.json')
    assert cam['depth_unit']=='mm' and cam['depth_aligned_to']=='color'
    assert sha(MASKS)==manifest['mask_sha256'] and all(sha(ROOT/p)==v for p,v in manifest['input_sha256'].items())
    assert all(sha(ROOT/p)==v for p,v in cap['metadata']['inputs_sha256'].items()) and cap['metadata']['runner_sha256']==sha(BUNDLE/'run_candidate_exact.py')
    ids=split[part]['image_ids']; dirs={Path(i['file_name']).parent.name:DATA/Path(i['file_name']).parent for i in labels['images']}; anns={a['id']:a for a in labels['annotations']};imids={Path(i['file_name']).parent.name:i['id'] for i in labels['images']}; byim=defaultdict(list)
    for a in labels['annotations']:byim[a['image_id']].append(a['id'])
    assert len(ids)==len(set(ids)) and set(ids)<=set(dirs)
    mp=MASKS if part=='train' else HERE/'val_masks.jsonl.gz'
    if not mp.exists(): collect(ids,dirs,mp)
    masks=read_masks(mp); assert set(masks)==set(ids); rows=[]; assigned=set();truth={};pred={}; matches=[]; alt=[]; residual=[]; maxbox=maxconf=maxvol=0.
    for fid in ids:
        ref=cap['predictions']['material_simple_v1'].get(fid); trace=cap['depth_trace'].get(fid,[]); gs=cap['truth'].get(fid,[]); truth[fid]=gs
        assert sorted(g['annotation_id'] for g in gs)==sorted(byim[imids[fid]])
        for g in gs:
            a=anns[g['annotation_id']];at=a['attributes'];x,y,w,h=a['bbox'];assert np.allclose(g['box'],[x,y,x+w,y+h]) and g['object_id']==at['object_id'] and g['volume']==at['volume_ml'] and g['material']==at['material'];assigned.add(g['object_id'])
        ds=masks[fid]['detections'];pred[fid]=[]
        if ref is not None:assert len(ds)==len(ref)==len(trace),fid
        else:assert not gs and part=='train'
        depth,plane=read_frame(dirs[fid],cam);assert depth.shape==(masks[fid]['height'],masks[fid]['width'])
        # Floor check on registered lower half, sparse grid, includes other surfaces.
        y,x=np.mgrid[depth.shape[0]//2:depth.shape[0]:16,0:depth.shape[1]:16];z=depth[y,x]*.001;dh=((x-cam['cx'])*plane[0]/cam['fx']+(y-cam['cy'])*plane[1]/cam['fy']+plane[2])*z+plane[3];residual.append(float(np.median(np.abs(dh[z>0]))))
        for j,d in enumerate(ds):
            assert d['detection_index']==j;mask=polygon_mask(depth,d);f,valid=features(depth,plane,mask,d['box'],d['form'],cam)
            p=dict(box=d['box'],score=d['score'],form=d['form'],material='plastic',take=0,volume=f['base_ml']);
            if ref is not None:
                p=ref[j].copy();bd=float(np.max(np.abs(np.asarray(p['box'])-d['box'])));cd=abs(p['score']-d['score']);vd=abs(p['volume']-f['base_ml']);maxbox=max(bd,maxbox);maxconf=max(cd,maxconf);maxvol=max(vd,maxvol)
                assert p['form']==d['form'] and bd<=.001 and cd<=1e-6 and np.isclose(p['volume'],f['base_ml'],rtol=1e-9,atol=1e-7),(fid,j,bd,cd,vd)
                assert (int(mask.sum()),valid)==(trace[j]['mask_pixels'],trace[j]['valid_pixels'])
            pred[fid].append(p);rows.append(dict(frame=fid,det=j,**f))
            if part=='val':
                for op,func in [('erosion',cv2.erode),('dilation',cv2.dilate)]:
                    mod=func(mask,np.ones((5,5),np.uint8));af,_=features(depth,plane,mod,d['box'],d['form'],cam);alt.append(dict(frame=fid,det=j,op=op,**af))
        indices={id(p):j for j,p in enumerate(pred[fid])}
        for g,p in zip(gs,metric._match(gs,pred[fid]),strict=True):matches.append(dict(frame=fid,det=indices.get(id(p),-1),object_id=g['object_id'],target=g['volume'],material=g['material'],mass=g['mass']))
    assert assigned==set(split[part]['object_ids']); save(HERE/f'{part}_frames.json',dict(truth=truth,pred=pred));pd.DataFrame(rows).to_csv(HERE/f'{part}_features.csv',index=False);pd.DataFrame(matches).to_csv(HERE/f'{part}_matches.csv',index=False)
    if alt:pd.DataFrame(alt).to_csv(HERE/'val_mask_sensitivity.csv',index=False)
    stats=dict(frames=len(ids),positive_frames=sum(bool(v) for v in truth.values()),objects=len(assigned),detections=len(rows),annotations=len(matches),matched=sum(m['det']>=0 for m in matches),mask_sha256=sha(mp),capture_sha256=sha(CAP[part]),max_bbox_px=maxbox,max_conf_delta=maxconf,max_volume_delta_ml=maxvol,median_frame_floor_residual_m=float(np.median(residual)),fallback_count=sum(f['base_fallback'] for f in rows));save(HERE/f'{part}_audit.json',stats);print(part,stats,flush=True)
if __name__=='__main__':prepare(sys.argv[1])
