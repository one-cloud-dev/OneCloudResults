from common import *
from train_eval import predict,CatBoostRegressor,pd
import csv,io

def render(rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerows(rows);return stream.getvalue().encode()
def main():
    path=RUNS/'dinov2_material_test_16750/submission_dinov2_material_only.csv'; integrity=load(path.parent/'integrity_report.json'); raw=path.read_bytes();assert sha(path)==integrity['candidate_sha256'] and sha(BUNDLE/'best.pt')==integrity['detector_sha256']
    rows=list(csv.reader(io.StringIO(raw.decode())));assert render(rows)==raw;header=rows[0];assert header==['image_id','prediction']; ids=[r[0] for r in rows[1:]]; sample=pd.read_csv(ROOT/'datasets/sample_submission.csv');assert ids==list(sample.image_id) and len(set(ids))==137
    dirs={f:DATA/'real/test'/f for f in ids};active=[fid for fid,line in rows[1:] if line!='none']; mp=HERE/'test_positive_masks.jsonl.gz'
    if not mp.exists():collect(active,dirs,mp)
    masks=read_masks(mp);assert set(active)==set(masks); model=CatBoostRegressor();model.load_model(str(HERE/'volume_train32.cbm'));cam=load(DATA/'camera.json');result=[header];restored=[header];changed=total=empty=0
    for fid,line in rows[1:]:
        if line=='none':
            assert not metric._parse(line,True,fid);result.append([fid,line]);restored.append([fid,line]);empty+=1;continue
        depth,plane=read_frame(dirs[fid],cam);dets=masks[fid]['detections']; originals=[t.split(' ') for t in line.split(';')]; values=[]
        for j,(tok,det) in enumerate(zip(originals,dets,strict=True)):
            assert len(tok)==10 and det['detection_index']==j and tok[:6]==[*(f'{x:.1f}' for x in det['box']),f"{det['score']:.3f}",det['form']],(fid,j)
            fs,_=features(depth,plane,polygon_mask(depth,det),det['box'],det['form'],cam);assert tok[7]==f"{fs['base_ml']:.0f}",(fid,j,'raw volume');values.append(fs)
        estimates=predict(model,pd.DataFrame(values));new=[];restore=[]
        for t,v in zip(originals,estimates,strict=True):
            pred=[*t];pred[7]=str(max(1,round(v)));changed+=pred[7]!=t[7];total+=1;assert pred[:7]+pred[8:]==t[:7]+t[8:];new.append(' '.join(pred));restore.append(' '.join(pred[:7]+[t[7]]+pred[8:]))
        cell=';'.join(new);assert len(metric._parse(cell,True,fid))==len(originals);result.append([fid,cell]);restored.append([fid,';'.join(restore)])
    assert render(restored)==raw==path.read_bytes() and (total,empty)==(144,19);p=HERE/'submission_dinov2_volume_diagnostic.csv';blob=render(result);assert not p.exists() or p.read_bytes()==blob;p.write_bytes(blob);assert list(csv.reader(io.StringIO(p.read_text())))==result
    audit=dict(frames=len(ids),detections=total,empty=empty,volume_changes=changed,other_changes=0,original_sha256=sha(path),candidate_sha256=sha(p),model_sha256=sha(HERE/'volume_train32.cbm'),test_mask_sha256=sha(mp),restored_bytes_exact=True,official_parser_rows=len(ids),submitted=False);save(HERE/'test_integrity.json',audit);print(audit,flush=True)
if __name__=='__main__':main()
