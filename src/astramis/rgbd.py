"""Astramis RGB-D features in the registered depth image; volumes mL, lengths m unless named."""
from pathlib import Path
import json
import cv2
import numpy as np

FEATURES = ['base_ml','projected_m2','width_px','height_px','aspect','mask_px','q10_m','q25_m','median_m','q75_m','q95_m','q99_m','iqr_m','robust_ml','missing_fraction','outlier_fraction','depth_noise_m','solid','compactness','fill_ratio','flatness','base_fallback']

def read_frame(directory, cam):
    depth = cv2.imread(str(directory/'depth.png'), cv2.IMREAD_UNCHANGED)
    plane = np.asarray(json.loads((directory/'meta.json').read_text())['floor_plane'])
    assert depth.dtype == np.uint16 and depth.shape == (cam['height'],cam['width'])
    assert abs(np.linalg.norm(plane[:3])-1)<1e-4 and plane[3]>0
    return depth,plane

def polygon_mask(depth, det):
    mask=np.zeros(depth.shape,np.uint8)
    contour=np.asarray(det['contour_xy'],np.int32).reshape(-1,2)
    if len(contour)>=3:
        cv2.fillPoly(mask,[contour],1)
    return mask

def features(depth,plane,mask,box,form,cam, fallback=420.):
    x1,y1,x2,y2=box
    w,h=x2-x1,y2-y1
    area=int(mask.sum())
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    perimeter=sum(cv2.arcLength(c,True) for c in contours)
    y,x=np.nonzero((mask>0)&(depth>0)); count=len(y)
    # Invalid continuous estimates are missing features. Frozen geometry fallback is positive.
    proj=robust=outlier=noise=flat=np.nan
    qs=np.full(6,np.nan); raw=np.nan
    if count>=30:
        z=depth[y,x].astype(float)*0.001
        pts=np.stack([(x-cam['cx'])*z/cam['fx'],(y-cam['cy'])*z/cam['fy'],z],axis=1)
        n=plane[:3]; e1=np.cross(n,[1.,0.,0.]); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
        heights=pts@n+plane[3]; qs=np.quantile(heights,[.10,.25,.50,.75,.95,.99]); iqr=qs[3]-qs[1]
        outlier=float(np.mean((heights<qs[1]-1.5*iqr)|(heights>qs[3]+1.5*iqr)))
        keys=np.floor((pts@e1)/.004).astype(np.int64)*100000+np.floor((pts@e2)/.004).astype(np.int64)
        order=np.argsort(keys); starts=np.r_[0,np.flatnonzero(np.diff(keys[order]))+1]
        tops=np.maximum.reduceat(heights[order],starts); proj=len(starts)*.004**2
        raw=np.maximum(tops,0).sum()*16.; robust=np.clip(tops,0,max(0.,qs[-1])).sum()*16.
        flat=max(0.,qs[4])/np.sqrt(proj)
        pair=(mask[:,1:]>0)&(mask[:,:-1]>0)&(depth[:,1:]>0)&(depth[:,:-1]>0)
        if pair.any(): noise=np.quantile(np.abs(depth[:,1:][pair].astype(float)-depth[:,:-1][pair]),.75)*.001
    valid=np.isfinite(raw) and raw>1
    vals=[raw if valid else fallback,proj,w,h,w/max(h,1.),area,*qs,qs[3]-qs[1],robust,1-count/max(1,area),outlier,noise,float(form=='solid'),4*np.pi*area/max(1.,perimeter**2),area/max(1.,w*h),flat,float(not valid)]
    return dict(zip(FEATURES,vals,strict=True)),count
