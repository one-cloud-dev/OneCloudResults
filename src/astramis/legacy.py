"""Геометрический объём и признаки RF из material_simple_v1."""

import cv2
import numpy as np


def raw_volume_ml(depth_mm,mask,plane,CAM,cell=0.004):
    v,u=np.nonzero(mask & (depth_mm>0))
    if len(v)<30: return None
    z=depth_mm[v,u]/1000.0
    pts=np.stack([(u-CAM['cx'])/CAM['fx']*z,(v-CAM['cy'])/CAM['fy']*z,z],axis=1)
    n,d=np.array(plane[:3]),plane[3]
    height=pts@n+d
    e1=np.cross(n,[1.0,0.0,0.0]); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
    keys=np.floor(pts@e1/cell).astype(np.int64)*100000+np.floor(pts@e2/cell).astype(np.int64)
    order=np.argsort(keys); keys,height=keys[order],height[order]
    starts=np.r_[0,np.nonzero(np.diff(keys))[0]+1]
    tops=np.maximum.reduceat(height,starts)
    return float(np.clip(tops,0,None).sum()*cell*cell*1e6)


def material_features(crop_rgb):
    if crop_rgb.size==0: return None
    h0,w0=crop_rgb.shape[:2]
    img=cv2.resize(crop_rgb,(64,64),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    hist=[np.histogram(hsv[:,:,i],bins=bins,range=limits)[0].astype(float)/4096 for i,bins,limits in [(0,16,(0,180)),(1,8,(0,256)),(2,8,(0,256))]]
    pixels=img.reshape(-1,3).astype(float)/255
    summary=np.r_[pixels.mean(0),pixels.std(0),np.percentile(pixels,[10,50,90],axis=0).ravel(),np.log1p(w0*h0),w0/max(h0,1),np.log1p(cv2.Laplacian(gray,cv2.CV_32F).var())]
    return np.r_[*hist,summary]
