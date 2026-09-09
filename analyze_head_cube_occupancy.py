#!/usr/bin/env python3
"""Diagnose why a Cartesian global-C256 C64 cut is OOD as a local object support."""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs/head_cube_occupancy_analysis"


def topo(coords:torch.Tensor,res:int)->dict:
    x=coords[:,1:].cpu().int().numpy(); pts={tuple(v) for v in x.tolist()}
    dist=np.minimum(x,res-1-x).min(1); deg=[]
    for p in pts:deg.append(sum((p[0]+a,p[1]+b,p[2]+c) in pts for a,b,c in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))))
    unseen=set(pts);cc=[]
    while unseen:
        stack=[unseen.pop()];n=1
        while stack:
            p=stack.pop()
            for a,b,c in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                q=(p[0]+a,p[1]+b,p[2]+c)
                if q in unseen:unseen.remove(q);stack.append(q);n+=1
        cc.append(n)
    face={}
    for i,a in enumerate("xyz"):
        face[a+"0"]=int((x[:,i]==0).sum());face[a+"max"]=int((x[:,i]==res-1).sum())
    q=(x+.5)/res-.5; cov=np.cov(q.T); eig=np.linalg.eigvalsh(cov)
    return {"tokens":len(x),"min":x.min(0).tolist(),"max":x.max(0).tolist(),"span_cells":(x.max(0)-x.min(0)+1).tolist(),
            "centroid_canonical":q.mean(0).tolist(),"std_canonical":q.std(0).tolist(),"cov_eigenvalues":eig.tolist(),
            "boundary_face_counts":face,"boundary_any_ratio":float((dist==0).mean()),"within_1_cell_of_boundary_ratio":float((dist<=1).mean()),
            "boundary_distance_quantiles":np.quantile(dist,[0,.1,.25,.5,.75,.9,1]).tolist(),
            "central_75pct_ratio":float(((x>=res/8)&(x<res-res/8)).all(1).mean()),"central_50pct_ratio":float(((x>=res/4)&(x<res-res/4)).all(1).mean()),
            "degree_histogram":np.bincount(deg,minlength=7).tolist(),"components":len(cc),"largest_component":max(cc)}


def projection(coords:torch.Tensor,res:int,camera:dict)->dict:
    q=2*(coords[:,1:].double()+.5)/res-1; g=q/2;f=1024/(2*math.tan(camera["camera_angle_x"]/2));d=camera["distance"]-g[:,2]
    uv=torch.stack((f*g[:,0]/d+512,-f*g[:,1]/d+512),1).numpy(); inside=(d.numpy()>0)&(uv[:,0]>=0)&(uv[:,0]<1024)&(uv[:,1]>=0)&(uv[:,1]<1024)
    r=np.linalg.norm((uv-512)/512,axis=1)
    return {"in_frame_ratio":float(inside.mean()),"pixel_bbox_all":[*uv.min(0).tolist(),*uv.max(0).tolist()],"pixel_centroid":uv.mean(0).tolist(),
            "normalized_radius_quantiles":np.quantile(r,[.1,.25,.5,.75,.9,.95,.99]).tolist(),"depth_quantiles":np.quantile(d.numpy(),[.01,.1,.5,.9,.99]).tolist()}


def image_compare(named:dict[str,tuple[torch.Tensor,int]],path:Path)->None:
    panel=512; sheet=Image.new("RGB",(len(named)*panel,3*panel),(248,248,248)); draw=ImageDraw.Draw(sheet)
    for col,(name,(coords,res)) in enumerate(named.items()):
        x=coords[:,1:].cpu().int().numpy()
        for row,axis in enumerate(range(3)):
            axes=[i for i in range(3) if i!=axis]; a=np.zeros((res,res),np.uint16)
            np.add.at(a,(x[:,axes[1]],x[:,axes[0]]),1); v=np.log1p(a)/max(np.log1p(a.max()),1e-9); rgb=np.uint8(v[...,None]*np.array([40,190,255])[None,None,:])
            im=Image.fromarray(rgb).resize((panel,panel),Image.Resampling.NEAREST);sheet.paste(im,(col*panel,row*panel));draw.text((col*panel+8,row*panel+8),f"{name} max-proj {'xyz'[axis]}",fill="white")
    path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    global_camera=json.loads((ROOT/"outputs/baseline1024_c128_8xc64_geometry_cuda4/camera.json").read_text())
    local_camera=json.loads((ROOT/"outputs/c128_head_crop_baseline1024_geometry_cuda4/camera.json").read_text())
    crop_box=json.loads((ROOT/"outputs/c128_head_crop_4096/manifest.json").read_text())["crop_xyxy_4096"]
    payload=torch.load(ROOT/"outputs/global_c256_encdown_context1024_flow_singleview_cuda4/support/global_c256_support.pt",map_location="cpu",weights_only=False);c=payload["coords"].int();xyz=c[:,1:]
    f=4096/(2*math.tan(global_camera["camera_angle_x"]/2));q=2*(xyz.double()+.5)/256-1;g=q/2;d=global_camera["distance"]-g[:,2]
    uv=torch.stack((f*g[:,0]/d+2048,-f*g[:,1]/d+2048),1);x0,y0,x1,y1=crop_box;inroi=(d>0)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
    ranking=[]
    for sx in (0,64,128,192):
      for sy in (0,64,128,192):
       for sz in (0,64,128,192):
        m=((xyz>=torch.tensor([sx,sy,sz]))&(xyz<torch.tensor([sx+64,sy+64,sz+64]))).all(1);n=int(m.sum());hit=int((m&inroi).sum())
        ranking.append({"start":[sx,sy,sz],"tokens":n,"projected_into_head_roi":hit,"roi_fraction":hit/max(n,1)})
    ranking.sort(key=lambda z:(z["projected_into_head_roi"],z["roi_fraction"]),reverse=True);best=ranking[0];start=torch.tensor(best["start"]);m=((xyz>=start)&(xyz<start+64)).all(1)
    legacy64=torch.cat((torch.zeros(int(m.sum()),1,dtype=torch.int32),xyz[m]-start),1);legacy32=torch.cat((torch.zeros(len(legacy64),1,dtype=torch.int32),torch.div(legacy64[:,1:],2,rounding_mode="floor")),1).unique(dim=0)
    native32=torch.load(ROOT/"outputs/c128_head_crop_baseline1024_geometry_cuda4/support/coords_c32.pt",map_location="cpu",weights_only=False)["coords"].int()
    fixedfresh=torch.load(ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/support/B_fresh_c32.pt",map_location="cpu",weights_only=False)["coords"].int()
    fixedinherit=torch.load(ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/support/C_inherited_c32.pt",map_location="cpu",weights_only=False)["coords"].int()
    names={"native_crop_MoGe_C32":native32,"global_compatible_fresh_C32":fixedfresh,"metric_inherited_C32":fixedinherit,"legacy_best_cube_down_C32":legacy32}
    stats={k:{"occupancy":topo(v,32),"projection_under_relevant_local_camera":projection(v,32,local_camera if k=="native_crop_MoGe_C32" else json.loads((ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/local_setup.json").read_text())["local_camera"])} for k,v in names.items()}
    # How many cut-boundary tokens demonstrably continue in the adjacent global cube?
    global_set={tuple(v) for v in xyz.tolist()}; local=legacy64[:,1:].numpy(); continuation={}
    for axis,a in enumerate("xyz"):
      for side,val,delta in (("0",0,-1),("max",63,1)):
        ids=np.where(local[:,axis]==val)[0];continued=0
        for i in ids:
          p=(local[i]+best["start"]).copy();p[axis]+=delta;continued+=tuple(p.tolist()) in global_set
        continuation[a+side]={"boundary_tokens":len(ids),"continue_outside_cube":continued,"continuation_ratio":continued/max(len(ids),1)}
    result={"hypothesis":"Cartesian cut support lies on/traverses local cube faces, unlike native SS support centered by its camera frustum.","head_roi":crop_box,
            "best_cartesian_cube_by_projected_C256_tokens":best,"top_cube_ranking":ranking[:12],"legacy_cut_boundary_continuation":continuation,"datasets":stats,
            "interpretation":{"coordinate_range_not_equivalence":"0..63 only specifies an address range; it does not make occupancy distribution, camera rays, truncation topology, or positional encoding distribution equal.",
            "key_test":"If edge occupancy is causal, an isotropically recentered metric cube should reduce boundary mass and fragmentation; the prior B/C experiment does exactly this for the camera while retaining inherited support, and inherited remains much worse than fresh because its support topology is still not the SS posterior."}}
    (OUT/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    image_compare({"native crop":(native32,32),"fixed fresh":(fixedfresh,32),"metric inherited":(fixedinherit,32),"legacy cut":(legacy32,32)},OUT/"occupancy_max_projection.png")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
