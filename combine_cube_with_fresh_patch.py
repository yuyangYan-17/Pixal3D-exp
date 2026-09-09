#!/usr/bin/env python3
"""Prototype: Cartesian C256 cube owns output; fresh local SS owns its interior."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from pixal3d.representations import Mesh
import run_head_similarity_experiment as exp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs/head_cube_fresh_patch_prototype_cuda4"

def subset(vertices:torch.Tensor,faces:torch.Tensor,mask:torch.Tensor)->Mesh:
    f=faces[mask].long();used,inv=torch.unique(f.reshape(-1),sorted=True,return_inverse=True)
    return Mesh(vertices[used].clone(),inv.reshape(-1,3).int())

def face_select(vertices,faces,lo,hi,mode="centroid",chunk=2_000_000):
    masks=[]
    for f in faces.split(chunk):
        tri=vertices[f.long()]
        if mode=="all":m=((tri>=lo)&(tri<=hi)).all((1,2))
        else:m=(((tri.mean(1)>=lo)&(tri.mean(1)<=hi)).all(1))
        masks.append(m.cpu())
    return torch.cat(masks)

def combine(a:Mesh,b:Mesh)->Mesh:
    return Mesh(torch.cat((a.vertices,b.vertices)),torch.cat((a.faces,b.faces+len(a.vertices))))

def detail(path:Path):
    from PIL import Image
    a=np.asarray(Image.open(path).convert("RGB"),float)/255.;m=np.max(np.abs(a-.5),2)>.03
    gx=np.linalg.norm(a[:,1:]-a[:,:-1],axis=2);gy=np.linalg.norm(a[1:]-a[:-1],axis=2);v=np.r_[gx[m[:,1:]&m[:,:-1]],gy[m[1:]&m[:-1]]]
    return {"foreground_pixels":int(m.sum()),"normal_gradient_mean":float(v.mean()),"normal_gradient_p90":float(np.quantile(v,.9))}

def main():
    p=argparse.ArgumentParser();p.add_argument("--cuda-device",type=int,default=4);p.add_argument("--shell-width",type=float,default=.018);args=p.parse_args()
    device=torch.device("cuda:0" if __import__('os').environ.get('CUDA_VISIBLE_DEVICES') else f"cuda:{args.cuda_device}");torch.cuda.set_device(device);OUT.mkdir(parents=True,exist_ok=True)
    start=torch.tensor([192,64,128]);lo=start.float()/256-.5;hi=(start.float()+64)/256-.5
    fresh=torch.load(ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/B_fresh/native_mesh_global_similarity.pt",map_location="cpu",weights_only=False)["mesh"]
    # Keep a small exterior halo during generation, but exact cube ownership in the exported core.
    mask=face_select(fresh.vertices,fresh.faces,lo,hi,"all");fresh_core=subset(fresh.vertices,fresh.faces,mask)
    halo=.025;mask_h=face_select(fresh.vertices,fresh.faces,lo-halo,hi+halo,"all");fresh_halo=subset(fresh.vertices,fresh.faces,mask_h);del fresh,mask,mask_h
    print(f"fresh core faces={len(fresh_core.faces):,} halo faces={len(fresh_halo.faces):,}",flush=True)
    payload=torch.load(ROOT/"outputs/global_c256_encdown_context1024_flow_singleview_cuda4/final/final_material_mesh.pt",map_location="cpu",weights_only=False);gm=payload["mesh"];gv,gf=gm.vertices,gm.faces;del payload,gm
    gmask=face_select(gv,gf,lo,hi,"centroid");global_core=subset(gv,gf,gmask)
    # Preserve global geometry only in a thin six-face interface shell. Fresh
    # owns the interior; a half-shell overlap avoids raster cracks. This is a
    # diagnostic composite, intentionally not a claimed watertight weld.
    shell=float(args.shell_width);cent=[]
    for f in gf.split(2_000_000):cent.append(f) # retain chunked source without a 128M-face centroid tensor
    shell_masks=[]
    for f,m in zip(cent,gmask.split(2_000_000)):
        tri=gv[f.long()];c=tri.mean(1);dist=torch.minimum(c-lo,hi-c).min(1).values
        shell_masks.append((m&(dist<=shell)).cpu())
    gsmask=torch.cat(shell_masks);global_shell=subset(gv,gf,gsmask);del gv,gf,gmask,gsmask,cent,shell_masks
    flo=lo+shell*.5;fhi=hi-shell*.5;fm=face_select(fresh_core.vertices,fresh_core.faces,flo,fhi,"centroid");fresh_interior=subset(fresh_core.vertices,fresh_core.faces,fm)
    hybrid=combine(fresh_interior,global_shell)
    meshes={"fresh_exact_cube":fresh_core,"fresh_with_generation_halo":fresh_halo,"global_C256_exact_cube":global_core,"hybrid_fresh_interior_global_shell":hybrid}
    for name,m in meshes.items():torch.save({"mesh":m,"global_bounds":[lo.tolist(),hi.tolist()],"role":name},OUT/f"{name}.pt")
    setup=json.loads((ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/local_setup.json").read_text());exp.render_fixed_multiview(meshes,setup,OUT,device)
    render_root=OUT/"phase_bc/fixed_multiview_global_coordinates";metrics={"cube_start_C256":start.tolist(),"global_bounds":[lo.tolist(),hi.tolist()],"shell_width":shell,"policy":"fresh local baseline owns cube interior; original global C256 owns thin boundary shell; no latent merge and no deformation",
      "meshes":{n:{"vertices":len(m.vertices),"faces":len(m.faces),"front_detail":detail(render_root/f"{n}_yaw000_normal.png")} for n,m in meshes.items()},
      "warning":"hybrid is an overlap-shell proof of concept, not watertight welding"}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n");print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
