#!/usr/bin/env python3
"""Render full global C256 with one Cartesian cube owned by fresh local decode."""
from __future__ import annotations
import json,os
from pathlib import Path
import torch
from pixal3d.representations import Mesh
import run_head_similarity_experiment as exp

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"outputs/head_full_global_fresh_cube_cuda4"

def main():
    device=torch.device("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:4");torch.cuda.set_device(device);OUT.mkdir(parents=True,exist_ok=True)
    fresh=torch.load(ROOT/"outputs/head_cube_fresh_patch_prototype_cuda4/fresh_exact_cube.pt",map_location="cpu",weights_only=False)["mesh"]
    d=torch.load(ROOT/"outputs/global_c256_encdown_context1024_flow_singleview_cuda4/final/final_material_mesh.pt",map_location="cpu",weights_only=False);g=d["mesh"];v,f=g.vertices,g.faces;del d,g
    lo=torch.tensor([.25,-.25,0.]);hi=torch.tensor([.5,0.,.25]);keep=[]
    # Keep crossing triangles to cover the interface; remove a global face only
    # when all three vertices are strictly inside the fresh-owned cube.
    for fc in f.split(2_000_000):
        tri=v[fc.long()];remove=((tri>lo)&(tri<hi)).all((1,2));keep.append((~remove).cpu())
    keep=torch.cat(keep);outside=f[keep];hybrid=Mesh(torch.cat((v,fresh.vertices)),torch.cat((outside,fresh.faces+len(v))))
    setup=json.loads((ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/local_setup.json").read_text())
    exp.render_fixed_multiview({"full_global_C256_fresh_cube":hybrid},setup,OUT,device)
    manifest={"representation":"two-source scene rendered as one triangle mesh","global_source":"global C256 mesh","fresh_source":"camera-compatible fresh-SS native local decode","cube_bounds":[lo.tolist(),hi.tolist()],"ownership":"remove global triangles only when all vertices are strictly inside cube; retain crossing triangles; insert fresh exact-cube faces","global_faces_kept":len(outside),"fresh_faces":len(fresh.faces),"combined_faces":len(hybrid.faces),"warning":"renderable proof of concept; overlap at crossing triangles is retained to avoid cracks, but vertices are not welded"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n");print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
