#!/usr/bin/env python3
"""Partition one continuous fresh local decode into global C256/C64 owners."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from pixal3d.representations import Mesh

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"outputs/head_fresh_patch_global_cube_partition"

def canonical_triangles(mesh:Mesh)->np.ndarray:
    tri=mesh.vertices[mesh.faces.long()].numpy();tri=np.sort(tri.view([('x','<f4'),('y','<f4'),('z','<f4')]),axis=1).view('<f4').reshape(-1,3,3);return tri

def main():
    OUT.mkdir(parents=True,exist_ok=True);source=torch.load(ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc/B_fresh/native_mesh_global_similarity.pt",map_location="cpu",weights_only=False)["mesh"]
    v,f=source.vertices,source.faces.long();cent=(v[f[:,0]]+v[f[:,1]]+v[f[:,2]])/3
    ijk=torch.floor((cent+.5)*4).int().clamp(0,3);owner=ijk[:,0]*16+ijk[:,1]*4+ijk[:,2]
    records=[];pieces=[]
    for cid in torch.unique(owner,sorted=True).tolist():
        ids=torch.where(owner==cid)[0];faces=f[ids];used,inv=torch.unique(faces.reshape(-1),sorted=True,return_inverse=True);m=Mesh(v[used],inv.reshape(-1,3));i,j,k=cid//16,(cid%16)//4,cid%4
        path=OUT/f"cube_{cid:03d}_start_{i*64}_{j*64}_{k*64}.pt";torch.save({"mesh":m,"cube_id":cid,"start_C256":[i*64,j*64,k*64],"source_face_ids":ids},path);pieces.append((ids.numpy(),canonical_triangles(m)));records.append({"cube_id":cid,"start_C256":[i*64,j*64,k*64],"faces":len(m.faces),"vertices":len(m.vertices),"path":str(path.resolve())})
    original=canonical_triangles(source);rebuilt=np.empty_like(original)
    for ids,tri in pieces:rebuilt[ids]=tri
    ho=hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest();hr=hashlib.sha256(np.ascontiguousarray(rebuilt).tobytes()).hexdigest()
    manifest={"policy":"one camera-compatible fresh local decode; exact inverse similarity; triangle-centroid hard owner in 4x4x4 global C64 layout; triangles are never clipped","source_faces":len(f),"partition_faces":sum(x["faces"] for x in records),"face_coverage_exactly_once":sum(x["faces"] for x in records)==len(f),"geometry_sha256_original":ho,"geometry_sha256_partition_concat":hr,"geometry_byte_identical_after_partition":ho==hr,"active_cubes":records,"assembly":"load every cube mesh and concatenate; duplicate boundary vertex positions may optionally be index-welded with zero coordinate movement"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n");print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
