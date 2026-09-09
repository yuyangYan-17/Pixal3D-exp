#!/usr/bin/env python3
"""Fresh SS interior plus a small inherited neck-interface C32 context."""
from __future__ import annotations
import json,math,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
import run_head_similarity_experiment as exp
import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled

ROOT=Path(__file__).resolve().parent; BASE=ROOT/"outputs/head_similarity_fresh_vs_inherited_cuda4/phase_bc";OUT=ROOT/"outputs/head_gated_fresh_support_cuda4"

def ordered_union(reference:torch.Tensor,candidate:torch.Tensor)->torch.Tensor:
    seen={tuple(x) for x in reference.cpu().tolist()};extra=[x for x in candidate.cpu().tolist() if tuple(x) not in seen]
    return torch.cat((reference.cpu(),torch.tensor(extra,dtype=reference.dtype)),0)

def ordered_like(reference:torch.Tensor,candidate:torch.Tensor)->torch.Tensor:
    pool={tuple(x):x for x in candidate.cpu().tolist()};shared=[pool[tuple(x)] for x in reference.cpu().tolist() if tuple(x) in pool];refset={tuple(x) for x in reference.cpu().tolist()};extra=[x for x in candidate.cpu().tolist() if tuple(x) not in refset]
    return torch.tensor(shared+extra,dtype=reference.dtype)

@torch.no_grad()
def main():
    from inference import init_pipeline
    os.environ.setdefault("ATTN_BACKEND","flash_attn");device=torch.device("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:4");torch.cuda.set_device(device);OUT.mkdir(parents=True,exist_ok=True)
    setup=json.loads((BASE/"local_setup.json").read_text());cl=setup["local_camera"];R=np.array(setup["R_global_to_local"]);b=np.array(setup["b_global"]);s=setup["side_s"]
    fresh=torch.load(BASE/"support/B_fresh_c32.pt",map_location="cpu",weights_only=False)["coords"].int();inh=torch.load(BASE/"support/C_inherited_c32.pt",map_location="cpu",weights_only=False)["coords"].int()
    A=fresh[:,1:].numpy();I=inh[:,1:].numpy();dist,_=cKDTree(A).query(I);g=b+s*((I+.5)/32-.5)@R
    # One face/edge-neighbour halo, limited to the global-X neck/interface side.
    keep=(dist<=math.sqrt(2)+1e-6)&(g[:,0]<.25);candidate=inh[torch.from_numpy(keep)];gated=ordered_union(fresh,candidate).to(device)
    image1024=Image.open(BASE/"inputs/fixed_camera_1024.png").convert("RGB");image512=image1024.resize((512,512),Image.Resampling.LANCZOS)
    pipe=init_pipeline("/home/nvme04/yyyan/download/model/Pixal3D",device=str(device),low_vram=True);pipe.shape_slat_sampler_params["steps"]=12
    cond=pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512,[image512],gated,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.)
    torch.manual_seed(exp.SEEDS["shape512"]);shape32=pipe.sample_shape_slat(cond,pipe.models["shape_slat_flow_model_512"],gated);exp.sparse_save(OUT/"shape512_endpoint.pt",shape32)
    raw64=tiled.upsample_and_quantize(pipe,shape32,512,64,True).cpu();fresh64=torch.load(BASE/"B_fresh/c64_support.pt",map_location="cpu",weights_only=False)["coords"].int();gated64=ordered_like(fresh64,raw64).to(device)
    cond64=pipe.get_proj_cond_shape(pipe.image_cond_model_shape_1024,[image1024],gated64,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.,grid_resolution_override=64)
    torch.manual_seed(exp.SEEDS["shape1024"]);shape64=pipe.sample_shape_slat(cond64,pipe.models["shape_slat_flow_model_1024"],gated64);exp.sparse_save(OUT/"shape1024_endpoint.pt",shape64)
    meshes,_=pipe.decode_shape_slat(shape64,1024);m=meshes[0].cpu();m.vertices=(torch.tensor(b)+s*(m.vertices.double()@torch.tensor(R,dtype=torch.double))).float();torch.save({"mesh":m,"setup":setup},OUT/"gated_fresh_mesh_global.pt")
    baseline=torch.load(BASE/"B_fresh/native_mesh_global_similarity.pt",map_location="cpu",weights_only=False)["mesh"]
    exp.render_fixed_multiview({"fresh_baseline":baseline,"gated_fresh_interface":m},setup,OUT,device)
    stats={"policy":"all fresh C32 retained in original order; append only inherited tokens within sqrt(2) cells of fresh and global X<0.25 neck side",
      "paired_noise":"fresh C32 rows remain first and in identical order; seed 4202 gives identical initial noise on every fresh row. At C64, decoder-produced rows shared with baseline fresh C64 are placed first in baseline order before seed 4203.",
      "fresh_C32":len(fresh),"candidate_inherited_rows":int(keep.sum()),"gated_unique_C32":len(gated),"added_unique_C32":len(gated)-len(fresh),"fresh_C64":len(fresh64),"gated_C64":len(gated64),"shared_C64":len({tuple(x) for x in fresh64.tolist()}&{tuple(x) for x in gated64.cpu().tolist()}),"new_vs_fresh_C64":len({tuple(x) for x in gated64.cpu().tolist()}-{tuple(x) for x in fresh64.tolist()}),
      "gated_C32_topology":exp.support_stats(gated.cpu(),32,OUT/"support","gated_C32"),"gated_C64_topology":exp.support_stats(gated64.cpu(),64,OUT/"support","gated_C64"),"vertices":len(m.vertices),"faces":len(m.faces)}
    (OUT/"metrics.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False)+"\n");print(json.dumps(stats,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
