#!/usr/bin/env python3
"""Real-SLAT decoder comparison for the three Codex.md baselines."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND","flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import torch
import torch.nn.functional as F

from sparse_fourier_validation import canonical_frequencies,fft_low,frequencies_for_cutoff,raw_nudft
from sparse_fourier_decode import payload,render_fixed,stats


def zero_fill(coords: torch.Tensor,features: torch.Tensor,resolution: int,cutoff: float) -> torch.Tensor:
    xyz=coords[:,-3:].long(); key=(xyz[:,0]*resolution+xyz[:,1])*resolution+xyz[:,2]
    volume=torch.zeros((resolution**3,features.shape[1]),device=features.device,dtype=features.dtype)
    volume[key]=features
    return fft_low(volume.reshape(resolution,resolution,resolution,-1),cutoff).reshape_as(volume)[key]


def local_average(coords: torch.Tensor,features: torch.Tensor,resolution: int,kernel: int=5) -> torch.Tensor:
    xyz=coords[:,-3:].long(); key=(xyz[:,0]*resolution+xyz[:,1])*resolution+xyz[:,2]
    flat=torch.zeros((resolution**3,features.shape[1]),device=features.device,dtype=features.dtype); flat[key]=features
    mask=torch.zeros((resolution**3,1),device=features.device,dtype=features.dtype); mask[key]=1
    volume=flat.reshape(resolution,resolution,resolution,-1).permute(3,0,1,2)[None]
    support=mask.reshape(resolution,resolution,resolution,1).permute(3,0,1,2)[None]
    numerator=F.avg_pool3d(volume,kernel,stride=1,padding=kernel//2)
    denominator=F.avg_pool3d(support,kernel,stride=1,padding=kernel//2)
    smooth=(numerator/(denominator+1e-12))[0].permute(1,2,3,0).reshape_as(flat)
    return smooth[key]


def main(args: argparse.Namespace) -> None:
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    device=torch.device(args.device); source=torch.load(args.slat_cache,map_location="cpu",weights_only=False)
    coords=source["shape_slat"]["coords"].to(device=device,dtype=torch.int32)
    shape=source["shape_slat"]["feats"].to(device); texture=source["texture_slat"]["feats"].to(device)
    points=(coords[:,-3:].to(torch.float64)+.5)/args.resolution-.5
    freq=frequencies_for_cutoff(canonical_frequencies(args.max_radius).to(device),args.cutoff)
    methods={
      "zero_fill_fft":(zero_fill(coords,shape,args.resolution,args.cutoff),zero_fill(coords,texture,args.resolution,args.cutoff)),
      "local_smoothing":(local_average(coords,shape,args.resolution),local_average(coords,texture,args.resolution)),
      "raw_nudft_mask":(raw_nudft(points,shape,freq),raw_nudft(points,texture,freq)),
    }
    pipeline=Pixal3DImageTo3DPipeline.from_pretrained(str(args.model_path)); pipeline._device=device; pipeline.low_vram=True
    rows=[]
    for name,(sf,tf) in methods.items():
        target=args.output/name; target.mkdir(parents=True,exist_ok=True); print(f"[baseline] {name}",flush=True)
        torch.save({"coords":coords.cpu(),"shape":sf.cpu(),"texture":tf.cpu(),"cutoff":args.cutoff},target/"latents.pt")
        started=time.perf_counter()
        with torch.no_grad(): decoded=pipeline.decode_latent(SparseTensor(sf,coords),SparseTensor(tf,coords),args.decode_resolution)
        torch.cuda.synchronize(); mesh=decoded[0]; row={"name":name,**stats(mesh,time.perf_counter()-started)}
        torch.save(payload(mesh),target/"mesh.pt"); render_fixed(mesh,target,device,args.render_resolution)
        (target/"metrics.json").write_text(json.dumps(row,indent=2),encoding="utf-8"); rows.append(row)
        del mesh,decoded; torch.cuda.empty_cache()
    summary={"cutoff":args.cutoff,"coordinates_unchanged":True,"methods":rows}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--slat-cache",type=Path,default=Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt"))
    p.add_argument("--output",type=Path,default=Path("sparse_fourier_validation/baselines/real_slat_cutoff_2")); p.add_argument("--model-path",type=Path,default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--device",default="cuda"); p.add_argument("--resolution",type=int,default=64); p.add_argument("--decode-resolution",type=int,default=1024)
    p.add_argument("--render-resolution",type=int,default=256); p.add_argument("--cutoff",type=float,default=2.0); p.add_argument("--max-radius",type=int,default=5)
    return p.parse_args()


if __name__=="__main__": main(parse_args())
