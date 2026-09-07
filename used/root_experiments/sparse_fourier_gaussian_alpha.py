#!/usr/bin/env python3
"""Support-normalized Euclidean Gaussian scale space and low+alpha*high decode.

This is a density-corrected continuous-space baseline, not a graph filter:
L_sigma(p_i)=sum_j exp(-||p_i-p_j||^2/2sigma^2)F_j / sum_j exp(...).
The decoder is always given the original coordinates.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
os.environ.setdefault("ATTN_BACKEND","flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from sparse_fourier_decode import payload, render_fixed, stats


@torch.no_grad()
def gaussian_scale(points: torch.Tensor, features: torch.Tensor, sigma_vox: float,
                   resolution: int, chunk: int=1024) -> torch.Tensor:
    p=points.to(torch.float32); f=features.to(torch.float32)
    sigma=float(sigma_vox)/float(resolution); out=[]
    for start in range(0,len(p),chunk):
        d2=torch.cdist(p[start:start+chunk],p).square()
        weights=torch.exp(-d2/(2*sigma*sigma))
        out.append((weights@f)/(weights.sum(1,keepdim=True)+1e-12))
    return torch.cat(out).to(features.dtype)


@torch.no_grad()
def carrier_preserving_gaussian(points: torch.Tensor, features: torch.Tensor,
                                sigma_vox: float, resolution: int, chunk: int=1024) -> torch.Tensor:
    """Smooth only modulation around the object sample mean, then restore DC."""
    carrier=features.mean(0,keepdim=True)
    low=carrier+gaussian_scale(points,features-carrier,sigma_vox,resolution,chunk)
    # Nonuniform boundaries can move the mean even after density normalization.
    return low+(features.mean(0,keepdim=True)-low.mean(0,keepdim=True))


def main(a):
    device=torch.device(a.device); source=torch.load(a.slat_cache,map_location="cpu",weights_only=False)
    coords=source["shape_slat"]["coords"].to(device=device,dtype=torch.int32)
    sf=source["shape_slat"]["feats"].to(device); tf=source["texture_slat"]["feats"].to(device)
    points=(coords[:,-3:].to(torch.float64)+.5)/a.resolution-.5
    pipeline=Pixal3DImageTo3DPipeline.from_pretrained(str(a.model_path)); pipeline._device=device; pipeline.low_vram=True
    rows=[]
    for sigma in a.sigmas:
        low_s=carrier_preserving_gaussian(points,sf,sigma,a.resolution,a.chunk); low_t=carrier_preserving_gaussian(points,tf,sigma,a.resolution,a.chunk)
        high_s=sf-low_s; high_t=tf-low_t
        sigma_dir=a.output/f"sigma_{sigma:g}"; sigma_dir.mkdir(parents=True,exist_ok=True)
        torch.save({"coords":coords.cpu(),"shape_low":low_s.cpu(),"shape_high":high_s.cpu(),"texture_low":low_t.cpu(),"texture_high":high_t.cpu(),"sigma_voxels":sigma},sigma_dir/"latents.pt")
        for alpha in a.alphas:
            shape=low_s+float(alpha)*high_s; tex=low_t+float(alpha)*high_t
            target=sigma_dir/f"alpha_{alpha:g}"; target.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            with torch.no_grad(): decoded=pipeline.decode_latent(SparseTensor(shape,coords),SparseTensor(tex,coords),a.decode_resolution)
            torch.cuda.synchronize(); mesh=decoded[0]
            row={"sigma_voxels":sigma,"alpha":alpha,**stats(mesh,time.perf_counter()-started),"shape_std":float(shape.std().item()),"texture_std":float(tex.std().item())}
            torch.save(payload(mesh),target/"mesh.pt")
            if alpha in (0.0,1.0) or sigma==a.sigmas[len(a.sigmas)//2]: render_fixed(mesh,target,device,a.render_resolution)
            (target/"metrics.json").write_text(json.dumps(row,indent=2)); rows.append(row)
            del mesh,decoded,shape,tex; torch.cuda.empty_cache()
        del low_s,low_t,high_s,high_t
    (a.output/"summary.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--slat-cache',type=Path,default=Path('outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt')); p.add_argument('--output',type=Path,default=Path('sparse_fourier_validation/real_slat/gaussian_alpha')); p.add_argument('--model-path',type=Path,default=Path('/home/nvme04/yyyan/download/model/Pixal3D')); p.add_argument('--device',default='cuda'); p.add_argument('--resolution',type=int,default=64); p.add_argument('--decode-resolution',type=int,default=1024); p.add_argument('--render-resolution',type=int,default=256); p.add_argument('--chunk',type=int,default=1024); p.add_argument('--sigmas',type=float,nargs='+',default=[.5,1.,2.,4.,8.]); p.add_argument('--alphas',type=float,nargs='+',default=[0.,.25,.5,.75,1.]); return p.parse_args()
if __name__=='__main__': main(parse_args())
