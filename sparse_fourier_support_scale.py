#!/usr/bin/env python3
"""Exploratory support-scale decomposition for decoder-compatible SLATs.

This is intentionally separate from the strict fixed-support Fourier result.
It tests whether the active coordinates themselves carry geometry detail.  A
low variant uses a nested dense-support subset; alpha>0 reintroduces removed
tokens and their feature residuals, reaching the original SLAT exactly at 1.
"""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
os.environ.setdefault("ATTN_BACKEND","flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from sparse_fourier_decode import payload,render_fixed,stats
from sparse_fourier_gaussian_alpha import carrier_preserving_gaussian

@torch.no_grad()
def density(points,sigma_vox,resolution,chunk=1024):
    p=points.float(); sig=sigma_vox/resolution; out=[]
    for st in range(0,len(p),chunk):
        d2=torch.cdist(p[st:st+chunk],p).square(); out.append(torch.exp(-d2/(2*sig*sig)).sum(1))
    return torch.cat(out)

def main(a):
    device=torch.device(a.device); src=torch.load(a.slat_cache,map_location='cpu',weights_only=False)
    coords=src['shape_slat']['coords'].to(device=device,dtype=torch.int32); sf=src['shape_slat']['feats'].to(device); tf=src['texture_slat']['feats'].to(device)
    points=(coords[:,-3:].float()+.5)/a.resolution-.5
    score=density(points,a.density_sigma,a.resolution,a.chunk)
    order=torch.argsort(score,descending=True)
    low_s=carrier_preserving_gaussian(points,sf,a.feature_sigma,a.resolution,a.chunk); low_t=carrier_preserving_gaussian(points,tf,a.feature_sigma,a.resolution,a.chunk)
    pipeline=Pixal3DImageTo3DPipeline.from_pretrained(str(a.model_path)); pipeline._device=device; pipeline.low_vram=True
    rows=[]
    for keep in a.keep_ratios:
        n=max(1,round(len(coords)*keep)); kept=order[:n]; removed=order[n:]
        for alpha in a.alphas:
            if alpha==0:
                c=coords[kept]; s=low_s[kept]; t=low_t[kept]
            else:
                # Union keeps fixed ordering unnecessary; coalesce-free support is valid.
                c=torch.cat((coords[kept],coords[removed]),0)
                s=torch.cat((low_s[kept]+alpha*(sf[kept]-low_s[kept]),alpha*sf[removed]),0)
                t=torch.cat((low_t[kept]+alpha*(tf[kept]-low_t[kept]),alpha*tf[removed]),0)
            target=a.output/f"keep_{keep:g}"/f"alpha_{alpha:g}"; target.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            with torch.no_grad(): decoded=pipeline.decode_latent(SparseTensor(s,c),SparseTensor(t,c),a.decode_resolution)
            torch.cuda.synchronize(); mesh=decoded[0]; row={'keep_ratio':keep,'alpha':alpha,'support_tokens':len(c),**stats(mesh,time.perf_counter()-started)}
            torch.save(payload(mesh),target/'mesh.pt')
            if alpha in (0.,1.): render_fixed(mesh,target,device,a.render_resolution)
            (target/'metrics.json').write_text(json.dumps(row,indent=2)); rows.append(row)
            del mesh,decoded,s,t,c; torch.cuda.empty_cache()
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--slat-cache',type=Path,default=Path('outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt')); p.add_argument('--output',type=Path,default=Path('sparse_fourier_validation/real_slat/support_scale')); p.add_argument('--model-path',type=Path,default=Path('/home/nvme04/yyyan/download/model/Pixal3D')); p.add_argument('--device',default='cuda'); p.add_argument('--resolution',type=int,default=64); p.add_argument('--decode-resolution',type=int,default=1024); p.add_argument('--render-resolution',type=int,default=256); p.add_argument('--density-sigma',type=float,default=2.); p.add_argument('--feature-sigma',type=float,default=1.); p.add_argument('--chunk',type=int,default=1024); p.add_argument('--keep-ratios',type=float,nargs='+',default=[.70,.85,.95]); p.add_argument('--alphas',type=float,nargs='+',default=[0.,.25,.5,.75,1.]); return p.parse_args()
if __name__=='__main__': main(parse_args())
