#!/usr/bin/env python3
"""Test variance-preserving gains for decoder-facing low/high SLATs.

The least-squares projection is mathematically unchanged; this probes the
decoder's learned feature scale, which is not part of the Fourier theorem.
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault("ATTN_BACKEND","flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from sparse_fourier_decode import payload, render_fixed, stats

def main(a):
    device=torch.device(a.device); source=torch.load(a.slat_cache,map_location="cpu",weights_only=False)
    coords=source["shape_slat"]["coords"].to(device=device,dtype=torch.int32)
    sf=source["shape_slat"]["feats"].to(device); tf=source["texture_slat"]["feats"].to(device)
    q=torch.load(a.projection,map_location="cpu",weights_only=False)
    sl=q["shape_low"].to(device); tl=q["texture_low"].to(device)
    pipeline=Pixal3DImageTo3DPipeline.from_pretrained(str(a.model_path)); pipeline._device=device; pipeline.low_vram=True
    rows=[]
    for gain in a.gains:
        # Channel-wise gain brings low-pass shape variance closer to the native latent.
        shape_mean=sf.mean(0,keepdim=True); tex_mean=tf.mean(0,keepdim=True)
        s=shape_mean+gain*(sl-shape_mean); t=tex_mean+gain*(tl-tex_mean)
        target=a.output/f"gain_{gain:g}"; target.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter()
        with torch.no_grad(): decoded=pipeline.decode_latent(SparseTensor(s,coords),SparseTensor(t,coords),a.decode_resolution)
        torch.cuda.synchronize(); mesh=decoded[0]; row={"gain":gain,**stats(mesh,time.perf_counter()-started),"shape_std":float(s.std().item()),"texture_std":float(t.std().item())}
        torch.save(payload(mesh),target/"mesh.pt"); render_fixed(mesh,target,device,a.render_resolution)
        (target/"metrics.json").write_text(json.dumps(row,indent=2)); rows.append(row)
        del mesh,decoded,s,t; torch.cuda.empty_cache()
    (a.output/"summary.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--slat-cache',type=Path,default=Path('outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt')); p.add_argument('--projection',type=Path,default=Path('sparse_fourier_validation/real_slat/projected_latents/rho_0.40.pt')); p.add_argument('--output',type=Path,default=Path('sparse_fourier_validation/real_slat/gain_sweep_rho_0.40')); p.add_argument('--model-path',type=Path,default=Path('/home/nvme04/yyyan/download/model/Pixal3D')); p.add_argument('--device',default='cuda'); p.add_argument('--decode-resolution',type=int,default=1024); p.add_argument('--render-resolution',type=int,default=256); p.add_argument('--gains',type=float,nargs='+',default=[1.,1.5,2.,2.5,3.]); return p.parse_args()
if __name__=='__main__': main(parse_args())
