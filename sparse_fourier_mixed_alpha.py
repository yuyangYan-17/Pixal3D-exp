#!/usr/bin/env python3
"""Probe separate shape/texture residual schedules for decoder progression."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('ATTN_BACKEND','flash_attn'); os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import torch
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from sparse_fourier_decode import payload,render_fixed,stats

def main(a):
 d=torch.device(a.device); src=torch.load(a.slat_cache,map_location='cpu',weights_only=False); q=torch.load(a.projection,map_location='cpu',weights_only=False)
 c=src['shape_slat']['coords'].to(device=d,dtype=torch.int32); fs=src['shape_slat']['feats'].to(d); ft=src['texture_slat']['feats'].to(d); sl=q['shape_low'].to(d); tl=q['texture_low'].to(d)
 pipe=Pixal3DImageTo3DPipeline.from_pretrained(str(a.model_path)); pipe._device=d; pipe.low_vram=True; rows=[]
 for ash,atx in a.pairs:
  s=sl+ash*(fs-sl); t=tl+atx*(ft-tl); name=f'shape_{ash:g}_tex_{atx:g}'; target=a.output/name; target.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
  with torch.no_grad(): decoded=pipe.decode_latent(SparseTensor(s,c),SparseTensor(t,c),a.decode_resolution)
  torch.cuda.synchronize(); m=decoded[0]; row={'name':name,'alpha_shape':ash,'alpha_texture':atx,**stats(m,time.perf_counter()-start)}; torch.save(payload(m),target/'mesh.pt'); render_fixed(m,target,d,a.render_resolution); (target/'metrics.json').write_text(json.dumps(row,indent=2)); rows.append(row); del m,decoded,s,t; torch.cuda.empty_cache()
 (a.output/'summary.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--slat-cache',type=Path,default=Path('outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt')); p.add_argument('--projection',type=Path,default=Path('sparse_fourier_validation/real_slat/projected_latents/rho_0.40.pt')); p.add_argument('--output',type=Path,default=Path('sparse_fourier_validation/real_slat/mixed_alpha_rho_0.40')); p.add_argument('--model-path',type=Path,default=Path('/home/nvme04/yyyan/download/model/Pixal3D')); p.add_argument('--device',default='cuda'); p.add_argument('--decode-resolution',type=int,default=1024); p.add_argument('--render-resolution',type=int,default=256); p.add_argument('--pairs',type=float,nargs=2,action='append',default=[[0.,0.],[0.,1.],[.5,0.],[.5,1.],[.75,.25],[1.,.25]]); return p.parse_args()
if __name__=='__main__': main(parse_args())
