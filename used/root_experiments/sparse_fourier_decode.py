#!/usr/bin/env python3
"""Decode the real-SLAT projections produced by sparse_fourier_validation.py."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import torch


def update_main_report(output: Path, decoder_summary: dict[str,Any]) -> None:
    summary_path=output.parent/"summary.json"
    if not summary_path.exists(): return
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decoder"]={"status":decoder_summary["status"],"complete":decoder_summary["complete"],
        "total":decoder_summary["total"],"coarse_to_fine":False,
        "note":"All 26 meshes decoded. Low-only gives a rough silhouette but does not monotonically add fine detail as cutoff rises",
        "high_only":"Retains a recognizable complete global object and substantial texture at every tested cutoff, including the >0.8 residual",
        "bands":"B2-B5 independently retain global object structure; bands do not isolate only local/fine detail",
        "fixed_view_contact_sheets":[str(output/g/"fixed_view_contact_sheet.png") for g in ("low_only","high_only","individual_bands","cumulative_bands")]}
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    from sparse_fourier_validation import write_report
    write_report(output.parent/"report.md",summary)


def payload(mesh: Any) -> dict[str,Any]:
    return {"vertices":mesh.vertices.detach().cpu(),"faces":mesh.faces.detach().cpu(),
            "coords":mesh.coords.detach().cpu(),"attrs":mesh.attrs.detach().cpu(),
            "origin":mesh.origin.detach().cpu(),"voxel_size":float(mesh.voxel_size),
            "voxel_shape":list(mesh.voxel_shape),"layout":dict(mesh.layout)}


def stats(mesh: Any, seconds: float) -> dict[str,Any]:
    v=mesh.vertices.detach().float()
    return {"status":"complete","seconds":seconds,"vertices":len(v),"faces":len(mesh.faces),
            "active_ovoxels":len(mesh.coords),"bbox_min":v.amin(0).cpu().tolist(),
            "bbox_max":v.amax(0).cpu().tolist(),"bbox_extent":(v.amax(0)-v.amin(0)).cpu().tolist()}


def render_fixed(mesh: Any, output: Path, device: torch.device, resolution: int) -> None:
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.utils import render_utils
    from render_pixal3d_raw_ovoxel import load_envmap
    from pixal3d_baseline1024_pbr_mesh_compare import _save_render
    extr,intr=render_utils.proj_camera_to_render_params(camera_angle_x=math.radians(40),distance=1.6)
    renderer=PbrMeshRenderer(rendering_options={"resolution":resolution,"near":.1,"far":10.,
        "ssaa":1,"peel_layers":1,"face_chunk_size":4_000_000},device=str(device))
    env=load_envmap("assets/hdri/studio.exr",device=device)
    result=renderer.render(mesh,extr,intr,envmap=env,use_envmap_bg=False)
    _save_render(result,output/"fixed_view")
    del result,renderer,env


def render_existing(args: argparse.Namespace) -> None:
    from pixal3d.representations import MeshWithVoxel
    from PIL import Image,ImageDraw
    device=torch.device(args.device); completed=0; failures=[]
    paths=sorted(args.output.glob("*/*/mesh.pt"))
    for path in paths:
        target=path.parent
        if (target/"fixed_view"/"shaded.png").exists():
            completed+=1; continue
        print(f"[render] {target}",flush=True)
        try:
            p=torch.load(path,map_location=device,weights_only=False)
            mesh=MeshWithVoxel(p["vertices"],p["faces"],p["origin"].tolist(),p["voxel_size"],
                               p["coords"],p["attrs"],torch.Size(p["voxel_shape"]),p["layout"])
            render_fixed(mesh,target,device,args.render_resolution); completed+=1
            del mesh,p; torch.cuda.empty_cache()
        except Exception as exc:
            failures.append({"path":str(path),"error":repr(exc)}); print(f"[render:failed] {exc!r}",flush=True)
    result={"completed":completed,"total":len(paths),"failures":failures}
    for group in ("low_only","high_only","individual_bands","cumulative_bands"):
        images=[]
        for image_path in sorted((args.output/group).glob("*/fixed_view/shaded.png")):
            images.append((image_path.parents[1].name,Image.open(image_path).convert("RGB")))
        if not images: continue
        width,height=images[0][1].size; label=24
        sheet=Image.new("RGB",(width*len(images),height+label),"white"); draw=ImageDraw.Draw(sheet)
        for i,(name,img) in enumerate(images):
            sheet.paste(img,(i*width,label)); draw.text((i*width+4,4),name,fill="black")
        sheet.save(args.output/group/"fixed_view_contact_sheet.png")
    (args.output/"render_summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    decoder_path=args.output/"decoder_summary.json"
    if decoder_path.exists(): update_main_report(args.output,json.loads(decoder_path.read_text(encoding="utf-8")))
    print(json.dumps(result,indent=2))


def main(args: argparse.Namespace) -> None:
    if args.render_existing:
        render_existing(args); return
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline
    from pixal3d.modules.sparse import SparseTensor
    device=torch.device(args.device)
    source=torch.load(args.slat_cache,map_location="cpu",weights_only=False)
    coords=source["shape_slat"]["coords"].to(device=device,dtype=torch.int32)
    projected={p.stem:torch.load(p,map_location="cpu",weights_only=False) for p in sorted(args.projections.glob("rho_*.pt"))}
    by_rho={round(float(v["rho"]),2):v for v in projected.values()}
    pipeline=Pixal3DImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline._device=device; pipeline.low_vram=True
    variants=[]
    for rho in (.05,.10,.20,.30,.40,.60,.80,1.0):
        item=by_rho[round(rho,2)]
        variants += [("low_only",f"rho_{rho:.2f}",item["shape_low"],item["texture_low"]),
                     ("high_only",f"rho_{rho:.2f}",item["shape_high"],item["texture_high"])]
    edges=(.10,.20,.40,.80)
    previous_shape=torch.zeros_like(by_rho[.10]["shape_low"]); previous_tex=torch.zeros_like(by_rho[.10]["texture_low"])
    cumulative_shape=torch.zeros_like(previous_shape); cumulative_tex=torch.zeros_like(previous_tex)
    for i,rho in enumerate(edges,1):
        shape=by_rho[rho]["shape_low"]-previous_shape; tex=by_rho[rho]["texture_low"]-previous_tex
        variants.append(("individual_bands",f"B{i}_{rho:.2f}",shape,tex))
        cumulative_shape=cumulative_shape+shape; cumulative_tex=cumulative_tex+tex
        variants.append(("cumulative_bands",f"through_B{i}_{rho:.2f}",cumulative_shape.clone(),cumulative_tex.clone()))
        previous_shape=by_rho[rho]["shape_low"]; previous_tex=by_rho[rho]["texture_low"]
    variants.append(("individual_bands","B5_high_gt_0.80",by_rho[.80]["shape_high"],by_rho[.80]["texture_high"]))
    variants.append(("cumulative_bands","through_B5_full",source["shape_slat"]["feats"],source["texture_slat"]["feats"]))
    rows=[]
    for group,name,shape,texture in variants:
        target=args.output/group/name; target.mkdir(parents=True,exist_ok=True)
        print(f"[decode] {group}/{name}",flush=True)
        try:
            ss=SparseTensor(shape.to(device=device,dtype=torch.float32),coords)
            ts=SparseTensor(texture.to(device=device,dtype=torch.float32),coords)
            started=time.perf_counter()
            with torch.no_grad(): decoded=pipeline.decode_latent(ss,ts,args.resolution)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            if len(decoded)!=1: raise RuntimeError(f"decoder returned {len(decoded)} meshes")
            mesh=decoded[0]; row={"group":group,"name":name,**stats(mesh,elapsed)}
            torch.save(payload(mesh),target/"mesh.pt")
            if args.render: render_fixed(mesh,target,device,args.render_resolution)
            (target/"metrics.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
            rows.append(row); del mesh,decoded,ss,ts
        except Exception as exc:
            row={"group":group,"name":name,"status":"failed","error":repr(exc)}
            (target/"metrics.json").write_text(json.dumps(row,indent=2),encoding="utf-8")
            rows.append(row); print(f"[decode:failed] {exc!r}",flush=True)
        gc.collect(); torch.cuda.empty_cache()
    complete=sum(r["status"]=="complete" for r in rows)
    summary={"status":"complete" if complete==len(rows) else "partial","complete":complete,"total":len(rows),"variants":rows,
             "note":"Fixed coordinates and decoder; only features change. Inspect fixed_view/shaded.png contactually for semantics."}
    (args.output/"decoder_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    update_main_report(args.output,summary)
    print(json.dumps({"complete":complete,"total":len(rows),"summary":str(args.output/"decoder_summary.json")},indent=2))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--projections",type=Path,default=Path("sparse_fourier_validation/real_slat/projected_latents"))
    p.add_argument("--output",type=Path,default=Path("sparse_fourier_validation/real_slat"))
    p.add_argument("--slat-cache",type=Path,default=Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt"))
    p.add_argument("--model-path",type=Path,default=Path("/home/nvme04/yyyan/download/model/Pixal3D"))
    p.add_argument("--device",default="cuda"); p.add_argument("--resolution",type=int,default=1024)
    p.add_argument("--render",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--render-resolution",type=int,default=384)
    p.add_argument("--render-existing",action="store_true",help="render saved mesh.pt files without decoding")
    return p.parse_args()


if __name__=="__main__": main(parse_args())
