#!/usr/bin/env python3
"""Visualize one arbitrary C64 cube inside the full global C256 SLAT support."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from pixal3d_baseline1024_pbr_mesh_compare import _make_camera_views


FORMAT = "pixal3d_global_c256_single_head_cube_visualization_v1"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); temporary = Path(f.name)
    os.replace(temporary, path)


def parse_triplet(value: str) -> tuple[int, int, int]:
    result = tuple(int(v.strip()) for v in value.split(","))
    if len(result) != 3: raise ValueError("expected x,y,z")
    return result


def project(points: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor,
            resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = torch.cat((points.float(), torch.ones((points.shape[0], 1))), 1)
    camera = p @ extrinsic.detach().cpu().float().T
    z = camera[:, 2]
    u = (intrinsic[0, 0].cpu()*camera[:,0]/z + intrinsic[0,2].cpu())*resolution
    v = (intrinsic[1, 1].cpu()*camera[:,1]/z + intrinsic[1,2].cpu())*resolution
    valid = torch.isfinite(u)&torch.isfinite(v)&(z>0)&(u>=0)&(u<resolution)&(v>=0)&(v<resolution)
    return torch.stack((u[valid],v[valid]),1).numpy(), z[valid].numpy(), torch.where(valid)[0].numpy()


def project_unclipped(points: torch.Tensor, extrinsic: torch.Tensor,
                      intrinsic: torch.Tensor, resolution: int) -> np.ndarray:
    p=torch.cat((points.float(),torch.ones((points.shape[0],1))),1)
    camera=p@extrinsic.detach().cpu().float().T;z=camera[:,2]
    if not bool((z>0).all()): raise RuntimeError("cube corner lies behind camera")
    u=(intrinsic[0,0].cpu()*camera[:,0]/z+intrinsic[0,2].cpu())*resolution
    v=(intrinsic[1,1].cpu()*camera[:,1]/z+intrinsic[1,2].cpu())*resolution
    return torch.stack((u,v),1).numpy()


def cube_edges() -> tuple[list[tuple[int,int,int]], list[tuple[int,int]]]:
    corners = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    edges = []
    for i,a in enumerate(corners):
        for j,b in enumerate(corners):
            if j>i and sum(int(x!=y) for x,y in zip(a,b))==1: edges.append((i,j))
    return corners,edges


def make_view(points: torch.Tensor, inside: torch.Tensor, cube_world: torch.Tensor,
              extrinsic: torch.Tensor, intrinsic: torch.Tensor, resolution: int,
              point_radius: int, angle: int, start: tuple[int,int,int]) -> Image.Image:
    uv,depth,ids=project(points,extrinsic,intrinsic,resolution)
    order=np.argsort(depth)[::-1]; uv=uv[order]; depth=depth[order]; ids=ids[order]
    near,far=float(depth.min()),float(depth.max()); shade=1-(depth-near)/max(far-near,1e-8)
    image=Image.new("RGB",(resolution,resolution),(0,0,0)); draw=ImageDraw.Draw(image,"RGBA")
    inside_np=inside.numpy()
    for (u,v),value,idx in zip(uv,shade,ids):
        if inside_np[idx]:
            color=(255,int(95+120*value),35,245); radius=point_radius+1
        else:
            c=int(75+145*value); color=(c,c,c,180); radius=point_radius
        draw.ellipse((u-radius,v-radius,u+radius,v+radius),fill=color)
    projected=project_unclipped(cube_world,extrinsic,intrinsic,resolution)
    _,edges=cube_edges()
    for i,j in edges:
        draw.line((tuple(projected[i]),tuple(projected[j])),fill=(255,220,0,255),width=max(4,resolution//700))
    try: font=ImageFont.truetype("DejaVuSans-Bold.ttf",max(24,resolution//85))
    except OSError: font=ImageFont.load_default()
    label=f"yaw {angle} deg | C64 start={start} end={tuple(v+64 for v in start)}"
    draw.rounded_rectangle((30,25,30+draw.textlength(label,font=font)+30,25+font.size+30),radius=12,fill=(0,0,0,200))
    draw.text((45,40),label,font=font,fill=(255,255,255,255))
    return image


def contact_sheet(paths: list[tuple[int,Path]], output: Path) -> None:
    thumb=768;margin=20;label=42
    sheet=Image.new("RGB",(len(paths)*(thumb+margin)+margin,thumb+label+2*margin),(16,16,16));draw=ImageDraw.Draw(sheet)
    for index,(angle,path) in enumerate(paths):
        with Image.open(path) as source: image=source.convert("RGB")
        x=margin+index*(thumb+margin);y=margin+label
        sheet.paste(image.resize((thumb,thumb),Image.Resampling.LANCZOS),(x,y))
        draw.text((x,margin),f"Full C256 SLAT + one C64 wire cube | yaw {angle}°",fill="white")
    sheet.save(output)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support",type=Path,default=Path("outputs/global_c256_cube_owner_flow_singleview_cuda4/support/global_c256_support.pt"))
    p.add_argument("--camera",type=Path,default=Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/global_camera.json"))
    p.add_argument("--start",default="162,82,145")
    p.add_argument("--angles",default="0,60,120,180,240,300")
    p.add_argument("--resolution",type=int,default=4096)
    p.add_argument("--point-radius",type=int,default=3)
    p.add_argument("--output-dir",type=Path,default=Path("outputs/global_c256_single_head_cube_162_82_145_visualization"))
    args=p.parse_args();start=parse_triplet(args.start);angles=tuple(int(v) for v in args.angles.split(","))
    if any(v<0 or v+64>256 for v in start): raise ValueError("C64 cube lies outside C256")
    payload=torch.load(args.support,map_location="cpu",weights_only=False);coords=payload["coords"].int();xyz=coords[:,1:].float()
    s=torch.tensor(start).float();inside=((xyz>=s)&(xyz<s+64)).all(1)
    camera=json.loads(args.camera.read_text());scale=float(camera.get("mesh_scale",1.0))
    points=(2*(xyz+.5)/256-1)/(2*scale)
    corners,_=cube_edges();boundary=s[None]+torch.tensor(corners).float()*64
    cube_world=(2*boundary/256-1)/(2*scale)
    extrinsics,intrinsic,_=_make_camera_views(camera["camera_angle_x"],camera["distance"],angles)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True);views=[]
    for angle in angles:
        print(f"[view] yaw={angle} inside={int(inside.sum()):,}",flush=True)
        image=make_view(points,inside,cube_world,extrinsics[angle],intrinsic,args.resolution,args.point_radius,angle,start)
        path=output/f"view_{angle:03d}_single_head_cube_wireframe.png";image.save(path);views.append((angle,path))
    sheet=output/"single_head_cube_wireframe_contact_sheet.png";contact_sheet(views,sheet)
    atomic_json(output/"manifest.json",{"format":FORMAT,"status":"complete","start_c256":list(start),
        "end_exclusive_c256":[v+64 for v in start],"start_c4096":[v*16 for v in start],
        "end_exclusive_c4096":[(v+64)*16 for v in start],"cube_size_c256":64,"cube_size_c4096":1024,
        "global_support_tokens":int(len(coords)),"inside_tokens":int(inside.sum()),"angles":list(angles),
        "resolution":args.resolution,"contact_sheet":str(sheet.resolve())})
    print(f"[done] {sheet}",flush=True)


if __name__=="__main__":main()
