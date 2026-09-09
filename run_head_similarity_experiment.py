#!/usr/bin/env python3
"""Strict single-head similarity experiment requested by ``codex.md``.

The driver is deliberately resumable.  ``--phase a`` only consumes the two
existing native meshes.  ``--phase bc`` constructs one metric local cube and
runs the native cascade twice, changing only the C32 support source.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
GLOBAL = ROOT / "outputs/baseline1024_c128_8xc64_geometry_cuda4"
LOCAL = ROOT / "outputs/c128_head_crop_baseline1024_geometry_cuda4"
CROP = ROOT / "outputs/c128_head_crop_4096"
DEFAULT_OUT = ROOT / "outputs/head_similarity_fresh_vs_inherited_cuda4"
SEEDS = {"ss": 4201, "shape512": 4202, "shape1024": 4203}


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def jsonable(x: Any) -> Any:
    if isinstance(x, Path): return str(x.resolve())
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def load_mesh(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)["mesh"]


def camera_K(camera: Mapping[str, float], resolution: int) -> np.ndarray:
    f = resolution / (2.0 * math.tan(float(camera["camera_angle_x"]) / 2.0))
    return np.array([[f, 0, resolution / 2], [0, f, resolution / 2], [0, 0, 1]], np.float64)


def recover_crop_transform(out: Path) -> dict[str, Any]:
    """Recover the actually executed crop/recenter affine from image evidence."""
    manifest = json.loads((CROP / "manifest.json").read_text())
    source = cv2.imread(str(CROP / "turtle_head_1024.png"), cv2.IMREAD_GRAYSCALE)
    final = cv2.imread(str(LOCAL / "inputs/canonical_1024.png"), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=12000)
    ka, da = sift.detectAndCompute(source, None)
    kb, db = sift.detectAndCompute(final, None)
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for a, b in pairs if a.distance < .7 * b.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    partial, inliers = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=.5, maxIters=20000
    )
    if partial is None or int(inliers.sum()) < 100:
        raise RuntimeError("could not recover local foreground recenter transform")
    # The source implementation permits only isotropic resize + translation.
    scale = float((partial[0, 0] + partial[1, 1]) / 2)
    local_from_raw_crop = np.array([[scale, 0, partial[0, 2]], [0, scale, partial[1, 2]], [0, 0, 1.]])
    x0, y0, _, _ = manifest["crop_xyxy_4096"]
    raw_crop_from_global = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]])
    A = local_from_raw_crop @ raw_crop_from_global
    pred = (local_from_raw_crop @ np.c_[pa, np.ones(len(pa))].T).T[:, :2]
    residual = np.linalg.norm(pred - pb, axis=1)
    info = {
        "definition": "u_local = A @ u_global4096 (homogeneous pixel-center coordinates)",
        "global_to_raw_crop": raw_crop_from_global,
        "raw_crop_to_final_1024": local_from_raw_crop,
        "A_global4096_to_final1024": A,
        "evidence": "SIFT correspondences between saved raw crop and saved final canonical input, constrained to the source-code isotropic-resize+translation family",
        "matches": len(good), "ransac_inliers": int(inliers.sum()),
        "median_reprojection_px": float(np.median(residual[inliers[:, 0] > 0])),
        "inferred_preprocess_square_extent_in_raw_crop": [
            float(-partial[0, 2] / scale), float(-partial[1, 2] / scale),
            float((1024 - partial[0, 2]) / scale), float((1024 - partial[1, 2]) / scale),
        ],
    }
    atomic_json(out / "phase_a/preprocess_transform.json", jsonable(info))
    return info


def make_camera(camera: Mapping[str, float], device: torch.device):
    import utils3d
    C = torch.tensor([0., 0., float(camera["distance"])], device=device)
    E = utils3d.torch.extrinsics_look_at(C, torch.zeros(3, device=device), torch.tensor([0., 1., 0.], device=device))
    fov = torch.tensor(float(camera["camera_angle_x"]), device=device)
    return E, utils3d.torch.intrinsics_from_fov_xy(fov, fov)


@torch.no_grad()
def render_buffers(mesh, camera, device, resolution=1024):
    from pixal3d.renderers import MeshRenderer
    E, K = make_camera(camera, device)
    renderer = MeshRenderer({"resolution": resolution, "near": .01, "far": 10., "ssaa": 1,
                             "chunk_size": 4_000_000, "antialias": False}, device=str(device))
    m = mesh.to(device)
    r = renderer.render(m, E, K, return_types=["coord", "depth", "normal", "mask"])
    answer = {k: v.detach().float().cpu() for k, v in r.items()}
    del r, renderer, m
    torch.cuda.empty_cache()
    return answer


def sample_map(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    # image C,H,W or H,W; xy is in pixel coordinates.
    if image.ndim == 2: image = image[None]
    h, w = image.shape[-2:]
    grid = torch.stack((2 * (xy[:, 0] + .5) / w - 1, 2 * (xy[:, 1] + .5) / h - 1), -1)
    return F.grid_sample(image[None], grid[None, None], mode="bilinear", align_corners=False)[0, :, 0].T


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None):
    n = len(src); w = np.ones(n) if weights is None else weights.astype(np.float64)
    w /= w.sum(); a = (src * w[:, None]).sum(0); b = (dst * w[:, None]).sum(0)
    x, y = src - a, dst - b
    U, S, Vt = np.linalg.svd((y * w[:, None]).T @ x)
    D = np.eye(3); D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / np.sum(w * np.sum(x*x, axis=1)))
    return scale, R, b - scale * (R @ a)


def robust_similarity(src: np.ndarray, dst: np.ndarray):
    s, R, t = umeyama(src, dst)
    for _ in range(12):
        e = np.linalg.norm(s * (src @ R.T) + t - dst, axis=1)
        med = np.median(e); sigma = 1.4826 * np.median(np.abs(e-med)) + 1e-9
        w = 1 / (1 + (e / (2.5*sigma))**2)
        w[e > np.quantile(e, .85)] = 0
        s, R, t = umeyama(src, dst, w)
    return s, R, t


def metric_stats(e: np.ndarray) -> dict[str, float]:
    return {"count": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.quantile(e, .9)), "p95": float(np.quantile(e, .95))}


def save_buffer_images(root: Path, prefix: str, b: Mapping[str, torch.Tensor]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mask = b["mask"].numpy() > .5
    normal = np.moveaxis(b["normal"].numpy(), 0, -1)
    Image.fromarray(np.uint8(np.clip(normal, 0, 1)*255)).save(root / f"{prefix}_normal.png")
    depth = b["depth"].numpy(); good = mask & np.isfinite(depth)
    vis = np.zeros_like(depth)
    if good.any():
        lo, hi = np.quantile(depth[good], [.01, .99]); vis[good] = np.clip((depth[good]-lo)/(hi-lo+1e-9),0,1)
    Image.fromarray(np.uint8(vis*255)).save(root / f"{prefix}_depth.png")
    shaded = .18 + .72 * np.clip(normal[..., 2], 0, 1); shaded[~mask] = 1
    Image.fromarray(np.uint8(shaded*255)).convert("RGB").save(root / f"{prefix}_gray.png")
    Image.fromarray(np.uint8(mask)*255).save(root / f"{prefix}_mask.png")


def support_stats(coords: torch.Tensor, resolution: int, root: Path, name: str) -> dict[str, Any]:
    xyz = coords[:, 1:].detach().cpu().int().numpy(); points = {tuple(x) for x in xyz.tolist()}
    degrees=[]; components=[]; unseen=set(points)
    for p in points:
        degrees.append(sum((p[0]+dx,p[1]+dy,p[2]+dz) in points for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))))
    while unseen:
        seed=unseen.pop(); stack=[seed]; n=1
        while stack:
            p=stack.pop()
            for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                q=(p[0]+d[0],p[1]+d[1],p[2]+d[2])
                if q in unseen: unseen.remove(q); stack.append(q); n+=1
        components.append(n)
    boundary=((xyz==0)|(xyz==resolution-1)).any(1)
    hist=np.bincount(degrees,minlength=7)
    root.mkdir(parents=True,exist_ok=True)
    for axis,label in enumerate("xyz"):
        other=[i for i in range(3) if i!=axis]; canvas=np.zeros((resolution,resolution),np.uint8)
        canvas[xyz[:,other[1]],xyz[:,other[0]]]=255
        Image.fromarray(canvas).resize((512,512),Image.Resampling.NEAREST).save(root/f"{name}_occupancy_{label}.png")
    return {"count":len(xyz),"six_neighbor_degree_histogram":hist.tolist(),"connected_components":len(components),
            "largest_component":max(components,default=0),"boundary_token_ratio":float(boundary.mean()) if len(boundary) else 0.}


def sparse_save(path: Path, st) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"coords":st.coords.detach().cpu(),"feats":st.feats.detach().cpu()},path)


def cond_compare(a: Mapping[str,Any], b: Mapping[str,Any], overlap: torch.Tensor) -> dict[str,Any]:
    ga=a["cond"]["global"].detach().float(); gb=b["cond"]["global"].detach().float()
    ca=a["cond"]["proj"].coords.detach().cpu(); cb=b["cond"]["proj"].coords.detach().cpu()
    fa=a["cond"]["proj"].feats.detach().float(); fb=b["cond"]["proj"].feats.detach().float()
    def key(x): return tuple(int(v) for v in x.tolist())
    ia={key(x):i for i,x in enumerate(ca)}; ib={key(x):i for i,x in enumerate(cb)}
    keys=[key(x) for x in overlap.detach().cpu()]; xa=torch.stack([fa[ia[k]] for k in keys]); xb=torch.stack([fb[ib[k]] for k in keys])
    delta=xa-xb; rel=float(delta.norm()/max(float(xa.norm()),1e-12)); cos=F.cosine_similarity(xa.reshape(len(xa),-1),xb.reshape(len(xb),-1),dim=1)
    gd=ga-gb
    return {"overlap_count":len(keys),"global_max_abs":float(gd.abs().max()),"global_relative_l2":float(gd.norm()/max(float(ga.norm()),1e-12)),
            "projected_max_abs":float(delta.abs().max()),"projected_relative_l2":rel,"projected_cosine_min":float(cos.min()),"projected_cosine_mean":float(cos.mean()),
            "projected_pixel_coordinates":"identical by construction: same integer coordinate, grid resolution, physical camera and image"}


def intersect_coords(a: torch.Tensor,b: torch.Tensor) -> torch.Tensor:
    sa={tuple(x) for x in a.detach().cpu().tolist()}; sb={tuple(x) for x in b.detach().cpu().tolist()}
    rows=sorted(sa&sb); return torch.tensor(rows,dtype=torch.int32,device=a.device)


def support_difference_image(a: torch.Tensor,b: torch.Tensor,resolution:int,path:Path) -> None:
    sa={tuple(x[1:]) for x in a.detach().cpu().tolist()}; sb={tuple(x[1:]) for x in b.detach().cpu().tolist()}
    canvas=np.zeros((resolution,resolution,3),np.uint8)
    for x,y,z in sa-sb: canvas[resolution-1-y,x]=(255,70,70)
    for x,y,z in sb-sa: canvas[resolution-1-y,x]=(70,170,255)
    for x,y,z in sa&sb:
        if not canvas[resolution-1-y,x].any(): canvas[resolution-1-y,x]=(210,210,210)
    path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(canvas).resize((768,768),Image.Resampling.NEAREST).save(path)


def build_local_setup(out:Path, device:torch.device) -> dict[str,Any]:
    cg=json.loads((GLOBAL/"camera.json").read_text()); A=np.asarray(recover_crop_transform(out)["A_global4096_to_final1024"])
    bg=render_buffers(load_mesh(GLOBAL/"final/geometry_mesh.pt"),cg,device)
    mask=bg["mask"].numpy()>.5; coord=np.moveaxis(bg["coord"].numpy(),0,-1)
    yy,xx=np.mgrid[:1024,:1024]; uv4=np.stack(((xx+.5)*4,(yy+.5)*4,np.ones_like(xx)),axis=-1)
    uvlocal=uv4@A.T; uvlocal=uvlocal[...,:2]/uvlocal[...,2:]
    lm=np.asarray(Image.open(LOCAL/"inputs/foreground_mask_1024.png").convert("L"))>220
    ix=np.clip(np.rint(uvlocal[...,0]).astype(int),0,1023); iy=np.clip(np.rint(uvlocal[...,1]).astype(int),0,1023)
    selected=mask&(uvlocal[...,0]>=0)&(uvlocal[...,0]<1024)&(uvlocal[...,1]>=0)&(uvlocal[...,1]<1024)&lm[iy,ix]
    pts=coord[selected]; lo=np.quantile(pts,.01,axis=0); hi=np.quantile(pts,.99,axis=0)
    b=(lo+hi)/2; extent=hi-lo
    # Visible depth underestimates the rear half; the 1.35 context factor is isotropic.
    s=float(max(extent.max()*1.35,.20)); C=np.array([0.,0.,float(cg["distance"])]); z=(C-b); d=float(np.linalg.norm(z)); z/=d
    up=np.array([0.,1.,0.]); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x); R=np.stack((x,y,z))
    dlocal=d/s; focal=0.94*1024*dlocal # cube's +/-0.5 spans 94% of image at its center plane
    fov=2*math.atan(512/focal); cl={"camera_angle_x":fov,"distance":dlocal,"mesh_scale":1.0}
    J=np.diag([1.,-1.,-1.]); Rrel=J@R@J
    Kg=camera_K(cg,4096); Kl=camera_K(cl,1024); H=Kl@Rrel@np.linalg.inv(Kg); H/=H[2,2]
    source=np.asarray(Image.open(json.loads((CROP/"manifest.json").read_text())["source_image"]).convert("RGB"))
    warped=cv2.warpPerspective(source,H,(1024,1024),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    image=Image.fromarray(warped); inputs=out/"phase_bc/inputs"; inputs.mkdir(parents=True,exist_ok=True)
    image.save(inputs/"fixed_camera_1024.png"); image.resize((512,512),Image.Resampling.LANCZOS).save(inputs/"fixed_camera_512.png")
    # Dense inverse warp grid and a sparse visual check.
    gx,gy=np.meshgrid(np.arange(1024),np.arange(1024)); p=np.stack((gx,gy,np.ones_like(gx)),axis=-1).reshape(-1,3)@np.linalg.inv(H).T; p=p[:,:2]/p[:,2:]
    np.save(inputs/"local_to_global4096_warp_grid.npy",p.reshape(1024,1024,2).astype(np.float32))
    check=source.copy(); draw=ImageDraw.Draw(Image.fromarray(check))
    # Save a useful correspondence overlay on the local image instead.
    overlay=image.copy(); od=ImageDraw.Draw(overlay)
    for qy in range(64,1024,128):
        for qx in range(64,1024,128): od.ellipse((qx-4,qy-4,qx+4,qy+4),fill=(255,40,40))
    overlay.save(inputs/"warp_correspondence_check.png")
    setup={"b_global":b,"side_s":s,"R_global_to_local":R,"inverse":"X_global=b+s*R.T@q_local","physical_camera_center_global":C,
           "camera_distance_local":dlocal,"local_camera":cl,"H_global4096_to_local1024":H,"selection_visible_quantiles":{"lo":lo,"hi":hi,"extent":extent},
           "fov_definition":"focal chosen so the local canonical cube width occupies 94% at its center plane"}
    atomic_json(out/"phase_bc/local_setup.json",jsonable(setup)); return setup


def detail_metric(buffers:Mapping[str,torch.Tensor], roi:np.ndarray|None=None)->dict[str,float]:
    n=np.moveaxis(buffers["normal"].numpy(),0,-1); m=buffers["mask"].numpy()>.5
    if roi is not None:m&=roi
    gx=np.linalg.norm(n[:,1:]-n[:,:-1],axis=2); gy=np.linalg.norm(n[1:]-n[:-1],axis=2)
    mx=m[:,1:]&m[:,:-1]; my=m[1:]&m[:-1]
    vals=np.r_[gx[mx],gy[my]]
    return {"normal_gradient_mean":float(vals.mean()),"normal_gradient_p90":float(np.quantile(vals,.9)),"foreground_pixels":int(m.sum())}


def endpoint_difference(pa:Path,pb:Path)->dict[str,float]:
    a=torch.load(pa,map_location="cpu",weights_only=False); b=torch.load(pb,map_location="cpu",weights_only=False)
    ia={tuple(x.tolist()):i for i,x in enumerate(a["coords"])}; ib={tuple(x.tolist()):i for i,x in enumerate(b["coords"])}; keys=sorted(ia.keys()&ib.keys())
    xa=torch.stack([a["feats"][ia[k]].float() for k in keys]); xb=torch.stack([b["feats"][ib[k]].float() for k in keys]); d=xa-xb
    return {"overlap_count":len(keys),"relative_l2":float(d.norm()/xa.norm()),"cosine_mean":float(F.cosine_similarity(xa,xb,dim=1).mean()),"max_abs":float(d.abs().max())}


@torch.no_grad()
def save_condition_diagnostics(out:Path,device:torch.device)->None:
    from inference import init_pipeline
    setup=json.loads((out/"phase_bc/local_setup.json").read_text()); cl=setup["local_camera"]
    pipe=init_pipeline("/home/nvme04/yyyan/download/model/Pixal3D",device=str(device),low_vram=True)
    image1024=Image.open(out/"phase_bc/inputs/fixed_camera_1024.png").convert("RGB"); image512=image1024.resize((512,512),Image.Resampling.LANCZOS)
    stages=(("Shape512",pipe.image_cond_model_shape_512,image512,32,out/"phase_bc/support/B_fresh_c32.pt",out/"phase_bc/support/C_inherited_c32.pt"),
            ("Shape1024",pipe.image_cond_model_shape_1024,image1024,64,out/"phase_bc/B_fresh/c64_support.pt",out/"phase_bc/C_inherited/c64_support.pt"))
    root=out/"phase_bc/condition_diagnostic";root.mkdir(parents=True,exist_ok=True)
    for label,model,image,res,pa,pb in stages:
        ca=torch.load(pa,map_location="cpu",weights_only=False)["coords"].to(device);cb=torch.load(pb,map_location="cpu",weights_only=False)["coords"].to(device)
        A=pipe.get_proj_cond_shape(model,[image],ca,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.,grid_resolution_override=res)
        B=pipe.get_proj_cond_shape(model,[image],cb,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.,grid_resolution_override=res)
        ov=intersect_coords(ca,cb); ia={tuple(x.tolist()):i for i,x in enumerate(ca.cpu())};ib={tuple(x.tolist()):i for i,x in enumerate(cb.cpu())};keys=[tuple(x.tolist()) for x in ov.cpu()]
        fa=torch.stack([A["cond"]["proj"].feats[ia[k]].detach().cpu() for k in keys]);fb=torch.stack([B["cond"]["proj"].feats[ib[k]].detach().cpu() for k in keys])
        proj_grid=model.proj_grid; pix,depth,valid=proj_grid.project_grid_indices(camera_angle_x=torch.tensor([cl["camera_angle_x"]],device=device),distance=torch.tensor([cl["distance"]],device=device),mesh_scale=torch.tensor([1.],device=device),grid_indices=ov[:,1:],grid_resolution=res)
        torch.save({"stage":label,"overlap_coords":ov.cpu(),"global_image_token_B":A["cond"]["global"].detach().cpu(),"global_image_token_C":B["cond"]["global"].detach().cpu(),"projected_token_B":fa,"projected_token_C":fb,"projected_pixel_coordinates_B":pix[0].cpu(),"projected_pixel_coordinates_C":pix[0].cpu().clone(),"projected_depth":depth[0].cpu(),"projection_valid":valid[0].cpu(),"condition_comparison":cond_compare(A,B,ov)},root/f"{label}_overlap_condition.pt")


@torch.no_grad()
def render_fixed_multiview(meshes:Mapping[str,Any],setup:Mapping[str,Any],out:Path,device:torch.device)->None:
    import utils3d
    from pixal3d.renderers import MeshRenderer, PbrMeshRenderer
    from pixal3d.representations import MeshWithVertexPbr
    from render_pixal3d_raw_ovoxel import load_envmap
    b=torch.tensor(setup["b_global"],device=device); s=float(setup["side_s"]); R=torch.tensor(setup["R_global_to_local"],device=device)
    d=float(setup["camera_distance_local"]); fov=torch.tensor(float(setup["local_camera"]["camera_angle_x"]),device=device)
    K=utils3d.torch.intrinsics_from_fov_xy(fov,fov); up=R.T@torch.tensor([0.,1.,0.],device=device)
    root=out/"phase_bc/fixed_multiview_global_coordinates"; root.mkdir(parents=True,exist_ok=True)
    angles=(0,60,120,180,240,300); env=load_envmap("studio",device=device)
    for name,mesh_cpu in meshes.items():
        mesh=mesh_cpu.to(device); panels=[]
        for angle in angles:
            a=math.radians(angle); q=torch.tensor([math.sin(a)*d,0.,math.cos(a)*d],device=device); C=b+s*(R.T@q)
            E=utils3d.torch.extrinsics_look_at(C,b,up)
            mr=MeshRenderer({"resolution":512,"near":.01,"far":10.,"ssaa":1,"chunk_size":4_000_000,"antialias":False},device=str(device))
            z=mr.render(mesh,E,K,return_types=["normal","depth","mask"])
            normal=np.moveaxis(z.normal.detach().cpu().numpy(),0,-1); mask=z.mask.detach().cpu().numpy()>.5; depth=z.depth.detach().cpu().numpy(); good=mask&np.isfinite(depth)
            dv=np.zeros_like(depth); lo,hi=np.quantile(depth[good],[.01,.99]) if good.any() else (0,1);dv[good]=np.clip((depth[good]-lo)/(hi-lo+1e-9),0,1)
            Image.fromarray(np.uint8(normal*255)).save(root/f"{name}_yaw{angle:03d}_normal.png");Image.fromarray(np.uint8(dv*255)).save(root/f"{name}_yaw{angle:03d}_depth.png")
            # Uniform neutral material makes this a geometry/PBR-lighting diagnostic,
            # not a texture comparison.
            attrs=torch.empty((len(mesh.vertices),6),device=device);attrs[:,:3]=.62;attrs[:,3]=0.;attrs[:,4]=.72;attrs[:,5]=1.
            pm=MeshWithVertexPbr(mesh.vertices,mesh.faces,attrs,{"base_color":slice(0,3),"metallic":slice(3,4),"roughness":slice(4,5),"alpha":slice(5,6)})
            pr=PbrMeshRenderer({"resolution":512,"near":.01,"far":10.,"ssaa":1,"peel_layers":1,"face_chunk_size":4_000_000},device=str(device))
            pbr=pr.render(pm,E,K,envmap=env,use_envmap_bg=False)["shaded"].detach().cpu().permute(1,2,0).numpy();Image.fromarray(np.uint8(np.clip(pbr,0,1)*255)).save(root/f"{name}_yaw{angle:03d}_neutral_pbr.png")
            gray=.18+.72*np.clip(normal[...,2],0,1);gray[~mask]=1;panel=np.uint8(gray*255);panels.append(Image.fromarray(panel).convert("RGB"))
            del mr,z,attrs,pm,pr,pbr;torch.cuda.empty_cache()
        sheet=Image.new("RGB",(3*512,2*512),"white")
        for i,p in enumerate(panels):sheet.paste(p,((i%3)*512,(i//3)*512))
        sheet.save(root/f"{name}_gray_contact_sheet.png");del mesh;torch.cuda.empty_cache()


@torch.no_grad()
def phase_bc(args,out:Path,device:torch.device)->dict[str,Any]:
    from inference import init_pipeline
    import pixal3d_cascade512_1024_tiled2048_crop_condition as tiled
    setup=build_local_setup(out,device); b=np.asarray(setup["b_global"]); s=float(setup["side_s"]); R=np.asarray(setup["R_global_to_local"]); cl=setup["local_camera"]
    image1024=Image.open(out/"phase_bc/inputs/fixed_camera_1024.png").convert("RGB"); image512=image1024.resize((512,512),Image.Resampling.LANCZOS)
    pipe=init_pipeline("/home/nvme04/yyyan/download/model/Pixal3D",device=str(device),low_vram=True)
    pipe.sparse_structure_sampler_params["steps"]=args.steps; pipe.shape_slat_sampler_params["steps"]=args.steps
    torch.manual_seed(SEEDS["ss"]); random.seed(SEEDS["ss"]); np.random.seed(SEEDS["ss"])
    css=pipe.get_proj_cond_ss([image512],camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.)
    fresh=pipe.sample_sparse_structure(css,32); del css; torch.cuda.empty_cache()
    gc=torch.load(GLOBAL/"support/coords_c128.pt",map_location="cpu",weights_only=False)["coords"]
    g=(gc[:,1:].double()+.5)/128.-.5; q=(torch.from_numpy(R).double()@(g-torch.from_numpy(b)).T).T/s
    valid=((q>=-.5)&(q<.5)).all(1); xyz=torch.floor((q[valid]+.5)*32).int().clamp(0,31)
    inherited=torch.cat((torch.zeros(len(xyz),1,dtype=torch.int32),xyz),1).unique(dim=0).to(device)
    supp=out/"phase_bc/support"; supp.mkdir(parents=True,exist_ok=True); torch.save({"coords":fresh.cpu()},supp/"B_fresh_c32.pt");torch.save({"coords":inherited.cpu()},supp/"C_inherited_c32.pt")
    stats={"B_C32":support_stats(fresh,32,supp,"B_C32"),"C_C32":support_stats(inherited,32,supp,"C_C32")}; support_difference_image(fresh,inherited,32,supp/"C32_difference.png")
    endpoints={}; cond_diag={}; c64s={}
    for name,coords in (("B_fresh",fresh),("C_inherited",inherited)):
        cond=pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512,[image512],coords,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.)
        endpoints[name]={"cond32":cond}
    ov32=intersect_coords(fresh,inherited); cond_diag["Shape512_C32"]=cond_compare(endpoints["B_fresh"]["cond32"],endpoints["C_inherited"]["cond32"],ov32)
    if cond_diag["Shape512_C32"]["projected_max_abs"]>1e-6 or cond_diag["Shape512_C32"]["global_max_abs"]>1e-6: raise RuntimeError("B/C Shape512 condition differs")
    for name,coords in (("B_fresh",fresh),("C_inherited",inherited)):
        torch.manual_seed(SEEDS["shape512"]); st=pipe.sample_shape_slat(endpoints[name].pop("cond32"),pipe.models["shape_slat_flow_model_512"],coords)
        endpoints[name]["shape512"]=st; sparse_save(out/f"phase_bc/{name}/shape512_endpoint.pt",st)
        c64=tiled.upsample_and_quantize(pipe,st,512,64,True); c64s[name]=c64; torch.save({"coords":c64.cpu()},out/f"phase_bc/{name}/c64_support.pt")
        stats[name+"_C64"]=support_stats(c64,64,supp,name+"_C64")
        parent=torch.div(c64[:,1:],2,rounding_mode="floor"); _,counts=torch.unique(parent,dim=0,return_counts=True)
        stats[name+"_children_per_C32_parent"]={str(i):int((counts==i).sum()) for i in range(1,9)}
    support_difference_image(c64s["B_fresh"],c64s["C_inherited"],64,supp/"C64_difference.png")
    cond64={}
    for name,c64 in c64s.items(): cond64[name]=pipe.get_proj_cond_shape(pipe.image_cond_model_shape_1024,[image1024],c64,camera_angle_x=cl["camera_angle_x"],distance=cl["distance"],mesh_scale=1.,grid_resolution_override=64)
    ov64=intersect_coords(c64s["B_fresh"],c64s["C_inherited"]); cond_diag["Shape1024_C64"]=cond_compare(cond64["B_fresh"],cond64["C_inherited"],ov64)
    if cond_diag["Shape1024_C64"]["projected_max_abs"]>1e-6 or cond_diag["Shape1024_C64"]["global_max_abs"]>1e-6: raise RuntimeError("B/C Shape1024 condition differs")
    cg=json.loads((GLOBAL/"camera.json").read_text()); renders={}; meshes={}
    for name,c64 in c64s.items():
        torch.manual_seed(SEEDS["shape1024"]); st=pipe.sample_shape_slat(cond64[name],pipe.models["shape_slat_flow_model_1024"],c64); sparse_save(out/f"phase_bc/{name}/shape1024_endpoint.pt",st)
        decoded,_=pipe.decode_shape_slat(st,1024); mesh=decoded[0].cpu(); mesh.vertices=(torch.from_numpy(b)+s*(mesh.vertices.double()@torch.from_numpy(R))).float()
        path=out/f"phase_bc/{name}/native_mesh_global_similarity.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save({"mesh":mesh,"transform":setup},path)
        buffers=render_buffers(mesh,cg,device);save_buffer_images(out/"phase_bc/renders",name,buffers);renders[name]=detail_metric(buffers);meshes[name]={"vertices":len(mesh.vertices),"faces":len(mesh.faces)}
    global_buffers=render_buffers(load_mesh(GLOBAL/"final/geometry_mesh.pt"),cg,device);renders["global_C128_C256"]=detail_metric(global_buffers)
    sa={tuple(x) for x in fresh.cpu().tolist()};sc={tuple(x) for x in inherited.cpu().tolist()}; sb={tuple(x) for x in c64s["B_fresh"].cpu().tolist()};sd={tuple(x) for x in c64s["C_inherited"].cpu().tolist()}
    result={"seeds":SEEDS,"support_stats":stats,"condition_diagnostic":cond_diag,"meshes":meshes,"render_detail_metrics":renders,
            "support_overlap":{"C32_jaccard":len(sa&sc)/max(len(sa|sc),1),"C64_jaccard":len(sb&sd)/max(len(sb|sd),1)}}
    atomic_json(out/"phase_bc/metrics.json",jsonable(result));return result


def phase_a(args, out: Path, device: torch.device) -> dict[str, Any]:
    transform = recover_crop_transform(out); A = np.asarray(transform["A_global4096_to_final1024"])
    cg = json.loads((GLOBAL / "camera.json").read_text()); cl = json.loads((LOCAL / "camera.json").read_text())
    bg = render_buffers(load_mesh(GLOBAL / "final/geometry_mesh.pt"), cg, device)
    bl = render_buffers(load_mesh(LOCAL / "final/geometry_mesh.pt"), cl, device)
    save_buffer_images(out / "phase_a/renders", "global_original", bg)
    save_buffer_images(out / "phase_a/renders", "local_original", bl)
    yy, xx = torch.meshgrid(torch.arange(1024), torch.arange(1024), indexing="ij")
    pl = torch.stack((xx.flatten(), yy.flatten()), 1).float()
    invA = torch.from_numpy(np.linalg.inv(A)).float()
    pg4 = (torch.cat((pl, torch.ones(len(pl), 1)), 1) @ invA.T); pg = pg4[:, :2] / 4.
    lm = bl["mask"].flatten() > .99
    gm = sample_map(bg["mask"], pg)[:, 0] > .99
    # Erosion rejects both silhouettes; front-facing normals reject completion/back faces.
    lerode = torch.from_numpy(cv2.erode(lm.reshape(1024,1024).numpy().astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)).flatten()
    gx, gy = pg.round().long().T; inside = (gx>=3)&(gx<1021)&(gy>=3)&(gy<1021)
    valid = lm & lerode & gm & inside
    nl = bl["normal"].permute(1,2,0).reshape(-1,3)
    ng = sample_map(bg["normal"], pg)
    valid &= (nl[:,2] > .52) & (ng[:,2] > .52)
    ids = torch.where(valid)[0][::max(1, int(valid.sum())//150000)]
    xl = bl["coord"].permute(1,2,0).reshape(-1,3)[ids].numpy()
    xg = sample_map(bg["coord"], pg[ids]).numpy()
    s, R, t = robust_similarity(xl, xg)
    pred = s*(xl@R.T)+t; e3=np.linalg.norm(pred-xg,axis=1)
    qx=np.quantile(xg[:,0],[.2,.35]); neck=xg[:,0]<=qx[0]; internal=xg[:,0]>=qx[1]
    # Local normals must undergo the fitted rotation (scale/translation do not
    # affect a normal) before comparison in the common global camera frame.
    local_normals=(nl[ids].numpy()*2-1)@R.T
    global_normals=ng[ids].numpy()*2-1
    local_normals/=np.linalg.norm(local_normals,axis=1,keepdims=True)+1e-12
    global_normals/=np.linalg.norm(global_normals,axis=1,keepdims=True)+1e-12
    normal_cos=np.sum(local_normals*global_normals,1)
    normal_err=np.degrees(np.arccos(np.clip(normal_cos,-1,1)))
    dl=bl["depth"].flatten()[ids].numpy(); dg=sample_map(bg["depth"],pg[ids])[:,0].numpy()
    # Fit depth-affine induced by the 3-D similarity only for reporting in global units.
    depth_err=np.abs((cg["distance"]-pred[:,2])-(cg["distance"]-xg[:,2]))
    result={"similarity":{"scale":s,"rotation":R,"translation":t,"det_rotation":np.linalg.det(R)},
            "correspondences":{"accepted":len(ids),"filters":"both z-buffers, 7px silhouette erosion, front-facing normals"},
            "head_internal":{"3d_error":metric_stats(e3[internal]),"depth_error":metric_stats(depth_err[internal]),"normal_error_degrees":metric_stats(normal_err[internal])},
            "neck_interface":{"definition":"lowest 20% target global-x correspondences","3d_error":metric_stats(e3[neck]),"depth_error":metric_stats(depth_err[neck]),"normal_error_degrees":metric_stats(normal_err[neck])}}
    Kg=camera_K(cg,4096); Kl=camera_K(cl,1024); M=np.linalg.inv(Kg)@np.linalg.inv(A)@Kl
    sv=np.linalg.svd(M,compute_uv=False); result["projective_mapping"]={"M":M,"singular_values":sv,"sigma_ratio":sv.max()/sv.min()}
    # Render the similarity-transformed local mesh under the one global camera.
    ml=load_mesh(LOCAL / "final/geometry_mesh.pt")
    ml.vertices = (s*(ml.vertices.double()@torch.from_numpy(R).T)+torch.from_numpy(t)).float()
    aligned=render_buffers(ml,cg,device); save_buffer_images(out/"phase_a/renders","local_similarity_in_global_camera",aligned)
    mg=bg["mask"].numpy()>.5; ma=aligned["mask"].numpy()>.5
    roi=np.zeros_like(mg); x0,y0,x1,y1=json.loads((CROP/"manifest.json").read_text())["crop_xyxy_4096"]
    roi[y0//4:y1//4,x0//4:x1//4]=True
    result["silhouette_iou_head_roi"]=float((mg&ma&roi).sum()/max((mg|ma)[roi].sum(),1))
    overlay=np.zeros((1024,1024,3),np.uint8); overlay[mg&roi]=(40,180,255); overlay[ma&roi]=(255,80,80); overlay[mg&ma&roi]=(255,255,255)
    Image.fromarray(overlay).save(out/"phase_a/renders/silhouette_overlay.png")
    atomic_json(out/"phase_a/metrics.json",jsonable(result)); return result


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=("a","bc","diagnostics","renders","all"),default="all")
    p.add_argument("--cuda-device",type=int,default=4); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--steps",type=int,default=12); return p.parse_args()


def main():
    args=parse_args(); visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip()!=str(args.cuda_device): raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={args.cuda_device}, got {visible}")
    device=torch.device("cuda:0" if visible else f"cuda:{args.cuda_device}"); torch.cuda.set_device(device)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    summary={"device":str(device),"physical_cuda":args.cuda_device,"seeds":SEEDS,"steps":args.steps}
    if args.phase in ("a","all"): summary["phase_a"]=phase_a(args,out,device)
    if args.phase in ("bc","all"): summary["phase_bc"]=phase_bc(args,out,device)
    if args.phase in ("renders","all"):
        setup=json.loads((out/"phase_bc/local_setup.json").read_text()); meshes={}
        for name in ("B_fresh","C_inherited"): meshes[name]=torch.load(out/f"phase_bc/{name}/native_mesh_global_similarity.pt",map_location="cpu",weights_only=False)["mesh"]
        ma=json.loads((out/"phase_a/metrics.json").read_text())["similarity"]; ml=load_mesh(LOCAL/"final/geometry_mesh.pt");ml.vertices=(float(ma["scale"])*(ml.vertices.double()@torch.tensor(ma["rotation"],dtype=torch.double).T)+torch.tensor(ma["translation"],dtype=torch.double)).float();meshes["A_crop_moge_similarity"]=ml
        meshes["global_baseline"]=load_mesh(GLOBAL/"final/geometry_mesh.pt")
        render_fixed_multiview(meshes,setup,out,device)
        metrics=json.loads((out/"phase_bc/metrics.json").read_text());metrics["endpoint_difference"]={"Shape512":endpoint_difference(out/"phase_bc/B_fresh/shape512_endpoint.pt",out/"phase_bc/C_inherited/shape512_endpoint.pt"),"Shape1024":endpoint_difference(out/"phase_bc/B_fresh/shape1024_endpoint.pt",out/"phase_bc/C_inherited/shape1024_endpoint.pt")};atomic_json(out/"phase_bc/metrics.json",metrics);summary["renders"]="complete"
    if args.phase in ("diagnostics","all"): save_condition_diagnostics(out,device);summary["condition_diagnostics"]="complete"
    atomic_json(out/"summary.json",jsonable(summary))


if __name__ == "__main__": main()
