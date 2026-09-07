#!/usr/bin/env python3
"""Measure pairwise and third-order nonlinear interactions in a frozen SLAT decoder.

The input is a saved, already denormalized SLAT.  No encoder, flow sampler,
voxelizer, training code, or parameter update is used.  For texture SLATs the
companion shape SLAT is decoded once and its subdivision path is reused for
every texture decode, so both decoder input support and continuous output
support are exactly fixed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Sparse extensions inspect these variables during import.
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).with_name("autotune_cache.json")),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pixal3d import models
from pixal3d.modules.sparse import SparseTensor


FORMAT = "pixal3d_slat_decoder_interactions_v1"
EPS = 1e-12
DEFAULT_MODEL = Path("/home/nvme04/yyyan/download/model/Pixal3D")
PBR_LAYOUT = {
    "base_color": slice(0, 3), "metallic": slice(3, 4),
    "roughness": slice(4, 5), "alpha": slice(5, 6),
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    return device


def load_payload(path: Path, kind: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, int]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    resolution = int(raw.get("resolution", 1024)) if isinstance(raw, Mapping) else 1024
    key = "texture_slat" if kind == "tex" else "shape_slat"
    if isinstance(raw, Mapping) and key in raw:
        item = raw[key]
    elif isinstance(raw, Mapping) and "coords" in raw and ("feats" in raw or "features" in raw):
        item = raw
    else:
        raise KeyError(f"{path} contains neither {key!r} nor a coords/features payload")
    feats_key = "feats" if "feats" in item else "features"
    slat = {"coords": item["coords"].cpu(), "feats": item[feats_key].float().cpu()}
    companion = None
    if kind == "tex":
        if not isinstance(raw, Mapping) or "shape_slat" not in raw:
            raise KeyError("texture analysis requires companion shape_slat in the input file")
        shape = raw["shape_slat"]
        companion = {"coords": shape["coords"].cpu(), "feats": shape["feats"].float().cpu()}
        if not torch.equal(slat["coords"], companion["coords"]):
            raise ValueError("texture and shape SLAT coordinates differ")
    if slat["coords"].ndim != 2 or slat["coords"].shape[1] not in (3, 4):
        raise ValueError(f"expected coords [N,3/4], got {tuple(slat['coords'].shape)}")
    if slat["feats"].ndim != 2 or slat["coords"].shape[0] != slat["feats"].shape[0]:
        raise ValueError("coords/features row count mismatch")
    return slat, companion, resolution


def tensor_stats(coords: torch.Tensor, feats: torch.Tensor) -> dict[str, Any]:
    x = feats.double()
    return {
        "coords_shape": list(coords.shape), "features_shape": list(feats.shape),
        "features_mean": float(x.mean()), "features_std": float(x.std(unbiased=False)),
        "features_min": float(x.min()), "features_max": float(x.max()),
        "features_frobenius_norm": float(torch.linalg.vector_norm(x)),
        "active_points": int(coords.shape[0]), "dtype_on_disk": str(feats.dtype),
    }


def deterministic_kmeans(xyz: np.ndarray, k: int, seed: int, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ with stable empty-cluster recovery."""
    if k < 1 or k > len(xyz):
        raise ValueError(f"num_groups must be in [1,{len(xyz)}], got {k}")
    x = xyz.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))].copy()]
    closest = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = closest.sum()
        idx = int(np.argmax(closest)) if total <= 0 else int(rng.choice(len(x), p=closest / total))
        centers.append(x[idx].copy())
        closest = np.minimum(closest, ((x - centers[-1]) ** 2).sum(1))
    centers_arr = np.stack(centers)
    labels = np.full(len(x), -1, np.int64)
    for _ in range(iterations):
        dist = ((x[:, None, :] - centers_arr[None, :, :]) ** 2).sum(2)
        new_labels = dist.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(k):
            chosen = x[labels == cluster]
            if len(chosen):
                centers_arr[cluster] = chosen.mean(0)
            else:
                centers_arr[cluster] = x[int(np.argmax(dist.min(1)))]
    return labels, centers_arr


def build_groups(mode: str, coords: torch.Tensor, feats: torch.Tensor, k: int, seed: int, out: Path) -> list[torch.Tensor]:
    groups: list[torch.Tensor] = []
    xyz = coords[:, -3:].float().numpy()
    if mode == "channel":
        if k > feats.shape[1]:
            raise ValueError(f"cannot split {feats.shape[1]} channels into {k} nonempty groups")
        bounds = np.linspace(0, feats.shape[1], k + 1, dtype=np.int64)
        for i in range(k):
            mask = torch.zeros_like(feats, dtype=torch.bool)
            mask[:, bounds[i]:bounds[i + 1]] = True
            groups.append(mask)
        atomic_json(out / "groups.json", [{"group": i, "channels": [int(bounds[i]), int(bounds[i + 1])]} for i in range(k)])
        return groups
    labels, centers = deterministic_kmeans(xyz, k, seed)
    records = []
    for i in range(k):
        rows = torch.from_numpy(labels == i)
        mask = rows[:, None].expand_as(feats).clone()
        groups.append(mask)
        p = xyz[labels == i]
        records.append({
            "group": i, "point_count": int(len(p)), "center": centers[i].tolist(),
            "bbox_min": p.min(0).tolist(), "bbox_max": p.max(0).tolist(),
            "feature_norm": float(torch.linalg.vector_norm(feats[rows].double())),
        })
    atomic_json(out / "clusters.json", records)
    atomic_torch_save(out / "cluster_labels.pt", {"labels": torch.from_numpy(labels), "centers": torch.from_numpy(centers)})
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    max_points = min(40000, len(xyz)); ids = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
    ax.scatter(xyz[ids, 0], xyz[ids, 1], xyz[ids, 2], c=labels[ids], cmap="tab10", s=2, alpha=.75)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=np.arange(k), cmap="tab10", s=90, marker="x")
    ax.set_title(f"Deterministic coordinate k-means (K={k})"); fig.tight_layout()
    fig.savefig(out / "spatial_groups_3d.png", dpi=180); plt.close(fig)
    return groups


def norm64(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.double()))


def save_cont(path: Path, coords: torch.Tensor, feats: torch.Tensor, label: str) -> None:
    # Inclusion-exclusion can be much smaller than an individual decoder
    # response; keep float32 on disk so serialization does not create a false
    # interaction residual through fp16 rounding.
    atomic_torch_save(path, {"format": FORMAT, "label": label, "coords": coords.cpu(), "features": feats.float().cpu()})


def load_cont(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["features"].float()


def save_image(path: Path, array: np.ndarray, signed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if signed:
        scale = float(np.percentile(np.abs(a), 99.5)) + EPS
        a = np.clip(a / (2 * scale) + .5, 0, 1)
    else:
        a = np.clip(a, 0, 1)
    Image.fromarray((a * 255 + .5).astype(np.uint8)).save(path)


def heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(matrix, cmap="magma")
    ax.set_xlabel("group j"); ax.set_ylabel("group i"); ax.set_title(title)
    ax.set_xticks(range(len(matrix))); ax.set_yticks(range(len(matrix)))
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def scatter_magnitude(coords: torch.Tensor, mag: torch.Tensor, path: Path, title: str) -> None:
    xyz = coords[:, -3:].float().numpy(); values = mag.float().numpy()
    n = min(60000, len(values))
    # Include the strongest samples and a deterministic uniform background.
    strong = np.argsort(values)[-min(n // 2, len(values)):]
    uniform = np.linspace(0, len(values) - 1, min(n - len(strong), len(values)), dtype=np.int64)
    ids = np.unique(np.concatenate([strong, uniform]))
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(xyz[ids, 0], xyz[ids, 1], xyz[ids, 2], c=values[ids], cmap="inferno", s=2)
    ax.set_title(title); fig.colorbar(p, ax=ax, shrink=.7, label="interaction L2")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def fft_split(image: np.ndarray, cutoff: float) -> tuple[float, float, float]:
    # Orthogonal radial masks make low+high energy equal total energy (Parseval up to scale).
    x = image.astype(np.float64)
    h, w = x.shape[:2]
    fy = np.fft.fftfreq(h)[:, None] / .5; fx = np.fft.fftfreq(w)[None, :] / .5
    low_mask = (np.sqrt(fx * fx + fy * fy) <= cutoff)[..., None]
    spectrum = np.fft.fft2(x, axes=(0, 1), norm="ortho")
    low = float(np.sum(np.abs(spectrum * low_mask) ** 2))
    high = float(np.sum(np.abs(spectrum * (~low_mask)) ** 2))
    return low, high, high / (low + high + EPS)


class ExperimentDecoder:
    def __init__(self, model_root: Path, kind: str, device: torch.device, resolution: int,
                 slat: Mapping[str, torch.Tensor], companion: Mapping[str, torch.Tensor] | None):
        self.kind, self.device, self.resolution = kind, device, resolution
        ckpts = model_root / "ckpts"
        self.decoder = models.from_pretrained(str(ckpts / ("tex_dec_next_dc_f16c32_fp16" if kind == "tex" else "shape_dec_next_dc_f16c32_fp16"))).to(device).eval()
        self.coords = slat["coords"].to(device)
        self.guide_subs = None; self.mesh = None
        if kind == "tex":
            assert companion is not None
            shape_decoder = models.from_pretrained(str(ckpts / "shape_dec_next_dc_f16c32_fp16")).to(device).eval()
            shape_decoder.set_resolution(resolution)
            shape = SparseTensor(coords=companion["coords"].to(device), feats=companion["feats"].to(device))
            with torch.inference_mode():
                meshes, self.guide_subs = shape_decoder(shape, return_subs=True)
            self.mesh = meshes[0]
            del shape_decoder, shape
        else:
            self.decoder.set_resolution(resolution)

    @torch.inference_mode()
    def decode(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = SparseTensor(coords=self.coords, feats=feats.to(self.device))
        if self.kind == "tex":
            out = self.decoder(value, guide_subs=self.guide_subs) * .5 + .5
            return out.coords.detach().cpu(), out.feats.detach().float().cpu()
        # The public shape decoder performs topology extraction.  Invoke its
        # inherited continuous forward to retain the final 7-channel tensor.
        from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
        out, _subs = SparseUnetVaeDecoder.forward(self.decoder, value, return_subs=True)
        return out.coords.detach().cpu(), out.feats.detach().float().cpu()


class FixedRenderer:
    def __init__(self, decoder: ExperimentDecoder, camera_path: Path | None, resolution: int,
                 yaw: int, chunk_size: int):
        self.enabled = decoder.kind == "tex" and decoder.mesh is not None and resolution > 0
        if not self.enabled:
            return
        from pixal3d.renderers import MeshRenderer
        from pixal3d.utils import render_utils
        camera = {"camera_angle_x": .517371749106554, "distance": 1.889538288116455}
        if camera_path and camera_path.is_file():
            camera.update(json.loads(camera_path.read_text()))
        extr, intr = render_utils.proj_camera_to_render_params(camera_angle_x=float(camera["camera_angle_x"]), distance=float(camera["distance"]))
        angle = math.radians(yaw); c, s = math.cos(angle), math.sin(angle)
        rot = torch.tensor([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=extr.dtype, device=extr.device)
        rot[:3, :3] = rot[:3, :3].T
        self.extr, self.intr = (extr @ rot).to(decoder.device), intr.to(decoder.device)
        self.renderer = MeshRenderer({"resolution": resolution, "near": max(.01, float(camera["distance"])-2),
            "far": float(camera["distance"])+10, "ssaa": 1, "chunk_size": chunk_size,
            "antialias": False}, device=str(decoder.device))
        self.decoder = decoder

    def render(self, coords: torch.Tensor, attrs: torch.Tensor) -> np.ndarray | None:
        if not self.enabled:
            return None
        from pixal3d.representations import MeshWithVoxel
        live = MeshWithVoxel(self.decoder.mesh.vertices, self.decoder.mesh.faces,
            origin=[-.5,-.5,-.5], voxel_size=1/self.decoder.resolution,
            coords=coords[:, -3:].to(self.decoder.device), attrs=attrs.to(self.decoder.device),
            voxel_shape=torch.Size([1, attrs.shape[1], *([self.decoder.resolution] * 3)]), layout=PBR_LAYOUT)
        result = self.renderer.render(live, self.extr, self.intr, return_types=["attr", "mask"])
        rgb = result["base_color"].permute(1,2,0).detach().float().cpu().numpy()
        mask = result["mask"].detach().float().cpu().numpy()[..., None]
        return np.clip(rgb * mask, 0, 1)


def perturb(base: torch.Tensor, masks: Sequence[torch.Tensor], ids: Iterable[int], lam: float) -> torch.Tensor:
    out = base.clone()
    for i in ids:
        out[masks[i]] *= 1.0 - lam
    return out


def summarize(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
            "P75": float(np.percentile(x,75)), "P90": float(np.percentile(x,90)), "max": float(x.max())}


def run_mode(args: argparse.Namespace, mode: str, decoder: ExperimentDecoder, renderer: FixedRenderer,
             input_coords: torch.Tensor, continuous_coords: torch.Tensor, base_feats: torch.Tensor,
             baseline: torch.Tensor, baseline_render: np.ndarray | None,
             out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    masks = build_groups(mode, input_coords, base_feats, args.num_groups, args.seed, out)
    delta_norm = [norm64(args.attenuation * base_feats[m]) for m in masks]
    k = len(masks); single_effect: list[float] = []
    single_paths: list[Path] = []; single_renders: list[np.ndarray | None] = []
    for i in range(k):
        print(f"[{mode}] single {i+1}/{k}", flush=True)
        c, y = decoder.decode(perturb(base_feats, masks, [i], args.attenuation))
        if not torch.equal(c, continuous_coords): raise RuntimeError(f"single {i} changed continuous output support")
        path = out / "singles" / f"group_{i:02d}" / "continuous.pt"; save_cont(path, c, y, f"single_{i}")
        if decoder.kind == "tex":
            atomic_json(path.parent / "mesh_reference.json", {
                "geometry": str((out.parent / "baseline" / "mesh.pt").resolve()),
                "voxel_attributes": str(path.resolve()),
                "note": "Texture experiment geometry is intentionally fixed; these two files constitute this output MeshWithVoxel.",
            })
        single_paths.append(path); effect = norm64(y - baseline); single_effect.append(effect)
        r = renderer.render(c, y); single_renders.append(r)
        if r is not None: save_image(path.parent / "render.png", r)
        atomic_json(path.parent / "metrics.json", {"group":i,"delta_norm":delta_norm[i],"single_effect":effect,
            "normalized_sensitivity":effect/(delta_norm[i]+EPS)})

    raw = np.zeros((k,k)); joint_m = np.zeros((k,k)); single_m = np.zeros((k,k))
    rows: list[dict[str, Any]] = []
    for pair_index, (i,j) in enumerate(itertools.combinations(range(k),2), 1):
        print(f"[{mode}] pair {pair_index}/{k*(k-1)//2}: ({i},{j})", flush=True)
        c, yij = decoder.decode(perturb(base_feats, masks, [i,j], args.attenuation))
        if not torch.equal(c, continuous_coords): raise RuntimeError(f"pair {(i,j)} changed continuous output support")
        yi, yj = load_cont(single_paths[i]), load_cont(single_paths[j])
        residual = yij - yi - yj + baseline
        e_int = norm64(residual); e_joint = norm64(yij-baseline)
        r_joint = e_int/(e_joint+EPS); r_single=e_int/(single_effect[i]+single_effect[j]+EPS)
        raw[i,j]=raw[j,i]=e_int; joint_m[i,j]=joint_m[j,i]=r_joint; single_m[i,j]=single_m[j,i]=r_single
        pair_dir=out/"pairs"/f"group_{i:02d}_{j:02d}"; save_cont(pair_dir/"continuous.pt",c,yij,f"pair_{i}_{j}")
        if decoder.kind == "tex":
            atomic_json(pair_dir / "mesh_reference.json", {
                "geometry": str((out.parent / "baseline" / "mesh.pt").resolve()),
                "voxel_attributes": str((pair_dir / "continuous.pt").resolve()),
                "note": "Texture experiment geometry is intentionally fixed; these two files constitute this output MeshWithVoxel.",
            })
        rij=renderer.render(c,yij); render_raw=render_joint=render_low=render_high=render_q=None
        if rij is not None and baseline_render is not None and single_renders[i] is not None and single_renders[j] is not None:
            img_res=rij-single_renders[i]-single_renders[j]+baseline_render
            render_raw=float(np.linalg.norm(img_res.astype(np.float64)))
            render_low,render_high,render_q=fft_split(img_res,args.fft_cutoff)
            save_image(pair_dir/"render.png",rij); save_image(pair_dir/"interaction_render_signed.png",img_res,signed=True)
            abs_img=np.abs(img_res); save_image(pair_dir/"interaction_render_abs.png",abs_img/(np.percentile(abs_img,99.5)+EPS))
            np.save(pair_dir/"interaction_render.npy",img_res.astype(np.float32))
        row={"group_i":i,"group_j":j,"raw_interaction":e_int,"R_joint":r_joint,"R_single":r_single,
             "single_effect_i":single_effect[i],"single_effect_j":single_effect[j],"joint_effect":e_joint,
             "delta_norm_i":delta_norm[i],"delta_norm_j":delta_norm[j],"render_interaction":render_raw,
             "render_fft_low_energy":render_low,"render_fft_high_energy":render_high,"render_high_fraction_Q":render_q,
             "pair_path":str(pair_dir)}
        rows.append(row); atomic_json(pair_dir/"metrics.json",row)
        del yi,yj,residual,yij; gc.collect()

    rows.sort(key=lambda r:r["R_joint"],reverse=True)
    for rank,row in enumerate(rows,1): row["rank"]=rank
    fields=["rank","group_i","group_j","raw_interaction","R_joint","R_single","single_effect_i","single_effect_j",
            "joint_effect","delta_norm_i","delta_norm_j","render_interaction","render_fft_low_energy",
            "render_fft_high_energy","render_high_fraction_Q","pair_path"]
    with (out/"interaction_matrix.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.save(out/"interaction_matrix.npy",joint_m)
    np.savez(out/"interaction_matrices.npz",raw=raw,joint=joint_m,single=single_m)
    heatmap(joint_m,out/"interaction_joint_heatmap.png","Joint-normalized interaction")
    heatmap(single_m,out/"interaction_single_heatmap.png","Single-effect-normalized interaction")
    heatmap(raw,out/"interaction_raw_heatmap.png","Raw interaction norm")

    # Spatial diagnostics for the ten strongest pair residuals.
    for row in rows[:10]:
        i,j=int(row["group_i"]),int(row["group_j"])
        yij=load_cont(Path(row["pair_path"])/"continuous.pt"); yi=load_cont(single_paths[i]); yj=load_cont(single_paths[j])
        residual=yij-yi-yj+baseline; mag=torch.linalg.vector_norm(residual.float(),dim=1)
        vis=out/"top_pair_visualizations"/f"rank_{row['rank']:02d}_group_{i:02d}_{j:02d}"
        atomic_torch_save(vis/"per_voxel_interaction.pt",{"coords":continuous_coords,"magnitude":mag.half(),"residual":residual.half()})
        scatter_magnitude(continuous_coords,mag,vis/"interaction_3d.png",f"rank {row['rank']}: groups {i},{j}")

    # Lambda sweep: decode each needed single once per lambda, then the top five pairs.
    top5=[(int(r["group_i"]),int(r["group_j"])) for r in rows[:5]]; needed=sorted(set(itertools.chain.from_iterable(top5)))
    sweep_rows=[]
    for lam in args.lambda_sweep:
        sy={}
        for i in needed: sy[i]=decoder.decode(perturb(base_feats,masks,[i],lam))[1]
        for i,j in top5:
            yij=decoder.decode(perturb(base_feats,masks,[i,j],lam))[1]
            ei=norm64(sy[i]-baseline); ej=norm64(sy[j]-baseline); resid=norm64(yij-sy[i]-sy[j]+baseline); joint=norm64(yij-baseline)
            sweep_rows.append({"lambda":lam,"group_i":i,"group_j":j,"raw_interaction":resid,
                "R_joint":resid/(joint+EPS),"R_single":resid/(ei+ej+EPS)})
    with (out/"lambda_sweep.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(sweep_rows[0])); w.writeheader(); w.writerows(sweep_rows)
    fig,ax=plt.subplots(figsize=(7,5))
    for i,j in top5:
        values=[r for r in sweep_rows if r["group_i"]==i and r["group_j"]==j]
        ax.plot([r["lambda"] for r in values],[r["R_joint"] for r in values],marker="o",label=f"({i},{j})")
    ax.set_xlabel("attenuation lambda"); ax.set_ylabel("R_joint"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out/"interaction_vs_lambda.png",dpi=180); plt.close(fig)

    # Four strongest participating groups, all four triplets.
    score=np.zeros(k)
    for r in rows:
        score[int(r["group_i"])]+=r["R_joint"]; score[int(r["group_j"])]+=r["R_joint"]
    top_groups=np.argsort(score)[::-1][:min(4,k)].tolist(); triples=[]
    pair_lookup={(min(int(r["group_i"]),int(r["group_j"])),max(int(r["group_i"]),int(r["group_j"]))):Path(r["pair_path"])/"continuous.pt" for r in rows}
    for i,j,l in itertools.combinations(sorted(top_groups),3):
        yijk=decoder.decode(perturb(base_feats,masks,[i,j,l],args.attenuation))[1]
        yij=load_cont(pair_lookup[(i,j)]); yil=load_cont(pair_lookup[(i,l)]); yjl=load_cont(pair_lookup[(j,l)])
        yi,yj,yl=load_cont(single_paths[i]),load_cont(single_paths[j]),load_cont(single_paths[l])
        r3=yijk-yij-yil-yjl+yi+yj+yl-baseline; raw3=norm64(r3); joint3=norm64(yijk-baseline)
        triples.append({"group_i":i,"group_j":j,"group_k":l,"raw_third_order":raw3,"R_third":raw3/(joint3+EPS),"joint_effect":joint3})
    atomic_json(out/"third_order.json",{"selected_groups":top_groups,"triplets":triples})

    summary={"group_mode":mode,"num_groups":k,"attenuation":args.attenuation,"pair_count":len(rows),
             "R_joint":summarize([r["R_joint"] for r in rows]),"R_single":summarize([r["R_single"] for r in rows]),
             "raw_interaction":summarize([r["raw_interaction"] for r in rows]),"top_10_pairs":rows[:10],
             "single_effects":single_effect,"delta_norms":delta_norm,"third_order":triples}
    atomic_json(out/"summary.json",summary); return summary


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--slat-kind",choices=("shape","tex"),default="tex")
    p.add_argument("--group-mode",choices=("channel","spatial","both"),default="both")
    p.add_argument("--num-groups",type=int,default=8); p.add_argument("--lambda",dest="attenuation",type=float,default=.25)
    p.add_argument("--lambda-sweep",type=float,nargs="+",default=[.05,.10,.25,.50])
    p.add_argument("--device",default="cuda:4"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-path",type=Path,default=DEFAULT_MODEL); p.add_argument("--camera",type=Path)
    p.add_argument("--render-resolution",type=int,default=256,help="0 disables render analysis (shape mode is always continuous-only)")
    p.add_argument("--render-yaw",type=int,default=0); p.add_argument("--render-chunk-size",type=int,default=1_000_000)
    p.add_argument("--fft-cutoff",type=float,default=.2)
    return p.parse_args()


def main() -> None:
    args=parse_args()
    if not 0 < args.attenuation < 1: raise ValueError("--lambda must lie strictly between 0 and 1")
    if not 0 < args.fft_cutoff < 1: raise ValueError("--fft-cutoff must lie strictly between 0 and 1")
    seed_everything(args.seed); device=resolve_device(args.device); args.output.mkdir(parents=True,exist_ok=True)
    slat,companion,resolution=load_payload(args.input,args.slat_kind); coords,base_feats=slat["coords"],slat["feats"]
    config={"format":FORMAT,"input":str(args.input.resolve()),"output":str(args.output.resolve()),"slat_kind":args.slat_kind,
            "group_mode":args.group_mode,"num_groups":args.num_groups,"lambda":args.attenuation,"lambda_sweep":args.lambda_sweep,
            "seed":args.seed,"device_requested":args.device,"device_resolved":str(device),"physical_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),
            "decoder_resolution":resolution,"model_path":str(args.model_path),"render_resolution":args.render_resolution,
            "render_yaw":args.render_yaw,"fft_cutoff":args.fft_cutoff,"normalization_boundary":"saved feats are already inverse-normalized decoder inputs"}
    atomic_json(args.output/"config.json",config); atomic_json(args.output/"baseline"/"slat_stats.json",tensor_stats(coords,base_feats))
    print(f"[load] kind={args.slat_kind} N={len(coords):,} C={base_feats.shape[1]} device={device}",flush=True)
    decoder=ExperimentDecoder(args.model_path,args.slat_kind,device,resolution,slat,companion)
    c0,y0=decoder.decode(base_feats); c1,y1=decoder.decode(base_feats.clone())
    if not torch.equal(c0,c1): raise RuntimeError("baseline reproduction changed output coordinates")
    max_abs=float((y1-y0).abs().max()); rel=norm64(y1-y0)/(norm64(y0)+EPS)
    reproduction={"baseline_reproduction_error_max_abs":max_abs,"baseline_reproduction_error_relative_l2":rel,
                  "coords_exact":True,"input_coords_exact":bool(torch.equal(decoder.coords.cpu(),coords)),
                  "continuous_coords_shape":list(c0.shape),"continuous_features_shape":list(y0.shape)}
    atomic_json(args.output/"baseline"/"reproduction.json",reproduction)
    if max_abs > 5e-5: raise RuntimeError(f"baseline reproduction failed: max_abs={max_abs}")
    save_cont(args.output/"baseline"/"continuous.pt",c0,y0,"baseline")
    if decoder.mesh is not None:
        atomic_torch_save(args.output/"baseline"/"mesh.pt",{"vertices":decoder.mesh.vertices.cpu(),"faces":decoder.mesh.faces.cpu(),"resolution":resolution})
    camera=args.camera
    if camera is None:
        candidate=args.input.parent.parent/"global_camera.json"
        camera=candidate if candidate.is_file() else None
    renderer=FixedRenderer(decoder,camera,args.render_resolution,args.render_yaw,args.render_chunk_size)
    r0=renderer.render(c0,y0)
    if r0 is not None: save_image(args.output/"baseline"/"renders"/"fixed_view.png",r0); np.save(args.output/"baseline"/"renders"/"fixed_view.npy",r0)
    modes=["channel","spatial"] if args.group_mode=="both" else [args.group_mode]
    summaries={}
    for mode in modes:
        summaries[mode]=run_mode(args,mode,decoder,renderer,coords,c0,base_feats,y0,r0,args.output/f"{mode}{args.num_groups}")
    final={"format":FORMAT,"baseline_reproduction":reproduction,"experiments":summaries,
           "interpretation_guardrail":"Grouping is only a probe basis; spatial clusters are not SLAT frequency bands."}
    atomic_json(args.output/"summary.json",final)
    print(f"[done] {args.output.resolve()}",flush=True)


if __name__ == "__main__":
    main()
