#!/usr/bin/env python3
"""Validate Euclidean sparse Fourier projection on synthetic and Pixal3D SLATs.

The implementation deliberately uses a real constant/cosine/sine basis.  This is
the real-valued form of exp(j 2 pi k.p), keeps conjugate pairs together, and makes
the projected SLAT real without discarding an imaginary residual.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


EPS = 1e-12
RATIOS = (1.0, 0.75, 0.5, 0.25, 0.1)


def canonical_frequencies(max_radius: int) -> torch.Tensor:
    """One representative of every non-zero +/- integer frequency pair."""
    rows = []
    for x in range(-max_radius, max_radius + 1):
        for y in range(-max_radius, max_radius + 1):
            for z in range(-max_radius, max_radius + 1):
                r2 = x*x + y*y + z*z
                if not (0 < r2 <= max_radius*max_radius):
                    continue
                # First non-zero coordinate is positive: exactly one of k and -k.
                first = x if x else (y if y else z)
                if first > 0:
                    rows.append((r2, x, y, z))
    rows.sort(key=lambda q: (q[0], q[1], q[2], q[3]))
    return torch.tensor([q[1:] for q in rows], dtype=torch.float64)


def frequencies_for_cutoff(pool: torch.Tensor, cutoff: float) -> torch.Tensor:
    return pool[torch.linalg.vector_norm(pool, dim=1) <= cutoff + 1e-10]


def basis(points: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Real orthogonal Fourier basis [1, sqrt(2)cos, sqrt(2)sin]."""
    points = points.to(dtype=torch.float64)
    frequencies = frequencies.to(device=points.device, dtype=points.dtype)
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    if frequencies.numel() == 0:
        return ones
    phase = 2.0 * math.pi * (points @ frequencies.T)
    return torch.cat((ones, math.sqrt(2.0)*torch.cos(phase), math.sqrt(2.0)*torch.sin(phase)), dim=1)


@dataclass
class Projection:
    low: torch.Tensor
    high: torch.Tensor
    coefficients: torch.Tensor
    condition_number: float
    rank: int
    columns: int


def project(points: torch.Tensor, features: torch.Tensor, frequencies: torch.Tensor) -> Projection:
    """Orthogonal least-squares projection; never forms/inverts A^T A."""
    original_dtype = features.dtype
    a = basis(points, frequencies)
    target = features.to(device=a.device, dtype=a.dtype)
    # CUDA exposes the QR-based GELs driver.  CPU defaults to rank-revealing GELSy.
    solution = torch.linalg.lstsq(a, target).solution
    low64 = a @ solution
    high64 = target - low64
    singular = torch.linalg.svdvals(a)
    tol = torch.finfo(a.dtype).eps * max(a.shape) * singular[0]
    rank = int((singular > tol).sum().item())
    cond = float((singular[0] / singular[-1]).item()) if singular[-1] > 0 else float("inf")
    return Projection(low64.to(original_dtype), high64.to(original_dtype), solution,
                      cond, rank, int(a.shape[1]))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((torch.linalg.vector_norm(a-b) / (torch.linalg.vector_norm(b)+EPS)).item())


def energy_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.square().sum() / (b.square().sum()+EPS)).item())


def fft_low(volume: torch.Tensor, cutoff: float) -> torch.Tensor:
    n = volume.shape[0]
    freq = torch.fft.fftfreq(n, d=1.0/n, device=volume.device)
    xx, yy, zz = torch.meshgrid(freq, freq, freq, indexing="ij")
    mask = xx.square()+yy.square()+zz.square() <= cutoff*cutoff + 1e-10
    extra = (None,)*(volume.ndim-3)
    return torch.fft.ifftn(torch.fft.fftn(volume, dim=(0,1,2))*mask[(...,)+extra],
                           dim=(0,1,2)).real


def grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(n, device=device, dtype=torch.float64)/n - 0.5
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x,y,z), -1).reshape(-1,3), torch.stack((x,y,z), -1)


def signal(points: torch.Tensor) -> torch.Tensor:
    def wave(k: Sequence[float]) -> torch.Tensor:
        kk = torch.tensor(k, device=points.device, dtype=points.dtype)
        return torch.sin(2*math.pi*(points@kk))
    return wave((1,1,0)) + 0.3*wave((3,0,1)) + 0.1*wave((7,2,0))


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def line_plot(path: Path, x: Sequence[float], series: dict[str, Sequence[float]],
              xlabel: str, ylabel: str, logy: bool=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4,4.2))
    for name, values in series.items():
        plt.plot(x, values, marker="o", label=name)
    if logy: plt.yscale("log")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True, alpha=.3)
    if len(series)>1: plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def support_indices(kind: str, points: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    n = len(points); gen = torch.Generator(device=points.device).manual_seed(seed)
    count = max(1, int(round(n*ratio)))
    if kind == "uniform_random":
        return torch.randperm(n, generator=gen, device=points.device)[:count]
    if kind == "surface":
        radius = torch.linalg.vector_norm(points, dim=1)
        # Select the thinnest shell around r=.36; deterministic and ratio controlled.
        return torch.argsort(torch.abs(radius-.36))[:count]
    if kind == "clustered":
        centers = torch.tensor([[-.27,-.22,-.18],[.24,.22,.19],[-.18,.25,.22]],
                               device=points.device, dtype=points.dtype)
        distance = torch.cdist(points, centers).amin(dim=1)
        return torch.argsort(distance)[:count]
    raise ValueError(kind)


def zero_fill_baseline(full: torch.Tensor, active: torch.Tensor, shape: tuple[int,int,int],
                       cutoff: float) -> torch.Tensor:
    z = torch.zeros((math.prod(shape),)+full.shape[1:], device=full.device, dtype=full.dtype)
    z[active] = full[active]
    return fft_low(z.reshape(shape+full.shape[1:]), cutoff).reshape_as(z)[active]


def raw_nudft(points: torch.Tensor, feats: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    a = basis(points, frequencies)
    # Dense-grid orthogonality assumption, intentionally no Gram correction.
    coeff = (a.T @ feats.to(a.dtype))/points.shape[0]
    return (a@coeff).to(feats.dtype)


def gaussian_local(points: torch.Tensor, feats: torch.Tensor, sigma: float=.09,
                   chunk: int=2048) -> torch.Tensor:
    out=[]
    p=points.to(torch.float32); f=feats.to(torch.float32)
    for start in range(0,len(p),chunk):
        d2=torch.cdist(p[start:start+chunk],p).square()
        w=torch.exp(-d2/(2*sigma*sigma)); w=w/(w.sum(1,keepdim=True)+EPS)
        out.append(w@f)
    return torch.cat(out).to(feats.dtype)


def numerical_checks(points: torch.Tensor, feats: torch.Tensor, frequencies: torch.Tensor,
                     seed: int) -> dict[str,float]:
    p=project(points,feats,frequencies)
    twice=project(points,p.low,frequencies).low
    inner=(p.low*p.high).sum().abs()
    orth=float((inner/(torch.linalg.vector_norm(p.low)*torch.linalg.vector_norm(p.high)+EPS)).item())
    gen=torch.Generator(device=feats.device).manual_seed(seed)
    q=torch.linalg.qr(torch.randn((feats.shape[1],feats.shape[1]),generator=gen,
                                  device=feats.device,dtype=feats.dtype)).Q
    rotated=project(points,feats@q,frequencies).low
    translation=torch.tensor([.137,-.083,.211],device=points.device,dtype=points.dtype)
    translated=project(points+translation,feats,frequencies).low
    # A generic rotation exposes the finite Cartesian lattice approximation.
    axis=torch.tensor([.3,.5,.8],device=points.device,dtype=points.dtype); axis/=torch.linalg.vector_norm(axis)
    angle=.73; k=torch.tensor([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]],device=points.device,dtype=points.dtype)
    rot=torch.eye(3,device=points.device,dtype=points.dtype)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
    rotated_xyz=project(points@rot.T,feats,frequencies).low
    return {"recomposition_error":rel(p.low+p.high,feats),"orthogonality":orth,
            "idempotence":rel(twice,p.low),"channel_rotation":rel(rotated,p.low@q),
            "translation":rel(translated,p.low),"coordinate_rotation":rel(rotated_xyz,p.low),
            "condition_number":p.condition_number,"rank":p.rank,"columns":p.columns}


def load_slat(path: Path, device: torch.device) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if "shape_slat" in obj:
        shape=obj["shape_slat"]; tex=obj.get("texture_slat",shape)
        coords=shape["coords"]; sf=shape["feats"]; tf=tex["feats"]
    elif "coords" in obj and ("feats" in obj or "features" in obj):
        coords=obj["coords"]; sf=obj.get("feats",obj.get("features")); tf=sf
    else: raise ValueError(f"Unsupported SLAT payload keys: {list(obj)}")
    xyz=coords[:,-3:].to(torch.float64)
    extent=max(float(xyz.max().item()+1),64.0)
    xyz=(xyz+.5)/extent-.5
    return xyz.to(device),sf.to(device),tf.to(device)


def shell_spectrum(points: torch.Tensor, feats: torch.Tensor, pool: torch.Tensor,
                   max_radius: int) -> list[dict[str,float]]:
    rows=[]
    previous=torch.zeros_like(feats)
    for radius in range(0,max_radius+1):
        freq=frequencies_for_cutoff(pool,float(radius))
        cur=project(points,feats,freq)
        band=cur.low-previous
        modes=max(1,cur.columns-(rows[-1]["columns"] if rows else 0))
        total=float(band.square().sum().item())
        rows.append({"radius":radius,"columns":cur.columns,"shell_sum":total,
                     "shell_mean":total/modes,"condition_number":cur.condition_number})
        previous=cur.low
    return rows


def write_report(path: Path, summary: dict[str,Any]) -> None:
    dense=summary["dense"]; sparse=summary["sparse_random"]; checks=summary["checks"]
    real=summary.get("real",{})
    verdict="Supported"
    reasons=[]
    if dense["fft_relative_error"]>1e-6: verdict="Rejected"; reasons.append("dense FFT consistency failed")
    if checks["orthogonality"]>1e-7 or checks["idempotence"]>1e-7:
        verdict="Rejected"; reasons.append("projection properties failed")
    if any(r["sfp_error"]>=r["zero_fill_error"] for r in sparse[1:]):
        verdict="Partially supported" if verdict!="Rejected" else verdict
        reasons.append("SFP did not beat zero-fill at every sparse ratio")
    if real and real.get("known_sinusoid_error",1)>0.05:
        verdict="Partially supported" if verdict!="Rejected" else verdict
        reasons.append("known sinusoid on real support was not recovered accurately")
    if real and real.get("coordinate_rotation",0)>0.05:
        verdict="Partially supported" if verdict!="Rejected" else verdict
        reasons.append("finite Cartesian frequency set was not rotation invariant")
    if real and real.get("condition_last",0)>1e5:
        verdict="Partially supported" if verdict!="Rejected" else verdict
        reasons.append("real-support design matrix became severely ill-conditioned")
    decoder=summary.get("decoder",{"status":"not_run"})
    baselines=summary.get("real_baselines",{})
    if decoder.get("status")!="complete":
        if verdict=="Supported": verdict="Partially supported"
        reasons.append("decoder semantic progression was not established")
    elif not decoder.get("coarse_to_fine",False):
        if verdict=="Supported": verdict="Partially supported"
        reasons.append("real decoder did not show stable coarse-to-fine progression")
    lines=["# Sparse Fourier Projection validation report","",f"## Verdict: {verdict}",""]
    if reasons: lines += ["Reasons: " + "; ".join(reasons) + ".",""]
    lines += [
      "## Answers required by Codex.md","",
      f"1. Dense FFT reproduced: **{'yes' if dense['fft_relative_error']<1e-6 else 'no'}**, relative error `{dense['fft_relative_error']:.3e}`.",
      f"2. Known single frequency identified: **{'yes' if summary['single_frequency']['dense_transition'] else 'no'}**.",
      "3. Dense-to-sparse error (100%, 75%, 50%, 25%, 10%): " + ", ".join(f"{r['sfp_error']:.3e}" for r in sparse)+".",
      "4. Zero-fill comparison: " + ", ".join(f"SFP {r['sfp_error']:.2e} vs ZF {r['zero_fill_error']:.2e}" for r in sparse)+".",
      f"5. Known sinusoid on real SLAT coordinates: error `{real.get('known_sinusoid_error',float('nan')):.3e}`.",
      f"6. Low/high orthogonality: `{checks['orthogonality']:.3e}`.",
      f"7. Idempotence error: `{checks['idempotence']:.3e}`.",
      f"8. Nested-space error: `{summary['nestedness']:.3e}`.",
      f"9. 32-D channel rotation error: `{checks['channel_rotation']:.3e}`.",
      f"10. Translation / rotation reconstruction errors: `{checks['translation']:.3e}` / `{checks['coordinate_rotation']:.3e}`. Rotation is only approximate for a finite Cartesian lattice.",
      "11. Real SLAT radial spectrum: see `real_slat/spectra/slat_radial_spectrum_{sum,mean}.png` and CSV.",
      f"12. Decode coarse-to-fine progression: **{decoder.get('status','not_run')}**; {decoder.get('note','no decoder evidence')}.",
      f"13. High-only semantics: {decoder.get('high_only','not established')}.",
      f"14. Band semantics: {decoder.get('bands','not established')}.",
      f"15. Conditioning: `{real.get('condition_first',float('nan')):.3e}` at first shell to `{real.get('condition_last',float('nan')):.3e}` at last shell.",
      f"16. Real-SLAT baselines at cutoff 2: {baselines.get('conclusion','not run')}.",
      "","## Scope and interpretation","",
      "The projection uses no graph, octree, hierarchy, PCA split, or decoder Jacobian. Coordinates are unchanged. The real cos/sin basis is exactly equivalent to conjugate-paired complex plane waves. No regularization was used. A mathematical spatial projection and decoded coarse/fine semantics are separate claims; the verdict does not infer the latter without decoder outputs.",""]
    exploration=summary.get("exploration",{})
    if exploration:
        lines[-1:-1]=["## Autonomous follow-up experiments","",
          f"- Best decoder-facing filter tested: `{exploration.get('best_filter','carrier-preserving support-normalized Gaussian')}`.",
          f"- Gaussian alpha-path result: {exploration.get('gaussian_alpha','non-monotonic').rstrip('.') }.",
          f"- Factorial attribution: {exploration.get('factorial','shape high controls geometry; texture high controls appearance').rstrip('.') }.",
          f"- Support-scale result: {exploration.get('support_scale','reducing support alone did not produce a solid coarse shape').rstrip('.') }.",
          f"- Real baselines: {exploration.get('real_baselines','see baseline metrics and fixed-view renders').rstrip('.') }.",
          f"- Mixed residual schedule: {exploration.get('mixed_alpha','shape alpha≈0.75 with texture alpha≈0.25 gives a recognizable but still reduced-detail mesh').rstrip('.') }.",
          "- Recommended next experiment: use the existing learned C64→C128→C256 support hierarchy, or freeze the decoder and optimize a decoder-compatible low latent against blurred geometry/render targets. Keep `F_H=F-F_L` and evaluate `D(F_L+αF_H)`, since high-only is not a valid semantic test for a nonlinear decoder.",
          "- Related references: PointConv (density-compensated point convolution), Multiresolution Deep Implicit Functions (residual bands), MINER (Laplacian-pyramid implicit representation), BACON/BARF (explicit band-limited progressive representations).", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device=torch.device(args.device); out=args.output; out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed)
    config=vars(args).copy(); config["output"]=str(args.output); config["slat_cache"]=str(args.slat_cache) if args.slat_cache else None
    config.update({"physical_cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"regularization_lambda":0.0,
                   "basis":"constant + conjugate-paired real cosine/sine","solver":"torch.linalg.lstsq"})
    (out/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    points, cube=grid(args.grid_size,device); values=signal(points)[:,None]
    pool=canonical_frequencies(args.max_radius).to(device); freq=frequencies_for_cutoff(pool,args.cutoff)
    gt=fft_low(values.reshape((args.grid_size,)*3+(1,)),args.cutoff).reshape_as(values)
    dense_proj=project(points,values,freq)
    dense={"fft_relative_error":rel(dense_proj.low,gt),"condition_number":dense_proj.condition_number,
           "rank":dense_proj.rank,"columns":dense_proj.columns}
    (out/"synthetic_dense").mkdir(exist_ok=True); (out/"synthetic_dense"/"metrics.json").write_text(json.dumps(dense,indent=2))
    if dense["fft_relative_error"]>1e-6:
        raise RuntimeError(f"Dense FFT sanity check failed: {dense['fft_relative_error']:.3e}")

    sparse_rows=[]
    for i,ratio in enumerate(RATIOS):
        idx=support_indices("uniform_random",points,ratio,args.seed+i)
        p=project(points[idx],values[idx],freq)
        sparse_rows.append({"ratio":ratio,"samples":len(idx),"sfp_error":rel(p.low,gt[idx]),
                            "zero_fill_error":rel(zero_fill_baseline(values,idx,(args.grid_size,)*3,args.cutoff),gt[idx]),
                            "raw_nudft_error":rel(raw_nudft(points[idx],values[idx],freq),gt[idx]),
                            "condition_number":p.condition_number})
    save_csv(out/"synthetic_sparse_random"/"errors.csv",sparse_rows)
    x=[r["ratio"]*100 for r in sparse_rows]
    line_plot(out/"synthetic_sparse_random"/"reconstruction_error_vs_sampling_ratio.png",x,{"SFP":[r["sfp_error"] for r in sparse_rows]},"sampling (%)","relative error",True)
    line_plot(out/"baselines"/"sparse_fourier_vs_zero_fill_fft.png",x,{"SFP":[r["sfp_error"] for r in sparse_rows],"zero-fill FFT":[r["zero_fill_error"] for r in sparse_rows],"raw NUDFT mask":[r["raw_nudft_error"] for r in sparse_rows]},"sampling (%)","relative error",True)

    pattern_rows=[]
    for kind in ("uniform_random","surface","clustered"):
        idx=support_indices(kind,points,.25,args.seed+30)
        p=project(points[idx],values[idx],freq)
        smooth=gaussian_local(points[idx],values[idx])
        pattern_rows.append({"pattern":kind,"samples":len(idx),"sfp_error":rel(p.low,gt[idx]),
                             "zero_fill_error":rel(zero_fill_baseline(values,idx,(args.grid_size,)*3,args.cutoff),gt[idx]),
                             "raw_nudft_error":rel(raw_nudft(points[idx],values[idx],freq),gt[idx]),
                             "local_smoothing_error":rel(smooth,gt[idx]),"condition_number":p.condition_number})
    save_csv(out/"synthetic_surface"/"pattern_baselines.csv",pattern_rows)

    k0=torch.tensor([3.,0.,0.],device=device,dtype=points.dtype)
    single=torch.sin(2*math.pi*(points@k0))[:,None]
    cutoffs=list(np.linspace(0,args.max_radius,args.max_radius*2+1)); leakage={}
    support_sets={"dense":torch.arange(len(points),device=device),"50%":support_indices("uniform_random",points,.5,args.seed+50),"25%":support_indices("uniform_random",points,.25,args.seed+51)}
    for name,idx in support_sets.items():
        leakage[name]=[energy_ratio(project(points[idx],single[idx],frequencies_for_cutoff(pool,c)).low,single[idx]) for c in cutoffs]

    checks_features=torch.cat([values.to(torch.float64)*(1+.03*i)+torch.sin(2*math.pi*(points@torch.tensor([i%4+1.,(i//4)%3,0.],device=device,dtype=points.dtype)))[:,None]*.02 for i in range(32)],dim=1).to(torch.float64)
    checks=numerical_checks(points,checks_features,freq,args.seed+70)
    save_csv(out/"numerical_checks"/"projection_properties.csv",[checks])
    save_csv(out/"numerical_checks"/"orthogonality.csv",[{"support":"dense","value":checks["orthogonality"]}])
    save_csv(out/"numerical_checks"/"idempotence.csv",[{"support":"dense","value":checks["idempotence"]}])
    save_csv(out/"numerical_checks"/"rotation_invariance.csv",[{"support":"dense","value":checks["coordinate_rotation"]}])
    save_csv(out/"numerical_checks"/"translation_invariance.csv",[{"support":"dense","value":checks["translation"]}])
    low_small=project(points,checks_features,frequencies_for_cutoff(pool,max(1,args.cutoff-1))).low
    nested=rel(project(points,project(points,checks_features,freq).low,frequencies_for_cutoff(pool,max(1,args.cutoff-1))).low,low_small)
    save_csv(out/"numerical_checks"/"nestedness.csv",[{"small_cutoff":max(1,args.cutoff-1),"large_cutoff":args.cutoff,"value":nested}])

    conv=[]
    full_freq=frequencies_for_cutoff(pool,float(args.max_radius))
    for target in (64,128,256,512):
        pairs=max(0,(target-1)//2); selected=full_freq[:min(pairs,len(full_freq))]
        p=project(points,values,selected)
        conv.append({"requested_columns":target,"actual_columns":p.columns,"relative_to_signal":rel(p.low,values),"condition_number":p.condition_number})
    save_csv(out/"numerical_checks"/"projection_convergence.csv",conv)
    line_plot(out/"numerical_checks"/"projection_convergence_vs_num_frequencies.png",[r["actual_columns"] for r in conv],{"error":[r["relative_to_signal"] for r in conv]},"basis columns","relative error",True)

    real_summary={}; real_spectrum=[]
    if args.slat_cache and args.slat_cache.exists():
        rp,rf,tf=load_slat(args.slat_cache,device)
        known=torch.sin(2*math.pi*(rp@torch.tensor([1.,1.,0.],device=device,dtype=rp.dtype)))[:,None]
        known_proj=project(rp,known,frequencies_for_cutoff(pool,math.sqrt(2)+1e-6))
        real_checks=numerical_checks(rp,rf.to(torch.float64),freq,args.seed+80)
        leakage["real SLAT"]=[energy_ratio(project(rp,known,frequencies_for_cutoff(pool,c)).low,known) for c in cutoffs]
        real_spectrum=shell_spectrum(rp,rf.to(torch.float64),pool,args.max_radius)
        specdir=out/"real_slat"/"spectra"; save_csv(specdir/"radial_spectrum.csv",real_spectrum)
        save_csv(out/"numerical_checks"/"condition_numbers.csv",real_spectrum)
        line_plot(specdir/"slat_radial_spectrum_sum.png",[r["radius"] for r in real_spectrum],{"shell sum":[r["shell_sum"] for r in real_spectrum]},"radius","energy",True)
        line_plot(specdir/"slat_radial_spectrum_mean.png",[r["radius"] for r in real_spectrum],{"shell mean":[r["shell_mean"] for r in real_spectrum]},"radius","mean energy",True)
        line_plot(out/"numerical_checks"/"condition_number_vs_cutoff.png",[r["radius"] for r in real_spectrum],{"condition number":[r["condition_number"] for r in real_spectrum]},"cutoff","condition number",True)
        real_summary={"path":str(args.slat_cache),"tokens":len(rp),"channels":rf.shape[1],"known_sinusoid_error":rel(known_proj.low,known),**real_checks,
                      "condition_first":real_spectrum[0]["condition_number"],"condition_last":real_spectrum[-1]["condition_number"]}
        save_csv(out/"synthetic_real_support"/"metrics.csv",[real_summary])
        # Save all requested low/high cutoffs with coordinates exactly unchanged.
        cutdir=out/"real_slat"/"projected_latents"; cutdir.mkdir(parents=True,exist_ok=True)
        for rho in (.05,.10,.20,.30,.40,.60,.80,1.0):
            cutoff=rho*args.max_radius; pr=project(rp,rf,frequencies_for_cutoff(pool,cutoff)); pt=project(rp,tf,frequencies_for_cutoff(pool,cutoff))
            torch.save({"rho":rho,"cutoff":cutoff,"coords_normalized":rp.cpu(),"shape_low":pr.low.cpu(),"shape_high":pr.high.cpu(),"texture_low":pt.low.cpu(),"texture_high":pt.high.cpu()},cutdir/f"rho_{rho:.2f}.pt")
    line_plot(out/"numerical_checks"/"single_frequency_recovery_curve.png",cutoffs,leakage,"cutoff","recovered energy ratio")
    single_summary={"frequency":[3,0,0],"dense_transition":bool(leakage["dense"][cutoffs.index(2.5)]<1e-6 and leakage["dense"][cutoffs.index(3.0)]>.999)}
    summary={"dense":dense,"sparse_random":sparse_rows,"patterns":pattern_rows,"single_frequency":single_summary,
             "checks":checks,"nestedness":nested,"real":real_summary,
             "decoder":{"status":"not_run","note":"projected real latents were saved for decoder execution","high_only":"not established","bands":"not established"}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    write_report(out/"report.md",summary)
    print(json.dumps({"output":str(out),"dense_fft_error":dense["fft_relative_error"],"verdict_report":str(out/"report.md")},indent=2))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("sparse_fourier_validation"))
    p.add_argument("--slat-cache",type=Path,default=Path("outputs/global4096_singleview_shared_slat_shape_tex_sr_cuda4/exp_c_baseline4096_from1024/baseline/baseline_c64_latents.pt"))
    p.add_argument("--device",default="cuda")
    p.add_argument("--grid-size",type=int,default=24)
    p.add_argument("--cutoff",type=float,default=2.0)
    p.add_argument("--max-radius",type=int,default=5)
    p.add_argument("--seed",type=int,default=20260826)
    return p.parse_args()


if __name__=="__main__": run(parse_args())
