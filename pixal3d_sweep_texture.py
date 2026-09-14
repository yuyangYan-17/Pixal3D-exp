"""Synchronized C64 texture flow, final voxel material, input-view foreground PSNR."""
from pathlib import Path
import math

import numpy as np
import torch
from PIL import Image

import pixal3d_cascade512_1024_tiled2048_crop_condition as ops

VERSION = 2


def load_texture_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("experiment", {}).get("version") != VERSION:
        raise RuntimeError(f"incompatible material cascade checkpoint: {path}")
    return payload


def texture_schedule(sampler, start_t, steps, rescale_t):
    """Use all requested steps between the geometry start_t and zero."""
    if not 0 < start_t <= 1:
        raise ValueError("start_t must be in (0, 1]")
    return [start_t * t for t in sampler.timestep_schedule(steps, rescale_t)]


def noise_encoded_texture(initial, noise, start_t, sigma_min):
    if initial.shape != noise.shape:
        raise ValueError("texture initialization and noise must align")
    return (1 - start_t) * initial + (sigma_min + (1 - sigma_min) * start_t) * noise


@torch.no_grad()
def prepare_baseline_material(state, args, mode):
    """Native C64 texture flow -> 1024 material field, once per image mode."""
    pipe = state.pipeline
    geometry_root = Path(args.geometry_source or args.output_dir).resolve()
    out = Path(args.output_dir).resolve() / "baseline_texture" / mode
    signature = dict(version=VERSION, mode=mode, seed=args.tex_seed, steps=12,
                     shape_source=str(geometry_root / "baseline" / "shape_c64_shape1024_denormalized.pt"))
    cache = out / "material1024.pt"
    if args.resume and cache.is_file():
        saved = load_texture_payload(cache)
        if saved["experiment"] != signature:
            raise RuntimeError(f"incompatible baseline material: {cache}")
        return saved["mesh"], signature
    print(f"[baseline texture/{mode}] native C64 flow, 12 steps -> material1024", flush=True)
    saved = ops.load_payload(Path(signature["shape_source"]))
    slat = ops.SparseTensor(saved["features"].to(state.device), saved["coords"].to(state.device))
    cond = pipe.get_proj_cond_shape(
        pipe.image_cond_model_tex_1024, [state.canonical["image_1024"]], slat.coords,
        camera_angle_x=float(state.camera["camera_angle_x"]), distance=float(state.camera["distance"]),
        mesh_scale=float(state.camera.get("mesh_scale", 1)), grid_resolution_override=64)
    mean, std = ops.normalization_tensors(pipe.shape_slat_normalization, state.device)
    shape = slat.replace((slat.feats - mean) / std)
    model = pipe.models["tex_slat_flow_model_1024"]
    x = slat.replace(torch.randn((len(slat.coords), model.in_channels - shape.feats.shape[1]),
                                generator=torch.Generator().manual_seed(args.tex_seed)).to(state.device))
    times = pipe.tex_slat_sampler.timestep_schedule(12, pipe.tex_slat_sampler_params.get("rescale_t", 1.0))
    selected = cond["cond" if mode == "conditional" else "neg_cond"]
    model.to(state.device)
    try:
        for t, next_t in zip(times, times[1:]):
            pred = model(x, torch.tensor([1000 * t], device=state.device), selected, concat_cond=shape)
            x = x.replace(x.feats - (t - next_t) * pred.feats)
    finally:
        model.cpu()
    mean, std = ops.normalization_tensors(pipe.tex_slat_normalization, state.device)
    tex = x.replace(x.feats * std + mean)
    del cond, selected, shape, pred, x
    ops.empty_cuda()
    meshes = pipe.decode_latent(slat, tex, 1024)
    if len(meshes) != 1 or not torch.isfinite(meshes[0].attrs).all():
        raise RuntimeError("invalid baseline material field")
    mesh = meshes[0].cpu()
    ops.atomic_save(cache, dict(mesh=mesh, experiment=signature))
    ops.atomic_json(out / "summary.json", dict(experiment=signature, material_voxels=len(mesh.coords),
                                              query="MeshWithVoxel.query_attrs, trilinear"))
    del meshes, slat, tex
    ops.empty_cuda()
    return mesh, signature


@torch.no_grad()
def prepare_texture_initial(state, args, mode, coords, source=None):
    """Query baseline material on the pre-Shape-flow voxel4096, then encode C256."""
    from pixal3d import models
    from pixal3d.models.sc_vaes.sparse_unet_vae import align_sparse_tensor_to_coords
    source = Path(source or state.out)
    out = state.out / "texture"
    cache = out / "initial_c256.pt"
    signature = dict(version=VERSION, mode=mode, seed=args.tex_seed,
                     geometry_source=str(source.resolve()), encoder=str(args.texture_encoder_path.resolve()),
                     query="baseline1024 trilinear at bridge voxel4096 dual vertices in world coordinates")
    if args.resume and cache.is_file():
        saved = load_texture_payload(cache)
        if saved["experiment"] != signature or not torch.equal(saved["coords"], coords):
            raise RuntimeError(f"incompatible initial texture: {cache}")
        return saved["features"]
    bridge_shape = ops.load_payload(source / "bridge" / "encoded_s256.pt")
    if not torch.equal(bridge_shape["coords"], coords):
        raise RuntimeError("final Shape flow support must equal pre-flow Encoder C256 support")
    del bridge_shape
    baseline, baseline_signature = prepare_baseline_material(state, args, mode)
    voxel_cache = source / "bridge" / "voxel4096.pt"
    if voxel_cache.is_file():
        voxel = ops.load_payload(voxel_cache)
        indices, dual = voxel["coords"], voxel["dual"]
        del voxel
    else:
        # Earlier geometry runs did not retain voxel4096; regenerate using the exact same mesh and QEF settings.
        import o_voxel
        bridge_mesh = ops.load_payload(source / "bridge" / "mesh2048" / "geometry_mesh.pt")["mesh"]
        print("[texture transfer] bridge mesh2048 -> voxel4096 (same Shape Encoder input)", flush=True)
        indices, dual, intersections = o_voxel.convert.mesh_to_flexible_dual_grid(
            bridge_mesh.vertices.float().cpu(), bridge_mesh.faces.long().cpu(), grid_size=4096,
            aabb=[[-0.5] * 3, [0.5] * 3], face_weight=1.0, boundary_weight=0.2,
            regularization_weight=1e-2, timing=True)
        del bridge_mesh, intersections
    baseline = baseline.to(state.device)
    attrs = torch.empty((len(indices), baseline.attrs.shape[1]), dtype=torch.float32)
    print(f"[texture transfer] trilinear queries at {len(indices):,} voxel4096 dual vertices", flush=True)
    for begin in range(0, len(indices), args.material_query_chunk_size):
        end = min(begin + args.material_query_chunk_size, len(indices))
        # The voxelizer returns AABB-translated dual positions in [0,1].
        xyz = dual[begin:end].to(state.device).float() - 0.5
        attrs[begin:end] = baseline.query_attrs(xyz).float().cpu()
    if not torch.isfinite(attrs).all():
        raise RuntimeError("baseline query returned nonfinite material")
    voxel_coords = torch.cat([torch.zeros_like(indices[:, :1]), indices], 1).int()
    ops.atomic_save(out / "baseline_material4096.pt", dict(
        coords=voxel_coords, attrs=attrs, resolution=4096, experiment=signature,
        baseline=baseline_signature, layout=baseline.layout))
    stats = dict(voxel_tokens=len(indices), nonzero_material_fraction=float((attrs.abs().sum(1) > 0).float().mean()),
                 baseline=baseline_signature, query=signature["query"])
    del baseline, indices, dual, xyz
    ops.empty_cuda()
    print("[texture transfer] Texture Encoder posterior mean -> C256", flush=True)
    encoder = models.from_pretrained(str(args.texture_encoder_path)).eval().to(state.device)
    voxels = ops.SparseTensor((attrs * 2 - 1).to(state.device), voxel_coords.to(state.device))
    del attrs, voxel_coords
    try:
        encoded = encoder(voxels, sample_posterior=False)
        # Both encoders have the same four S2C reductions of the exact same voxel support.
        if len(encoded.coords) != len(coords):
            raise RuntimeError(f"Texture/Shape C256 support sizes differ: {len(encoded.coords)} vs {len(coords)}")
        encoded, alignment = align_sparse_tensor_to_coords(encoded, coords, missing="error")
        mean, std = ops.normalization_tensors(state.pipeline.tex_slat_normalization, state.device)
        normalized = ((encoded.feats.float() - mean) / std).cpu()
        if not torch.isfinite(normalized).all():
            raise RuntimeError("nonfinite encoded texture")
    finally:
        encoder.cpu()
    stats.update(alignment=alignment, latent_tokens=len(coords), experiment=signature)
    ops.atomic_save(cache, dict(coords=coords, features=normalized, experiment=signature))
    ops.atomic_json(out / "initial_summary.json", stats)
    del encoder, encoded, voxels
    ops.empty_cuda()
    return normalized


class UnlitEnvironment:
    def shade(self, pos, normal, kd, ks, view_pos, specular=True):
        return kd


def save_texture_sweep(root):
    import json
    from PIL import ImageDraw
    rows = []
    for mode in ("conditional", "unconditional"):
        sheet = Image.new("RGB", (4 * 384, 3 * (384 + 32)))
        draw = ImageDraw.Draw(sheet)
        for start in range(12):
            folder = root / mode / f"start_{start:02d}" / "texture"
            if not (folder / "metrics.json").is_file():
                continue
            metrics = json.loads((folder / "metrics.json").read_text())
            rows.append(dict(mode=mode, start_step=start, **metrics))
            x, y = start % 4 * 384, start // 4 * 416
            with Image.open(folder / "input_view_texture.png") as img:
                sheet.paste(img.resize((384, 384)), (x, y))
            psnr = metrics["foreground_psnr_db"]
            label = f"{psnr:.3f} dB" if psnr is not None else "inf dB"
            draw.text((x + 5, y + 389), f"start={start} PSNR={label}", fill="white")
        sheet.save(root / f"{mode}_texture_3x4.png")
    ops.atomic_json(root / "texture_psnr.json", dict(completed=len(rows), total=24, experiments=rows))


def foreground_metrics(prediction, reference, mask):
    if prediction.shape != reference.shape or mask.shape != reference.shape[:2]:
        raise ValueError("RGB images and foreground mask must align")
    if not mask.any():
        raise ValueError("input foreground is empty")
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite texture render")
    mse = float(np.mean((prediction[mask].astype(np.float64) - reference[mask]) ** 2))
    return dict(foreground_mse=mse, foreground_psnr_db=-10 * math.log10(mse) if mse else None,
                perfect_match=mse == 0, foreground_pixels=int(mask.sum()),
                definition="RGB [0,1], canonical input alpha > 0.5; input foreground only, including missed geometry")


@torch.no_grad()
def run_texture(state, args, mode, start_step, source=None):
    if mode not in ("conditional", "unconditional"):
        raise ValueError(f"invalid texture condition mode: {mode}")
    source = Path(source or state.out)
    out = state.out / "texture"
    pipe = state.pipeline
    shape_times = pipe.shape_slat_sampler.timestep_schedule(12, pipe.shape_slat_sampler_params.get("rescale_t", 1.0))
    if not 0 <= start_step < 12:
        raise ValueError("start_step must be 0..11")
    start_t = shape_times[start_step]
    signature = dict(version=VERSION, mode=mode, steps=args.tex_steps, seed=args.tex_seed,
                     resolution=args.texture_resolution, cfg=False, geometry_source=str(source.resolve()),
                     start_step=start_step, start_t=start_t, initialization="baseline1024_query_voxel4096_tex_encoder",
                     encoder=str(args.texture_encoder_path.resolve()))
    marker = out / "metrics.json"
    if args.resume and marker.is_file():
        import json
        result = json.loads(marker.read_text())
        if result.get("experiment") != signature:
            raise RuntimeError(f"incompatible texture cache: {marker}")
        return result
    print(f"[texture/{mode}] source={source}, steps={args.tex_steps}, seed={args.tex_seed}", flush=True)
    payload = ops.load_payload(source / "shape" / "global_c256_shape_denormalized.pt")
    coords, raw = payload["coords"], payload["features"].float()
    mean, std = ops.normalization_tensors(pipe.shape_slat_normalization, raw.device)
    shape = (raw - mean) / std
    records, _ = ops.build_overlap_records(coords, 256, 64, 32)
    active = [r for r in records if r["global_row_ids"].numel()]
    model = pipe.models["tex_slat_flow_model_1024"]
    cache = out / "latent.pt"
    if args.resume and cache.is_file():
        saved = load_texture_payload(cache)
        if saved["experiment"] != signature or not torch.equal(saved["coords"], coords):
            raise RuntimeError(f"incompatible texture latent: {cache}")
        x = saved["features"]
    else:
        initial = prepare_texture_initial(state, args, mode, coords, source=source)
        cond = pipe.get_proj_cond_shape(
            pipe.image_cond_model_tex_1024, [state.canonical["image_1024"]], coords.to(state.device),
            camera_angle_x=float(state.camera["camera_angle_x"]), distance=float(state.camera["distance"]),
            mesh_scale=float(state.camera.get("mesh_scale", 1)), grid_resolution_override=256)
        selected = cond["cond" if mode == "conditional" else "neg_cond"]
        global_features = selected["global"].cpu()
        projected = selected["proj"].feats.cpu()
        conditions = {r["cube_id"]: {"global": global_features, "proj": projected[r["global_row_ids"]]}
                      for r in active}
        del cond, selected, projected
        ops.empty_cuda()
        noise = torch.randn((len(coords), model.in_channels - shape.shape[1]),
                        generator=torch.Generator().manual_seed(args.tex_seed))
        x = noise_encoded_texture(initial, noise, start_t, pipe.tex_slat_sampler.sigma_min)
        times = texture_schedule(pipe.tex_slat_sampler, start_t, args.tex_steps,
                                 pipe.tex_slat_sampler_params.get("rescale_t", 1.0))
        ops.atomic_json(out / "schedule.json", dict(start_step=start_step, start_t=start_t, times=times,
                        executed_steps=len(times)-1, formula="x_t=(1-t)*encoded_texture+(sigma_min+(1-sigma_min)*t)*noise"))
        del initial, noise
        model.to(state.device)
        try:
            for step, (t, next_t) in enumerate(zip(times, times[1:])):
                velocity = torch.empty_like(x)
                writes = torch.zeros(len(coords), dtype=torch.int16)
                for offset in range(0, len(active), args.flow_batch_size):
                    batch = active[offset:offset + args.flow_batch_size]
                    packed = ops.pack_sparse(batch, x).to(state.device)
                    shape_cond = ops.pack_sparse(batch, shape).to(state.device)
                    image_cond = ops.pack_conditions(batch, conditions, packed.coords, state.device)["cond"]
                    pred = model(packed, torch.full((len(batch),), 1000 * t, device=state.device),
                                 image_cond, concat_cond=shape_cond)
                    if not torch.equal(pred.coords, packed.coords):
                        raise RuntimeError("texture flow changed sparse coordinates")
                    feats = pred.feats.float().cpu()
                    cursor = 0
                    for record in batch:
                        rows, mask = record["global_row_ids"], record["owner_mask"]
                        velocity[rows[mask]] = feats[cursor:cursor + len(rows)][mask]
                        writes[rows[mask]] += 1
                        cursor += len(rows)
                    del packed, shape_cond, image_cond, pred, feats
                if not (writes == 1).all() or not torch.isfinite(velocity).all():
                    raise RuntimeError("invalid texture velocity or owner coverage")
                x = x - (t - next_t) * velocity
                print(f"[texture/{mode}] synchronized step {step + 1}/{args.tex_steps}, {len(active)} contexts", flush=True)
        finally:
            model.cpu()
        ops.atomic_save(cache, dict(coords=coords, features=x, experiment=signature))
        del conditions
    mean, std = ops.normalization_tensors(pipe.tex_slat_normalization, x.device)
    tex = ops.SparseTensor((x * std + mean).to(state.device), coords.to(state.device))
    slat = ops.SparseTensor(raw.to(state.device), coords.to(state.device))
    del shape, x, raw, payload
    ops.empty_cuda()
    print("[texture] decode global C256 geometry + guided material at 4096", flush=True)
    meshes = pipe.decode_latent(slat, tex, 4096)
    mesh = meshes[0]
    del slat, tex
    ops.empty_cuda()
    ops.atomic_save(out / "textured_mesh.pt", dict(mesh=mesh.cpu(), experiment=signature))
    from pixal3d.renderers import PbrMeshRenderer
    from pixal3d.utils.render_utils import proj_camera_to_render_params
    extr, intr = proj_camera_to_render_params(float(state.camera["camera_angle_x"]), float(state.camera["distance"]))
    renderer = PbrMeshRenderer(dict(resolution=args.texture_resolution, near=0.1, far=100,
                                   ssaa=1, peel_layers=1, face_chunk_size=args.normal_face_chunk_size), device=state.device)
    rendered = renderer.render(mesh, extr, intr, envmap=UnlitEnvironment())
    prediction = rendered["base_color"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    size = (args.texture_resolution,) * 2
    reference_image = state.canonical["image_4096"].resize(size, Image.Resampling.LANCZOS)
    reference = np.asarray(reference_image).astype(np.float32) / 255
    mask = np.asarray(state.canonical["foreground_mask_4096"].resize(size, Image.Resampling.LANCZOS)) > 127
    metrics = foreground_metrics(prediction, reference, mask)
    metrics.update(experiment=signature, render="unlit base_color", status="complete",
                   foreground_render_coverage=float((rendered["mask"].cpu().numpy()[mask] > 0).mean()))
    Image.fromarray((prediction * 255).round().astype(np.uint8)).save(out / "input_view_texture.png")
    reference_image.save(out / "input_reference.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(out / "foreground_mask.png")
    ops.atomic_json(marker, metrics)
    print(f"[texture/{mode}] foreground PSNR={metrics['foreground_psnr_db']} dB", flush=True)
    del meshes, mesh, rendered
    ops.empty_cuda()
    return metrics
