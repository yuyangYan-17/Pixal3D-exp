"""XY crop ownership, overlapping depth contexts, and synchronous shape flow.

Each batch reads the same frozen global state. Only owner-image predictions
contribute; overlapping depth predictions are averaged before one Euler update.
"""

import contextlib
import gc
import math
from types import MethodType
import numpy as np
import torch as T
from . import common
from .common import SparseTensor, tensor_hash
from .batching import batch_stable_linear


def project(coords, grid, cam):
    points = (coords[:, 1:].double().numpy() + 0.5) / grid - 0.5
    depth = cam["distance"] - points[:, 2]
    f = 2048 / math.tan(cam["camera_angle_x"] / 2)
    # Coordinates are pixel edges, with the half-pixel offset matching the
    # existing crop projection convention at the actual 4096 resolution.
    uv = np.c_[2048 + f * points[:, 0] / depth, 2048 - f * points[:, 1] / depth] + 0.5
    return points, uv, depth


def construct(
    coords,
    stage,
    cam,
    out=None,
    image=None,
    padding=32,
    depth_stride=None,
    image_stride=None,
):
    grid, context, stride, size = (
        (128, 2048, 1024, 32) if stage == 512 else (256, 1024, 1024, 64)
    )
    stride = stride if image_stride is None else image_stride
    assert (
        isinstance(stride, int)
        and 0 < stride <= context
        and (4096 - context) % stride == 0
    )
    depth_stride = size // 2 if depth_stride is None else depth_stride
    assert (
        isinstance(depth_stride, int)
        and 0 < depth_stride <= size
        and (grid - size) % depth_stride == 0
    )
    assert isinstance(padding, int) and padding >= 0
    coords = coords.int().cpu()
    assert ((coords[:, 1:] >= 0) & (coords[:, 1:] < grid)).all()
    assert len(T.unique(coords, dim=0)) == len(coords)
    points, uv, depth = project(coords, grid, cam)
    owner = T.full((len(coords),), -1, dtype=T.long)
    best = np.full(len(coords), np.inf)
    tiles, blocks = [], []
    for y in range(0, 4096 - context + 1, stride):
        for x in range(0, 4096 - context + 1, stride):
            tid = len(tiles)
            core = (
                (depth > 0)
                & (uv[:, 0] >= x)
                & (uv[:, 0] < x + context)
                & (uv[:, 1] >= y)
                & (uv[:, 1] < y + context)
            )
            keep = (
                (depth > 0)
                & (uv[:, 0] >= x - padding)
                & (uv[:, 0] < x + context + padding)
                & (uv[:, 1] >= y - padding)
                & (uv[:, 1] < y + context + padding)
            )
            rows = T.from_numpy(np.flatnonzero(keep)).long()
            p0 = np.array(
                [(x + context / 2) / 4096 - 0.5, 0.5 - (y + context / 2) / 4096, 0.0]
            )
            origin = np.array([-context / 8192, -context / 8192, -0.5])
            shift_float = (0.5 + p0 + origin) * grid
            shift = T.from_numpy(np.rint(shift_float).astype(np.int32))
            assert np.allclose(shift.numpy(), shift_float, atol=1e-12)
            lc = coords[rows].clone()
            lc[:, 1:] -= shift
            restored = lc.clone()
            restored[:, 1:] += shift
            assert T.equal(restored, coords[rows])
            assert np.allclose(
                (lc[:, 1:].double().numpy() + 0.5) / grid + origin,
                points[keep] - p0,
                atol=1e-12,
            )
            distance = (
                ((uv[keep] - [x + context / 2, y + context / 2]) / context) ** 2
            ).sum(1)
            wins = core[keep] & (
                distance < best[keep]
            )  # halo may never claim ownership
            winrows = rows[T.from_numpy(wins)]
            owner[winrows] = tid
            best[winrows.numpy()] = distance[wins]
            tile = dict(
                tile_id=tid,
                rows=rows,
                coords=lc,
                crop=[
                    x - padding,
                    y - padding,
                    x + context + padding,
                    y + context + padding,
                ],
                core_crop=[x, y, x + context, y + context],
                core=T.from_numpy(core[keep]),
                padding=padding,
                P0=p0.tolist(),
                origin=origin.tolist(),
                index_offset=shift,
                grid=grid,
            )
            tiles.append(tile)
            for z in range(0, grid - size + 1, depth_stride):
                lr = T.where((lc[:, 3] >= z) & (lc[:, 3] < z + size))[0]
                bc = lc[lr].clone()
                bc[:, 3] -= z
                assert ((bc[:, 3] >= 0) & (bc[:, 3] < size)).all()
                blocks.append(
                    dict(
                        block_id=len(blocks),
                        tile_id=tid,
                        rows=lr,
                        global_ids=rows[lr],
                        coords=bc,
                        z_start=z,
                    )
                )
            if out is not None and image is not None:
                dest = out / "tiles" / f"{tid:02d}"
                dest.mkdir(parents=True, exist_ok=True)
                crop = image.crop(tuple(tile["crop"]))
                crop.save(dest / "raw.png")
                for resolution in (512, 1024):
                    crop.resize(
                        (resolution, resolution), common.Image.Resampling.LANCZOS
                    ).save(dest / f"input_{resolution}.png")
    if (owner < 0).any():
        raise RuntimeError(
            f"{int((owner < 0).sum())} global points project outside every crop; refusing to drop them"
        )
    counts = T.zeros(len(coords), dtype=T.int32)
    weight_sums = T.zeros(len(coords), dtype=T.float32)
    all_counts = T.zeros_like(counts)
    for b in blocks:
        b["owned"] = owner[b["global_ids"]] == b["tile_id"]
        # Sample a triangular window at voxel centers: strictly positive even
        # at outer-domain edges. For 50% overlap, neighboring weights sum to 1.
        relative_z = (b["coords"][:, 3].float() + 0.5) / size
        b["depth_weights"] = 1.0 - (2.0 * relative_z - 1.0).abs()
        weight_sums.index_add_(
            0, b["global_ids"][b["owned"]], b["depth_weights"][b["owned"]]
        )
        assert not (b["owned"] & ~tiles[b["tile_id"]]["core"][b["rows"]]).any(), (
            "halo points must be context only"
        )
        counts.index_add_(
            0, b["global_ids"][b["owned"]], T.ones(int(b["owned"].sum()), dtype=T.int32)
        )
        all_counts.index_add_(
            0, b["global_ids"], T.ones(len(b["global_ids"]), dtype=T.int32)
        )
    assert (counts > 0).all()
    assert len(tiles) == len(range(0, 4096 - context + 1, stride)) ** 2 and len(
        blocks
    ) == len(tiles) * len(range(0, grid - size + 1, depth_stride))
    data = dict(
        coords=coords,
        tiles=tiles,
        blocks=blocks,
        owner=owner,
        counts=counts,
        all_counts=all_counts,
        depth_weight_sums=weight_sums,
    )
    if out is not None:
        common.save(out / "mapping.pt", **data)
        common.js(
            out / "mapping.json",
            dict(
                stage=stage,
                grid=grid,
                crop_context=context,
                crop_stride=stride,
                padding=padding,
                expanded_crop_context=context + 2 * padding,
                halo_context_only=True,
                halo_points_per_tile=[int((~t["core"]).sum()) for t in tiles],
                tiles=len(tiles),
                candidate_blocks=len(blocks),
                active_blocks=sum(bool(len(b["rows"])) for b in blocks),
                global_points=len(coords),
                uncovered=0,
                depth_context=size,
                depth_stride=depth_stride,
                shape_depth_blend="voxel_center_triangle_v1",
                tile_points=[len(t["rows"]) for t in tiles],
                owned_points=T.bincount(owner, minlength=len(tiles)).tolist(),
                winner_depth_coverage=T.bincount(counts.long()).tolist(),
                local_xy_bounds=[
                    dict(
                        tile=t["tile_id"],
                        min=t["coords"][:, 1:3].amin(0).tolist(),
                        max=t["coords"][:, 1:3].amax(0).tolist(),
                    )
                    for t in tiles
                    if len(t["rows"])
                ],
                nominal_xy_width=context * grid // 4096,
                rotation="identity",
                scale=1,
                ownership="minimum squared normalized image distance to crop center; row-major tie break",
                reduction="average only the winning image depth-window velocities, then one global Euler update",
            ),
        )
    return data


@contextlib.contextmanager
def projection_adapter(extractor, tile, cam, resolution):
    pg = extractor.proj_grid

    def adapted(
        self,
        camera_angle_x,
        distance,
        mesh_scale,
        transform_matrix=None,
        grid_indices=None,
        grid_resolution=None,
    ):
        idx = grid_indices.detach().cpu().numpy().reshape(-1, 3)
        global_idx = idx + tile["index_offset"].numpy()
        coords = T.from_numpy(np.c_[np.zeros(len(idx), dtype=np.int32), global_idx])
        _, uv, depth = project(coords, tile["grid"], cam)
        x, y, x1, y1 = tile["crop"]
        normalized = (uv - [x, y]) / (x1 - x)
        pixels = normalized * resolution - 0.5
        valid = (depth > 0) & (normalized >= 0).all(1) & (normalized < 1).all(1)
        assert valid.all(), "selected crop support must project into its own image"
        return tuple(
            T.as_tensor(
                v,
                device=camera_angle_x.device,
                dtype=T.bool if v.dtype == bool else T.float32,
            )[None]
            for v in (pixels, depth, valid)
        )

    pg.project_grid_indices = MethodType(adapted, pg)
    try:
        yield
    finally:
        del pg.project_grid_indices


@T.no_grad()
def conditions(pipe, data, stage, out, cam):
    bank = {}
    for tile in data["tiles"]:
        if not len(tile["rows"]):
            continue
        dest = out / "tiles" / f"{tile['tile_id']:02d}"
        target = dest / "conditions.pt"
        if target.exists():
            c = common.load_payload(target)
            assert T.equal(c["coords"], tile["coords"])
            bank[tile["tile_id"]] = {k: c[k] for k in ("global_", "proj")}
            continue
        extractor = getattr(pipe, f"image_cond_model_shape_{stage}")
        image = common.Image.open(dest / f"input_{stage}.png").convert("RGB")
        with projection_adapter(extractor, tile, cam, stage):
            cond = pipe.get_proj_cond_shape(
                extractor,
                [image],
                tile["coords"].to(pipe.device),
                camera_angle_x=cam["camera_angle_x"],
                distance=cam["distance"],
                mesh_scale=1.0,
                grid_resolution_override=stage // 16,
            )["cond"]
        bank[tile["tile_id"]] = dict(
            global_=cond["global"].cpu(), proj=cond["proj"].feats.cpu()
        )
        assert len(bank[tile["tile_id"]]["proj"]) == len(tile["rows"])
        common.save(target, coords=tile["coords"], **bank[tile["tile_id"]])
        del cond
        common.empty_cuda()
        print("CONDITION", stage, tile["tile_id"], len(tile["rows"]), flush=True)
    return bank


def groups(blocks, batch_size, max_tokens):
    group = []
    tokens = 0
    for b in blocks:
        if not len(b["rows"]):
            continue
        if group and (len(group) >= batch_size or tokens + len(b["rows"]) > max_tokens):
            yield group
            group = []
            tokens = 0
        group.append(b)
        tokens += len(b["rows"])
    if group:
        yield group


@T.no_grad()
def predict(pipe, model, group, x, bank, t, params):
    cs = []
    xs = []
    ps = []
    gs = []
    lengths = []
    for bid, b in enumerate(group):
        c = b["coords"].clone()
        c[:, 0] = bid
        cs.append(c)
        xs.append(x[b["global_ids"]])
        lengths.append(len(c))
        con = bank[b["tile_id"]]
        ps.append(con["proj"][b["rows"]])
        gs.append(con["global_"])
    c = T.cat(cs).to(pipe.device)
    xt = SparseTensor(T.cat(xs).to(pipe.device), c)
    pr = SparseTensor(T.cat(ps).to(pipe.device), c)
    g = T.cat(gs).to(pipe.device)
    with batch_stable_linear(model):
        pred = pipe.shape_slat_sampler._inference_model(
            model,
            xt,
            t,
            cond={"global": g, "proj": pr},
            neg_cond={
                "global": T.zeros_like(g),
                "proj": pr.replace(T.zeros_like(pr.feats)),
            },
            **params,
        )
    assert T.equal(pred.coords, c) and T.isfinite(pred.feats).all()
    return list(pred.feats.float().cpu().split(lengths))


@T.no_grad()
def predict_safe(pipe, model, group, x, bank, t, params):
    try:
        return predict(pipe, model, group, x, bank, t, params)
    except T.cuda.OutOfMemoryError:
        if len(group) == 1:
            raise
    gc.collect()
    common.empty_cuda()
    mid = len(group) // 2
    print("OOM batch split", len(group), "->", mid, len(group) - mid, flush=True)
    return predict_safe(pipe, model, group[:mid], x, bank, t, params) + predict_safe(
        pipe, model, group[mid:], x, bank, t, params
    )


def reduce_predictions(summed, count, group, values, weight_sum=None):
    for b, v in zip(group, values):
        mask = b["owned"]
        ids = b["global_ids"][mask]
        if weight_sum is None:
            summed.index_add_(0, ids, v[mask])
        else:
            weights = b["depth_weights"][mask]
            summed.index_add_(0, ids, v[mask] * weights[:, None])
            weight_sum.index_add_(0, ids, weights)
        count.index_add_(0, ids, T.ones(len(ids), dtype=T.int32))


@T.no_grad()
def flow(pipe, data, bank, stage, out, args, initial=None, smoke=False):
    model = pipe.models[f"shape_slat_flow_model_{stage}"]
    sampler = pipe.shape_slat_sampler
    params = dict(pipe.shape_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    seed = 43 if stage == 512 else 44
    noise = T.randn(
        (len(data["coords"]), model.in_channels),
        generator=T.Generator().manual_seed(seed),
    )
    assert times[0] == 1.0, "this experiment starts from the full-noise endpoint"
    x = noise.clone()
    common.save(
        out / "initial.pt",
        coords=data["coords"],
        noise=noise,
        state=x,
        seed=seed,
        encoder_latent=initial if initial is not None else None,
    )
    common.js(
        out / "sampler.json",
        dict(
            params=params,
            times=times,
            seed=seed,
            initialization="one global Gaussian field at t=1, shared by every image/depth block",
        ),
    )
    batches = list(groups(data["blocks"], args.batch_size, args.max_tokens))
    model.to(pipe.device)
    try:
        if smoke:
            active = [b for b in data["blocks"] if len(b["rows"])]
            first = max(active, key=lambda b: len(b["rows"]))
            other = max(
                (b for b in active if b["tile_id"] != first["tile_id"]),
                key=lambda b: len(b["rows"]),
            )
            group = [first, other]
            errors = []
            for t in (1.0, 0.5):
                bat = T.cat(predict(pipe, model, group, x, bank, t, params))
                serial = T.cat(
                    [predict(pipe, model, [b], x, bank, t, params)[0] for b in group]
                )
                diff = (bat - serial).abs()
                relative = float(
                    diff.square().mean().sqrt()
                    / serial.square().mean().sqrt().clamp_min(1e-8)
                )
                errors.append(
                    dict(t=t, max_error=float(diff.max()), relative_rms=relative)
                )
                assert relative < 1e-4 and float(diff.max()) < 0.005, errors[-1]
            common.js(
                out / "batch_equivalence.json",
                dict(
                    pass_=True, block_ids=[b["block_id"] for b in group], records=errors
                ),
            )
        for step, (t, tn) in enumerate(zip(times, times[1:])):
            if smoke and step >= 1:
                break
            checkpoint = out / f"step_{step:02d}.pt"
            if checkpoint.exists():
                c = common.load_payload(checkpoint)
                assert c.get("depth_blend") == "voxel_center_triangle_v1", (
                    "Old uniform-blend checkpoint; use a new output directory"
                )
                assert (
                    T.equal(c["coords"], data["coords"])
                    and c["t"] == t
                    and c["t_next"] == tn
                )
                assert c["input_hash"] == tensor_hash(x)
                x = c["features"]
                assert T.isfinite(x).all()
                continue
            before = tensor_hash(x)
            summed = T.zeros_like(x)
            count = T.zeros(len(x), dtype=T.int32)
            weight_sum = T.zeros(len(x), dtype=T.float32)
            for bi, group in enumerate(batches):
                values = predict_safe(pipe, model, group, x, bank, t, params)
                reduce_predictions(summed, count, group, values, weight_sum)
                print(
                    "BATCH",
                    stage,
                    step + 1,
                    bi + 1,
                    "/",
                    len(batches),
                    "blocks",
                    len(group),
                    flush=True,
                )
            assert tensor_hash(x) == before, (
                "global state changed before the synchronization barrier"
            )
            assert T.equal(count, data["counts"]) and (count > 0).all()
            T.testing.assert_close(weight_sum, data["depth_weight_sums"])
            assert (weight_sum > 0).all()
            v = summed / weight_sum[:, None]
            x = x - (t - tn) * v
            assert T.isfinite(x).all()
            common.save(
                checkpoint,
                coords=data["coords"],
                features=x,
                velocity=v,
                depth_blend="voxel_center_triangle_v1",
                t=t,
                t_next=tn,
                input_hash=before,
            )
            common.js(
                out / f"step_{step:02d}.json",
                dict(
                    stage=stage,
                    step=step,
                    t=t,
                    t_next=tn,
                    input_hash=before,
                    output_hash=tensor_hash(x),
                    global_points=len(x),
                    uncovered=0,
                    frozen_global_state=True,
                    winner_depth_counts_exact=True,
                    depth_blend="voxel_center_triangle_v1",
                    model_batches=len(batches),
                ),
            )
            print("GLOBAL SYNC", stage, step + 1, "/12", len(x), flush=True)
    finally:
        model.cpu()
        common.empty_cuda()
    common.save(
        out / ("smoke_endpoint.pt" if smoke else "endpoint.pt"),
        coords=data["coords"],
        features=x,
        normalized=True,
    )
    return x


@T.no_grad()
def upsample(
    pipe,
    coords,
    features,
    out,
    input_description="complete synchronized C128, original global indices",
):
    target = out / "global_C256.pt"
    if target.exists():
        saved = common.load_payload(target)
        assert saved["input_hash"] == tensor_hash(features)
        return saved["coords"]
    decoder = pipe.models["shape_slat_decoder"]
    decoder.to(pipe.device)
    decoder.low_vram = True
    try:
        raw = common.denormalize_shape(pipe, features)
        candidates = (
            decoder.upsample(
                SparseTensor(raw.to(pipe.device), coords.to(pipe.device)),
                upsample_times=1,
            )
            .cpu()
            .int()
        )
        result = candidates.unique(dim=0)
    finally:
        decoder.cpu()
        decoder.low_vram = False
    assert ((result[:, 1:] >= 0) & (result[:, 1:] < 256)).all()
    common.save(target, coords=result, input_hash=tensor_hash(features))
    common.js(
        out / "contract.json",
        dict(
            calls=1,
            upsample_times=1,
            input_points=len(coords),
            output_points=len(result),
            input=input_description,
        ),
    )
    del raw, candidates
    common.empty_cuda()
    return result
