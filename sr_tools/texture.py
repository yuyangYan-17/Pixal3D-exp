"""Material flow on the final shape support, with the same owner/depth reduction."""

import gc
import torch as T
from . import shape as sync, common
from .common import SparseTensor


def validate_geometry(shape, data):
    assert shape["normalized"] and T.equal(shape["coords"], data["coords"])
    assert T.isfinite(shape["features"]).all()
    assert len(data["tiles"]) > 0 and len(data["blocks"]) > 0
    counts = T.zeros(len(data["coords"]), dtype=T.int32)
    for tile in data["tiles"]:
        restored = tile["coords"].clone()
        restored[:, 1:] += tile["index_offset"]
        assert T.equal(restored, data["coords"][tile["rows"]])
    for b in data["blocks"]:
        tile = data["tiles"][b["tile_id"]]
        assert T.equal(b["global_ids"], tile["rows"][b["rows"]])
        restored = b["coords"].clone()
        restored[:, 3] += b["z_start"]
        assert T.equal(restored, tile["coords"][b["rows"]])
        assert T.equal(b["owned"], data["owner"][b["global_ids"]] == b["tile_id"])
        ids = b["global_ids"][b["owned"]]
        counts.index_add_(0, ids, T.ones(len(ids), dtype=T.int32))
    assert T.equal(counts, data["counts"]) and (counts > 0).all()


@T.no_grad()
def conditions(pipe, source, out, data, cam):
    bank = {}
    for tile in data["tiles"]:
        if not len(tile["rows"]):
            continue
        dest = out / "tiles" / f"{tile['tile_id']:02d}"
        target = dest / "conditions.pt"
        if target.exists():
            cached = common.load_payload(target)
            assert cached["extractor"] == "image_cond_model_tex_1024"
            assert T.equal(cached["coords"], tile["coords"])
            bank[tile["tile_id"]] = {k: cached[k] for k in ("global_", "proj")}
            continue
        image = common.Image.open(
            source / "shape1024/tiles" / f"{tile['tile_id']:02d}" / "input_1024.png"
        ).convert("RGB")
        ext = pipe.image_cond_model_tex_1024
        with sync.projection_adapter(ext, tile, cam, 1024):
            cond = pipe.get_proj_cond_shape(
                ext,
                [image],
                tile["coords"].to(pipe.device),
                camera_angle_x=cam["camera_angle_x"],
                distance=cam["distance"],
                mesh_scale=1.0,
                grid_resolution_override=64,
            )["cond"]
        bank[tile["tile_id"]] = dict(
            global_=cond["global"].cpu(), proj=cond["proj"].feats.cpu()
        )
        assert len(bank[tile["tile_id"]]["proj"]) == len(tile["rows"])
        common.save(
            target,
            coords=tile["coords"],
            extractor="image_cond_model_tex_1024",
            **bank[tile["tile_id"]],
        )
        del cond
        common.empty_cuda()
        print("TEXTURE CONDITION", tile["tile_id"], len(tile["rows"]), flush=True)
    return bank


@T.no_grad()
def predict(
    pipe,
    model,
    group,
    x,
    shape,
    bank,
    t,
    params,
    unconditional=False,
    global_only=False,
):
    coords = []
    states = []
    shapes = []
    projs = []
    globals_ = []
    lengths = []
    for bid, b in enumerate(group):
        c = b["coords"].clone()
        c[:, 0] = bid
        coords.append(c)
        lengths.append(len(c))
        states.append(x[b["global_ids"]])
        shapes.append(shape[b["global_ids"]])
        con = bank[b["tile_id"]]
        proj = con["proj"][b["rows"]]
        # A visibility-aware diagnostic may keep the sparse context rows for
        # receptive-field continuity while removing their invalid local image
        # evidence.  Ownership is still reduced by the caller's ``owned``
        # mask; this only changes what the conditional branch sees as context.
        condition_mask = b.get("condition_mask")
        if condition_mask is not None:
            proj = proj.clone()
            condition_weight = float(b.get("condition_weight", 0.0))
            proj[~condition_mask] *= condition_weight
        projs.append(proj)
        globals_.append(con["global_"])
    c = T.cat(coords).to(pipe.device)
    xt = SparseTensor(T.cat(states).to(pipe.device), c)
    concat = SparseTensor(T.cat(shapes).to(pipe.device), c)
    proj = SparseTensor(T.cat(projs).to(pipe.device), c)
    glob = T.cat(globals_).to(pipe.device)
    with sync.batch_stable_linear(model):
        if unconditional or global_only:
            # Bypass CFG and its interval wrapper; retain the geometric concat.
            from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler

            pred = FlowEulerSampler._inference_model(
                pipe.tex_slat_sampler,
                model,
                xt,
                t,
                cond={
                    "global": T.zeros_like(glob) if unconditional else glob,
                    "proj": proj.replace(T.zeros_like(proj.feats)),
                },
                concat_cond=concat,
            )
        else:
            pred = pipe.tex_slat_sampler._inference_model(
                model,
                xt,
                t,
                concat_cond=concat,
                cond={"global": glob, "proj": proj},
                neg_cond={
                    "global": T.zeros_like(glob),
                    "proj": proj.replace(T.zeros_like(proj.feats)),
                },
                **params,
            )
    assert T.equal(pred.coords, c) and T.isfinite(pred.feats).all()
    return list(pred.feats.float().cpu().split(lengths))


@T.no_grad()
def predict_safe(
    pipe,
    model,
    group,
    x,
    shape,
    bank,
    t,
    params,
    unconditional=False,
    global_only=False,
):
    try:
        return predict(
            pipe,
            model,
            group,
            x,
            shape,
            bank,
            t,
            params,
            unconditional=unconditional,
            global_only=global_only,
        )
    except T.cuda.OutOfMemoryError:
        if len(group) == 1:
            raise
    gc.collect()
    common.empty_cuda()
    mid = len(group) // 2
    print("TEXTURE OOM batch split", len(group), flush=True)
    return predict_safe(
        pipe,
        model,
        group[:mid],
        x,
        shape,
        bank,
        t,
        params,
        unconditional=unconditional,
        global_only=global_only,
    ) + predict_safe(
        pipe,
        model,
        group[mid:],
        x,
        shape,
        bank,
        t,
        params,
        unconditional=unconditional,
        global_only=global_only,
    )


@T.no_grad()
def flow(pipe, data, shape, bank, out, args):
    model = pipe.models["tex_slat_flow_model_1024"]
    sampler = pipe.tex_slat_sampler
    params = dict(pipe.tex_slat_sampler_params)
    params.pop("steps", None)
    times = sampler.timestep_schedule(12, params.pop("rescale_t", 1.0))
    channels = model.in_channels - shape.shape[1]
    assert channels > 0 and times[0] == 1.0
    x = T.randn(
        (len(data["coords"]), channels), generator=T.Generator().manual_seed(46)
    )
    common.save(out / "initial.pt", coords=data["coords"], noise=x, state=x, seed=46)
    shape_hash = sync.tensor_hash(shape)
    common.js(
        out / "sampler.json",
        dict(
            params=params,
            times=times,
            seed=46,
            initialization="one global Gaussian field at t=1",
            concat_cond="final normalized Shape1024, exact same global rows",
            shape_hash=shape_hash,
        ),
    )
    batches = list(sync.groups(data["blocks"], args.batch_size, args.max_tokens))
    model.to(pipe.device)
    try:
        if args.smoke:
            active = [b for b in data["blocks"] if len(b["rows"])]
            first = max(active, key=lambda b: len(b["rows"]))
            second = max(
                (b for b in active if b["tile_id"] != first["tile_id"]),
                key=lambda b: len(b["rows"]),
            )
            group = [first, second]
            errors = []
            for t in (1.0, 0.5):
                bat = T.cat(predict(pipe, model, group, x, shape, bank, t, params))
                serial = T.cat(
                    [
                        predict(pipe, model, [b], x, shape, bank, t, params)[0]
                        for b in group
                    ]
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
            if args.smoke and step >= 1:
                break
            target = out / f"step_{step:02d}.pt"
            if target.exists():
                cached = common.load_payload(target)
                assert T.equal(cached["coords"], data["coords"]) and cached[
                    "input_hash"
                ] == sync.tensor_hash(x)
                assert (
                    cached["shape_hash"] == shape_hash
                    and cached["t"] == t
                    and cached["t_next"] == tn
                )
                x = cached["features"]
                assert T.isfinite(x).all()
                continue
            before = sync.tensor_hash(x)
            summed = T.zeros_like(x)
            counts = T.zeros(len(x), dtype=T.int32)
            for bi, group in enumerate(batches):
                values = predict_safe(pipe, model, group, x, shape, bank, t, params)
                sync.reduce_predictions(summed, counts, group, values)
                print(
                    "TEXTURE BATCH",
                    step + 1,
                    bi + 1,
                    "/",
                    len(batches),
                    "blocks",
                    len(group),
                    flush=True,
                )
            assert before == sync.tensor_hash(x) and shape_hash == sync.tensor_hash(
                shape
            )
            assert T.equal(counts, data["counts"]) and (counts > 0).all()
            velocity = summed / counts[:, None]
            x = x - (t - tn) * velocity
            assert T.isfinite(x).all()
            common.save(
                target,
                coords=data["coords"],
                features=x,
                velocity=velocity,
                t=t,
                t_next=tn,
                input_hash=before,
                shape_hash=shape_hash,
            )
            common.js(
                out / f"step_{step:02d}.json",
                dict(
                    step=step,
                    t=t,
                    t_next=tn,
                    input_hash=before,
                    output_hash=sync.tensor_hash(x),
                    global_points=len(x),
                    uncovered=0,
                    frozen_global_state=True,
                    shape_unchanged=True,
                    winner_depth_counts_exact=True,
                    model_batches=len(batches),
                ),
            )
            print("TEXTURE GLOBAL SYNC", step + 1, "/12", len(x), flush=True)
    finally:
        model.cpu()
        common.empty_cuda()
    common.save(
        out / ("smoke_endpoint.pt" if args.smoke else "endpoint.pt"),
        coords=data["coords"],
        features=x,
        normalized=True,
    )
    return x
