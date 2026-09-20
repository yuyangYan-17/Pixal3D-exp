"""GPU checks for visibility, sparse queries, encoder alignment and actual flow routing."""

import json
from pathlib import Path
from types import SimpleNamespace
import torch
from . import common, visibility, texture_guidance as guidance, texture


@torch.no_grad()
def run(out, source=None):
    from pixal3d import models
    from pixal3d.pipelines import samplers
    from pixal3d.representations import Mesh
    from .baseline import ENCODER

    out.mkdir(parents=True, exist_ok=True)
    # Same projected triangle, front +Z fully occludes the back -Z triangle.
    front = torch.tensor([[-0.15, -0.15, 0.2], [0.15, -0.15, 0.2], [0, 0.15, 0.2]])
    back = front.clone()
    back[:, :2] *= (2.1 + 0.2) / (2.1 - 0.2)
    back[:, 2] = -0.2
    mesh = Mesh(
        torch.cat((front, back)),
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    )
    ids = visibility.visible_faces(
        mesh, dict(camera_angle_x=0.48, distance=2.1), resolution=128, face_chunk=1
    )
    assert ids.tolist() == [0], ids
    field = dict(
        coords=torch.tensor([[0, 2, 2, 2]], dtype=torch.int32), attrs=torch.ones(1, 6)
    )
    xyz = torch.tensor([[2.5, 2.5, 2.5], [10.5, 10.5, 10.5]]) / 1024 - 0.5
    queried, valid = guidance.query_field(field, xyz)
    assert valid.tolist() == [True, True]
    # A point outside the sparse support receives the nearest material rather
    # than the former zero-filled invalid sample.
    torch.testing.assert_close(queried, torch.tensor([[1.0] * 6, [1.0] * 6]))
    # Actual shape and texture encoders, same sparse support across a parent boundary.
    xyz = torch.cartesian_prod(
        torch.arange(2044, 2052), torch.arange(2044, 2048), torch.arange(2044, 2048)
    ).int()
    coords = torch.cat((torch.zeros(len(xyz), 1, dtype=torch.int32), xyz), 1).cuda()
    sparse = common.SparseTensor(
        torch.full((len(coords), 3), 0.5, device="cuda"), coords
    )
    shape_encoder = models.from_pretrained(str(ENCODER)).eval().cuda()
    encoded = shape_encoder(
        sparse, sparse.replace(torch.ones_like(sparse.feats)), sample_posterior=False
    )
    c256 = encoded.coords.cpu()
    del shape_encoder, encoded
    common.empty_cuda()
    tex_encoder = models.from_pretrained(str(guidance.TEX_ENCODER)).eval().cuda()
    encoded = tex_encoder(
        common.SparseTensor(torch.zeros(len(coords), 6, device="cuda"), coords),
        sample_posterior=False,
    )
    assert torch.equal(c256, encoded.coords.cpu())
    flag, _, _ = visibility.latent_visibility(xyz, xyz[:1], c256)
    assert flag.sum() == 1
    del tex_encoder, encoded, sparse, coords
    common.empty_cuda()
    # Load only the real texture network and use two real crop conditions.
    config = json.loads(
        Path("/home/nvme04/yyyan/download/model/Pixal3D/pipeline.json").read_text()
    )["args"]
    model = (
        models.from_pretrained(
            "/home/nvme04/yyyan/download/model/Pixal3D/"
            + config["models"]["tex_slat_flow_model_1024"]
        )
        .eval()
        .cuda()
    )
    spec = config["tex_slat_sampler"]
    sampler = getattr(samplers, spec["name"])(**spec["args"])
    pipe = SimpleNamespace(device=torch.device("cuda:0"), tex_slat_sampler=sampler)
    if source is None:
        source = Path("outputs/sr_0_img_z_tent")
    mapping = common.load_payload(source / "sr/shape1024/mapping.pt")
    original = common.load_payload(source / "sr/shape1024/endpoint.pt")["features"]
    active = [b for b in mapping["blocks"] if len(b["rows"]) >= 32]
    chosen = [
        active[0],
        next(b for b in active if b["tile_id"] != active[0]["tile_id"]),
    ]
    groups = []
    bank = {}
    shape_rows = []
    for i, b in enumerate(chosen):
        block = dict(b)
        block["rows"] = b["rows"][:32]
        block["coords"] = b["coords"][:32]
        block["global_ids"] = torch.arange(i * 32, (i + 1) * 32)
        block["conditional"] = i == 0
        block["owned"] = torch.ones(32, dtype=torch.bool)
        groups.append(block)
        shape_rows.append(original[b["global_ids"][:32]])
        p = common.load_payload(
            source / f"sr/texture/tiles/{b['tile_id']:02d}/conditions.pt"
        )
        bank[b["tile_id"]] = {k: p[k] for k in ["global_", "proj"]}
    shape = torch.cat(shape_rows)
    x = torch.randn(64, model.in_channels - shape.shape[1])
    guide = torch.randn_like(x)
    params = dict(spec["params"])
    params.pop("steps")
    times = sampler.timestep_schedule(12, params.pop("rescale_t"))
    checks = []
    for n in range(1, 5):
        for step in [n - 1, n]:
            t = times[step]
            result = guidance.routed_predictions(
                pipe, model, groups, x, shape, bank, t, params, step, n, guide
            )
            if step < n:
                expected = sampler._xstart_to_pred(x[32:], t, guide[32:])
                torch.testing.assert_close(result[1], expected)
                recovered = sampler._pred_to_xstart(x[32:], t, result[1])
                torch.testing.assert_close(recovered, guide[32:], atol=2e-6, rtol=2e-6)
            else:
                expected = texture.predict(
                    pipe,
                    model,
                    [groups[1]],
                    x,
                    shape,
                    bank,
                    t,
                    params,
                    unconditional=True,
                )[0]
                torch.testing.assert_close(result[1], expected)
            checks.append(
                dict(n=n, step=step + 1, phase="guide" if step < n else "unconditional")
            )
    # Image unconditional must be independent of both image-token inputs.
    a = texture.predict(
        pipe, model, [groups[1]], x, shape, bank, 0.5, params, unconditional=True
    )[0]
    changed = {
        k: {"global_": v["global_"] + 10, "proj": v["proj"] - 10}
        for k, v in bank.items()
    }
    b = texture.predict(
        pipe, model, [groups[1]], x, shape, changed, 0.5, params, unconditional=True
    )[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    common.js(
        out / "result.json",
        dict(
            status="PASS",
            visible_face_ids=ids.tolist(),
            query_coverage_probe=valid.tolist(),
            same_shape_texture_C256=True,
            routing=checks,
            image_unconditional_invariant=True,
        ),
    )
    print("GUIDANCE GPU SMOKE PASS", out, flush=True)
