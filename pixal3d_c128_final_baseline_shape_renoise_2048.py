#!/usr/bin/env python3
"""Fixed baseline C128 shape re-noise: conditional, unconditional and visibility gated."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from PIL import Image, ImageDraw, ImageFont

import pixal3d.models as models
import pixal3d_c128_baseline_endpoint_renoise_uncond_2048 as base
import pixal3d_global_c256_cube_owner_flow_singleview as cubes
from inference import MODEL_PATH, IMAGE_COND_CONFIGS, build_image_cond_model
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.pipelines import samplers
from pixal3d.modules.sparse import SparseTensor
from pixal3d.renderers import MeshRenderer

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'outputs/c128_baseline_endpoint_renoise_uncond_2048_cuda4'
BASELINE = ROOT / base.DEFAULT_BASELINE
VISIBILITY = ROOT / 'outputs/c128_baseline_visibility/visibility.pt'
DEFAULT_OUTPUT = ROOT / 'outputs/c128_final_baseline_shape_renoise_2048_cuda4'
FORMAT = 'c128_final_baseline_shape_renoise_2048_v1'
MODES = ('conditional', 'unconditional', 'visibility_gated')
STEPS = 12


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def init_shape_pipeline(model_path, device):
    config = json.loads((model_path / 'pipeline.json').read_text())['args']
    loaded = {}
    for name in ('shape_slat_flow_model_1024', 'shape_slat_decoder'):
        print(f'[model] loading {name}', flush=True)
        loaded[name] = models.from_pretrained(str(model_path / config['models'][name])).eval()
    sampler_config = config['shape_slat_sampler']
    pipeline = Pixal3DImageTo3DPipeline(
        models=loaded,
        shape_slat_sampler=getattr(samplers, sampler_config['name'])(**sampler_config['args']),
        shape_slat_sampler_params=sampler_config['params'],
        shape_slat_normalization=config['shape_slat_normalization'],
        low_vram=True,
    )
    pipeline._device = device
    return pipeline


@torch.no_grad()
def shape_condition(pipeline, coords, records, camera, output):
    path = output / 'conditions/full_image_shape_c128.pt'
    fingerprint = hashlib.sha256((BASELINE / 'canonical_1024.png').read_bytes()).hexdigest()
    if path.is_file():
        payload = load(path)
        assert torch.equal(payload['coords'], coords)
        assert payload['image_sha256'] == fingerprint and payload['camera'] == camera
    else:
        print('[condition] shape-specific full-image C128 projection', flush=True)
        extractor = build_image_cond_model(IMAGE_COND_CONFIGS['shape_1024'])
        with Image.open(BASELINE / 'canonical_1024.png') as source:
            image = source.convert('RGB')
        assert image.size == (1024, 1024)
        cond = pipeline.get_proj_cond_shape(
            extractor, [image], coords.to(pipeline.device),
            camera_angle_x=float(camera['camera_angle_x']), distance=float(camera['distance']),
            mesh_scale=float(camera.get('mesh_scale', 1.)), grid_resolution_override=128,
        )['cond']
        payload = {'format': FORMAT, 'coords': coords, 'global': cond['global'].float().cpu(),
                   'proj': cond['proj'].feats.float().cpu(), 'image_sha256': fingerprint,
                   'camera': camera, 'extractor': IMAGE_COND_CONFIGS['shape_1024']}
        base.atomic_save(path, payload)
        extractor.cpu()
        del extractor, cond
        image.close()
        base.empty_cuda()
    assert payload['proj'].shape[0] == len(coords)
    return {'cubes': {int(rec['cube_id']): {
        'global_row_ids': rec['global_row_ids'], 'global': payload['global'],
        'proj': payload['proj'].index_select(0, rec['global_row_ids']).contiguous(),
    } for rec in records}}, base.tensor_hash(payload['proj'])


def route_condition(full, routing, mode):
    by_id = {r['cube_id']: r['conditional'] for r in routing}
    result = {'cubes': {}}
    for cube_id, cond in full['cubes'].items():
        enabled = mode == 'conditional' or (mode == 'visibility_gated' and by_id[cube_id])
        result['cubes'][cube_id] = {
            'global_row_ids': cond['global_row_ids'],
            'global': cond['global'] if enabled else torch.zeros_like(cond['global']),
            'proj': cond['proj'] if enabled else torch.zeros_like(cond['proj']),
        }
    return result


def make_grid(rows, mode, view, output):
    width, bar, gap = 1024, 128, 24
    canvas = Image.new('RGB', (4*width+5*gap, 3*(width+bar)+4*gap), (22,24,29))
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 35)
    small = ImageFont.truetype(font_path, 29)
    for i, row in enumerate(rows):
        x = gap + i%4*(width+gap); y = gap + i//4*(width+bar+gap)
        draw.text((x+12,y+10), f"n={row['n']:02d} | {mode} | {view}", fill='white', font=font)
        draw.text((x+12,y+62), f"Baseline shape + noise t={row['start_t']:.4f} | Flow {12-row['n']} steps", fill=(255,204,102), font=small)
        with Image.open(row[f'{view}_normal']) as image:
            assert image.size == (1024,1024)
            canvas.paste(image.convert('RGB'), (x,y+bar))
    canvas.save(output)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--model-path', type=Path, default=Path(MODEL_PATH))
    parser.add_argument('--physical-cuda', type=int, default=4)
    parser.add_argument('--noise-seed', type=int, default=44)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_cuda):
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the selected physical GPU')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    shape = load(SHARED / 'support/fixed_c128_shape_slat.pt')
    shape_norm = load(SHARED / 'support/fixed_c128_shape_normalized.pt')
    coords = shape['coords'].int()
    assert torch.equal(coords, shape_norm['coords'].int())
    reference_summary = json.loads((SHARED / 'summary.json').read_text())
    assert base.tensor_hash(shape['raw_features']) == reference_summary['fixed_shape_raw_sha256']
    assert base.tensor_hash(shape_norm['normalized_features']) == reference_summary['fixed_shape_normalized_sha256']
    vis = load(VISIBILITY)
    assert torch.equal(coords[:,1:], vis['coords'].int())
    vis_summary = json.loads(VISIBILITY.with_name('summary.json').read_text())
    endpoint_summary = json.loads(Path(reference_summary['baseline_endpoint_summary']).read_text())
    assert Path(vis_summary['baseline_mesh']).resolve() == Path(endpoint_summary['shared_geometry']).resolve()
    assert (SHARED / 'c2048/flexible_dual_grid.pt').is_file()
    records, owner = base.build_records(coords)
    routing = []
    for rec in records:
        rows = rec['global_row_ids']; fraction = float(vis['visible'][rows].float().mean())
        routing.append({'cube_id': int(rec['cube_id']), 'start': list(rec['start']),
                        'tokens': len(rows), 'visible_tokens': int(vis['visible'][rows].sum()),
                        'visible_fraction': fraction, 'conditional': fraction > .1})
    print('[routing]', json.dumps(routing), flush=True)
    pipeline = init_shape_pipeline(args.model_path, device)
    clean = cubes.normalize(shape['raw_features'].float(), pipeline.shape_slat_normalization)
    assert torch.equal(clean, shape_norm['normalized_features'].float())
    epsilon = torch.randn(clean.shape, generator=torch.Generator(device='cpu').manual_seed(args.noise_seed))
    clean_hash, noise_hash = base.tensor_hash(clean), base.tensor_hash(epsilon)
    camera = json.loads((BASELINE / 'global_camera.json').read_text())
    full, cond_hash = shape_condition(pipeline, coords, records, camera, output)
    sampler = pipeline.shape_slat_sampler
    schedule = sampler.timestep_schedule(STEPS, 1.)
    sigma = float(sampler.sigma_min)
    manifest = {'format': FORMAT, 'status': 'running', 'physical_cuda': args.physical_cuda,
                'semantics': 'For every n, re-noise the SAME final baseline mesh -> C2048 voxelization -> encoded C128 shape; no prefix model updates; then local C64 shape flow steps n+1..12.',
                'baseline_mesh': endpoint_summary['shared_geometry'],
                'shape_source': str(SHARED / 'support/fixed_c128_shape_slat.pt'),
                'visibility_source': str(VISIBILITY), 'visibility_definition': vis_summary['inheritance'],
                'shape_sha256': clean_hash, 'noise_sha256': noise_hash, 'noise_seed': args.noise_seed,
                'shape_condition_sha256': cond_hash, 'schedule': schedule, 'sigma_min': sigma,
                'context':64, 'stride':64, 'effective_batch':8, 'routing':routing,
                'threshold':.1, 'comparison':'>', 'guidance_strength':1., 'guidance_rescale':0.,
                'support_tokens':len(coords), 'support_fixed_during_flow':True, 'decode_resolution':2048,
                'render_resolution':1024, 'normal_convention':'Repository MeshRenderer: flat triangle normals in each view camera coordinates, existing two-sided orientation rule, RGB=(normal+1)/2; neutral gray background.',
                'render_ssaa':2, 'texture':False, 'visual_metrics':False}
    base.atomic_json(output / 'run_manifest.json', manifest)
    base.atomic_save(output / 'fixed_shape_normalized.pt', {'coords':coords, 'normalized_features':clean})
    base.atomic_save(output / 'fixed_noise.pt', {'coords':coords, 'epsilon':epsilon, 'sha256':noise_hash})
    clean_gpu, noise_gpu = clean.to(device), epsilon.to(device)
    for n in range(1,13):
        t = schedule[n]; weight = sigma+(1-sigma)*t
        state = ((1-t)*clean_gpu+weight*noise_gpu).cpu()
        base.atomic_save(output / f'noised_states/n_{n:02d}.pt', {
            'coords':coords, 'normalized_features':state, 't':t, 'clean_weight':1-t, 'noise_weight':weight,
            'shape_sha256':clean_hash, 'noise_sha256':noise_hash})
    del clean_gpu, noise_gpu
    groups = cubes.pack_groups(records, 44, 10_000_000, require_owned=True)
    assert len(groups)==1 and len(groups[0])==8
    params = {'guidance_strength':1., 'guidance_rescale':0., 'guidance_interval':(0.,1.)}
    model = pipeline.models['shape_slat_flow_model_1024'].to(device).eval()
    assert model.in_channels == clean.shape[1]
    all_results = {}
    for mode in MODES:
        cond = route_condition(full, routing, mode)
        results = []
        for n in range(1,13):
            path = output / mode / f'flow/n_{n:02d}.pt'
            if path.is_file():
                saved = load(path); row = saved['record']
                assert saved['format']==FORMAT and torch.equal(saved['coords'],coords)
                assert row['shape_sha256']==clean_hash and row['noise_sha256']==noise_hash
                assert row['shape_condition_sha256']==cond_hash and row['routing']==routing and row['mode']==mode
                results.append(row); print(f'[flow {mode} n={n:02d}] cache hit', flush=True); continue
            state = load(output / f'noised_states/n_{n:02d}.pt')['normalized_features']
            trace = []
            print(f'[flow {mode} n={n:02d}] {12-n} steps', flush=True)
            for step in range(n,12):
                t, tn = schedule[step:step+2]; start = time.perf_counter(); proposals=[]
                for group in groups:
                    values, _ = cubes._one_prediction(group, state, cond, sampler, model, params, t, tn, device)
                    proposals.extend((int(rec['cube_id']),rec['global_row_ids'],v) for rec,v in zip(group,values))
                velocity = cubes.validate_owner_scatter(owner, proposals, clean.shape[1])
                state = cubes.jacobi_update(state, velocity, t, tn)
                assert torch.isfinite(state).all()
                trace.append({'step':step+1,'t':t,'t_next':tn,'seconds':time.perf_counter()-start})
            row = {'mode':mode, 'n':n, 'start_t':schedule[n], 'suffix_steps':12-n, 'steps':trace,
                   'shape_sha256':clean_hash, 'noise_sha256':noise_hash, 'shape_condition_sha256':cond_hash,
                   'routing':routing, 'state_sha256':base.tensor_hash(state), 'flow_path':str(path)}
            base.atomic_save(path, {'format':FORMAT, 'coords':coords, 'normalized_features':state, 'record':row})
            results.append(row)
        all_results[mode] = results
    model.cpu(); del model, full, cond; base.empty_cuda()
    # Flow batches are independent: gated rows must exactly reproduce their corresponding route.
    conditional_rows = torch.zeros(len(coords),dtype=torch.bool)
    for rec,route in zip(records,routing):
        if route['conditional']: conditional_rows[rec['global_row_ids']] = True
    audit = []
    for n in range(1,13):
        states = {m:load(output/m/f'flow/n_{n:02d}.pt')['normalized_features'] for m in MODES}
        c_err = float((states['conditional']-states['visibility_gated'])[conditional_rows].abs().max())
        u_err = float((states['unconditional']-states['visibility_gated'])[~conditional_rows].abs().max())
        assert c_err==0 and u_err==0
        if n==12:
            initial = load(output/'noised_states/n_12.pt')['normalized_features']
            assert all(torch.equal(s,initial) for s in states.values())
        audit.append({'n':n,'conditional_rows_max_abs':c_err,'unconditional_rows_max_abs':u_err})
    base.atomic_json(output/'flow_audit.json', {'status':'passed','rows':audit})
    print('[audit] all gated-route latent rows match pure routes exactly', flush=True)
    views, intrinsics = base.camera_views(camera)
    renderer = MeshRenderer({'resolution':1024,'near':max(.01,float(camera['distance'])-2),
                             'far':float(camera['distance'])+10,'ssaa':2,'chunk_size':2_000_000,
                             'antialias':False}, device=str(device))
    for mode, rows in all_results.items():
        for row in rows:
            n=row['n']; directory=output/mode/f'variants/n_{n:02d}'
            directory.mkdir(parents=True,exist_ok=True)
            render_paths={view:directory/f'{view}_normal_camera.png' for view in views}
            geometry_meta=directory/'geometry.json'
            if n==12 and mode!='conditional':
                source=output/'conditional/variants/n_12'
                for path in render_paths.values():shutil.copy2(source/path.name,path)
                shutil.copy2(source/'geometry.json',geometry_meta)
            if not all(p.is_file() for p in render_paths.values()) or not geometry_meta.is_file():
                print(f'[decode/render {mode} n={n:02d}]',flush=True)
                state=load(Path(row['flow_path']))['normalized_features']
                raw=cubes.denormalize(state,pipeline.shape_slat_normalization)
                slat=SparseTensor(raw.to(device),coords.to(device))
                meshes,subs=pipeline.decode_shape_slat(slat,2048)
                mesh=meshes[0]
                assert mesh.vertices.numel() and mesh.faces.numel() and torch.isfinite(mesh.vertices).all()
                meta={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'state_sha256':row['state_sha256']}
                del meshes,subs,slat; base.empty_cuda()
                for view,extrinsics in views.items():
                    rendered=renderer.render(mesh,extrinsics.to(device),intrinsics.to(device),return_types=['normal','mask'])
                    normal=rendered['normal'].permute(1,2,0).contiguous()
                    base.save_image(normal.cpu().numpy(),render_paths[view])
                    del rendered,normal
                base.atomic_json(geometry_meta,meta)
                del mesh; base.empty_cuda()
            meta=json.loads(geometry_meta.read_text()); assert meta['state_sha256']==row['state_sha256']
            row.update({f'{view}_normal':str(path) for view,path in render_paths.items()})
            row.update(vertices=meta['vertices'],faces=meta['faces'])
            base.atomic_json(directory/'record.json',row)
        for view in views:
            make_grid(rows,mode,view,output/mode/f'{view}_normals_3x4.png')
        base.atomic_json(output/mode/'summary.json',{**manifest,'status':'complete','mode':mode,'results':rows})
        print(f'[complete] {mode}',flush=True)
    base.atomic_json(output/'summary.json',{**manifest,'status':'complete','results':all_results})
    base.atomic_json(output/'run_manifest.json',{**manifest,'status':'complete'})
    shutil.copy2(__file__,output/Path(__file__).name)
    print('[done]',output,flush=True)


if __name__=='__main__':
    main()
