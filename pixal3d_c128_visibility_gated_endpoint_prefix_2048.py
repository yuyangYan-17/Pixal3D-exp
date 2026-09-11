#!/usr/bin/env python3
"""Uniform endpoint-guided prefixes followed by visibility-gated C64 texture flow."""
from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','4')
import json, time, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import pixal3d_c128_baseline_endpoint_renoise_uncond_2048 as base
import pixal3d_c128_baseline_endpoint_renoise_cond_2048 as cond_base
import pixal3d_global_c256_cube_owner_flow_singleview as cubes
from inference import init_pipeline, MODEL_PATH

ROOT=Path(__file__).resolve().parent
SHARED=ROOT/'outputs/c128_baseline_endpoint_renoise_uncond_2048_cuda4'
ENDPOINTS=ROOT/base.DEFAULT_ENDPOINTS
BASELINE=ROOT/base.DEFAULT_BASELINE
VISIBILITY=ROOT/'outputs/c128_baseline_visibility/visibility.pt'
OUT=ROOT/'outputs/c128_visibility_gated_endpoint_prefix_2048_cuda4'
FORMAT='c128_visibility_gated_sequential_endpoint_prefix_v1'


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def validate_inputs():
    summary=json.loads((ENDPOINTS/'summary.json').read_text())
    assert summary['status']=='complete' and summary['texture_steps']==12 and summary['texture_rescale_t']==1
    assert np.allclose(summary['texture_schedule'],np.linspace(1,0,13))
    shape=load(SHARED/'support/fixed_c128_shape_slat.pt')
    norm=load(SHARED/'support/fixed_c128_shape_normalized.pt')
    coords=shape['coords'].int()
    assert torch.equal(coords,norm['coords'].int())
    vis=load(VISIBILITY)
    assert torch.equal(coords[:,1:],vis['coords'].int())
    visibility_summary=json.loads(VISIBILITY.with_name('summary.json').read_text())
    assert Path(visibility_summary['baseline_mesh']).resolve()==Path(summary['shared_geometry']).resolve()
    endpoints=[]
    for k in range(1,13):
        p=load(SHARED/f'encoded_c128_endpoints/step_{k:02d}.pt')
        assert torch.equal(coords,p['coords'].int())
        assert base.tensor_hash(p['normalized_features'])==p['normalized_sha256']
        assert abs(p['t']-summary['texture_schedule'][k-1])<1e-8
        assert abs(p['t_next']-summary['texture_schedule'][k])<1e-8
        assert Path(p['source_ovoxel1024']).resolve()==(ENDPOINTS/f'ovoxel1024/texture_endpoint_step_{k:02d}.pt').resolve()
        assert Path(p['source_ovoxel1024']).is_file()
        assert (ENDPOINTS/f'texture_endpoint_latents/step_{k:02d}.pt').is_file()
        endpoints.append(p['normalized_features'].float())
    assert (SHARED/'c2048/flexible_dual_grid.pt').is_file()
    noise=load(SHARED/'flow/fixed_noise.pt')
    assert torch.equal(coords,noise['coords'].int())
    return shape,norm,vis,endpoints,noise['epsilon'].float(),summary


def make_render_grid(records, path, view, metric):
    from PIL import ImageDraw, ImageFont
    width, label, gap = 1024, 180, 24
    canvas = Image.new('RGB', (4*width+5*gap, 3*(width+label)+4*gap), (18,20,25))
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 37)
    small = ImageFont.truetype(font_path, 31)
    for i, row in enumerate(records):
        x=gap+(i%4)*(width+gap); y=gap+(i//4)*(width+label+gap)
        title='Input view / vs GT' if view=='front' else 'Back view / vs baseline'
        draw.text((x+12,y+10), f"n={row['n']:02d} | {title}", font=font, fill='white')
        draw.text((x+12,y+65), f"Endpoint-guided {row['n']} | visibility-gated {12-row['n']}",font=small,fill=(255,204,102))
        m=row[metric]
        draw.text((x+12,y+118), f"FG PSNR {m['foreground_psnr_db']:.4f} dB | SSIM {m['foreground_ssim']:.4f}",font=small,fill=(102,204,255))
        with Image.open(row[f'{view}_render']) as image:
            assert image.size==(1024,1024)
            canvas.paste(image.convert('RGB'),(x,y+label))
    canvas.save(path)


@torch.no_grad()
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='4':raise RuntimeError('This experiment requires physical CUDA 4')
    torch.cuda.set_device(0);device=torch.device('cuda:0');OUT.mkdir(parents=True,exist_ok=True)
    shape,norm,vis,endpoints,noise,baseline_summary=validate_inputs()
    coords=shape['coords'].int();shape_norm=norm['normalized_features'].float()
    shape_hash=base.tensor_hash(shape['raw_features'])
    records,owner=base.build_records(coords)
    routing=[]
    for record in records:
        rows=record['global_row_ids'].long();fraction=float(vis['visible'][rows].float().mean())
        routing.append({'cube_id':int(record['cube_id']),'start':list(record['start']),'tokens':len(rows),'visible_tokens':int(vis['visible'][rows].sum()),'visible_fraction':fraction,'conditional':fraction>.1})
    print('[routing]',json.dumps(routing),flush=True)
    manifest={'format':FORMAT,'physical_cuda':4,'status':'running','threshold':.1,'threshold_comparison':'>','routing':routing,'prefix':'sequential Euler updates using encoded endpoint k at step k, starting from shared fixed epsilon','source_steps_1_to_7':'validated completed baseline/encoder caches reused','baseline_summary':str(ENDPOINTS/'summary.json'),'encoded_endpoints':str(SHARED/'encoded_c128_endpoints'),'visibility':str(VISIBILITY),'shape_sha256':shape_hash,'context':64,'stride':64,'configured_flow_batch_size':44,'effective_flow_batch_size':8}
    base.atomic_json(OUT/'run_manifest.json',manifest)
    pipeline=init_pipeline(MODEL_PATH,device=str(device),low_vram=True)
    camera=json.loads((BASELINE/'global_camera.json').read_text())
    full=cond_base.full_image_texture_condition(pipeline,BASELINE/'canonical_1024.png',camera,coords,records,ROOT/'outputs/c128_baseline_endpoint_renoise_cond_2048_cuda4',True)
    for r in routing:
        if not r['conditional']:
            c=full['cubes'][r['cube_id']]
            c['global']=torch.zeros_like(c['global']);c['proj']=torch.zeros_like(c['proj'])
            assert not c['global'].any() and not c['proj'].any()
    manifest['condition_fingerprint_sha256']=base.tensor_hash(torch.cat([full['cubes'][int(r['cube_id'])]['proj'] for r in records]))
    sampler=pipeline.tex_slat_sampler;schedule=sampler.timestep_schedule(12,1.)
    groups=cubes.pack_groups(records,44,10_000_000,require_owned=True)
    assert len(groups)==1 and len(groups[0])==8
    params={'steps':12,'rescale_t':1.,'guidance_strength':1.,'guidance_rescale':0.,'guidance_interval':(0.,1.)}
    # With guidance=1, model sees this per-cube cond directly. Zero global AND
    # projected tokens implement the same image-unconditional branch; shape stays.
    model=pipeline.models['tex_slat_flow_model_1024'].eval().to(device)
    prefix_states=[];prefix_trace=[];state=noise.to(device)
    for step in range(12):
        t,tn=map(float,schedule[step:step+2]);endpoint=endpoints[step].to(device)
        velocity=sampler._xstart_to_pred(state,t,endpoint)
        # Check inversion of the sampler's endpoint/velocity parameterization.
        reconstructed=sampler._pred_to_xstart(state,t,velocity)
        assert torch.allclose(reconstructed,endpoint,atol=2e-5,rtol=2e-5)
        state=state+(tn-t)*velocity
        assert torch.isfinite(state).all()
        prefix_states.append(state.cpu())
        prefix_trace.append({'step':step+1,'t':t,'t_next':tn,'route':'encoded_baseline_endpoint','endpoint_sha256':base.tensor_hash(endpoints[step])})
        base.atomic_save(OUT/f'guided_prefix_states/step_{step+1:02d}.pt',{'coords':coords,'normalized_features':state.cpu(),'step':step+1})
    del state,velocity,endpoint,reconstructed
    results=[]
    for n in range(1,13):
        path=OUT/f'flow/prefix_{n:02d}/final_texture_normalized.pt'
        if path.is_file():
            saved=load(path)
            assert saved['format']==FORMAT and torch.equal(saved['coords'],coords)
            results.append(saved['record']);continue
        state=prefix_states[n-1].clone();trace=list(prefix_trace[:n])
        print(f'[variant {n:02d}] guided={n}; gated suffix={12-n}',flush=True)
        for step in range(n,12):
            start=time.perf_counter();t,tn=map(float,schedule[step:step+2]);proposals=[]
            for group in groups:
                values,timing=cubes._one_prediction(group,state,full,sampler,model,params,t,tn,device,shape_norm)
                proposals.extend((int(r['cube_id']),r['global_row_ids'],value) for r,value in zip(group,values))
            velocity=cubes.validate_owner_scatter(owner,proposals,state.shape[1]);state=cubes.jacobi_update(state,velocity,t,tn)
            assert torch.isfinite(state).all()
            trace.append({'step':step+1,'t':t,'t_next':tn,'route':'visibility_gated_image_condition','conditional_cubes':[r['cube_id'] for r in routing if r['conditional']],'unconditional_cubes':[r['cube_id'] for r in routing if not r['conditional']],'seconds':time.perf_counter()-start})
            print(f'  step {step+1:02d} {time.perf_counter()-start:.2f}s',flush=True)
        record={'n':n,'guided_prefix_steps':n,'suffix_steps':12-n,'suffix_label':'vis-gated','shape_c128_fixed':True,'steps':trace,'path':str(path),'state_sha256':base.tensor_hash(state)}
        base.atomic_save(path,{'format':FORMAT,'coords':coords,'normalized_features':state,'record':record})
        results.append(record)
    model.cpu();del model; base.empty_cuda()
    assert base.tensor_hash(shape['raw_features'])==shape_hash
    evaluated=base.decode_render_evaluate(pipeline,shape['raw_features'].float(),coords,results,ENDPOINTS,BASELINE,camera,OUT,device,1024,4_000_000,True)
    # Recompute metrics from saved RGB PNGs, ensuring repeatability on resume.
    gt=np.asarray(Image.open(BASELINE/'canonical_1024.png').convert('RGB'),np.float32)/255
    mask=np.asarray(Image.open(BASELINE/'raw_ovoxel_render/alpha.png').convert('L'))>127
    for r in evaluated:
        pred=np.asarray(Image.open(r['front_render']).convert('RGB'),np.float32)/255
        r['front_vs_gt']=base.compare_metrics(gt,pred,mask)
        base.atomic_json(OUT/f"variants/n_{r['n']:02d}/metrics.json",r)
    for view,metric,title in [('front','front_vs_gt','input vs GT'),('back','back_vs_baseline','back vs baseline')]:
        make_render_grid(evaluated,OUT/f'{view}_renders_3x4.png',view,metric)
    base.make_metric_plot(evaluated,OUT/'foreground_metrics.png')
    baseline_pred=np.asarray(Image.open(OUT/'baseline1024_reference/front/render.png').convert('RGB'),np.float32)/255
    summary={**manifest,'status':'complete','results':evaluated,'baseline_front_vs_gt':base.compare_metrics(gt,baseline_pred,mask),'metrics_definition':{'reference':str(BASELINE/'canonical_1024.png'),'foreground_mask':str(BASELINE/'raw_ovoxel_render/alpha.png'),'mask_rule':'fixed baseline alpha > 127; same pixels for all variants','psnr':'RGB MSE over foreground pixels; data_range=1; saved 8-bit RGB PNG','ssim':'skimage 7x7 uniform-window channelwise SSIM map, averaged over fixed foreground pixels','render':'native 2048 decoded geometry/material; 1024x1024 front/back studio PBR'},'schedule':schedule}
    base.atomic_json(OUT/'summary.json',summary)
    with (OUT/'foreground_metrics.csv').open('w') as fp:
        writer=csv.writer(fp);writer.writerow(['n','foreground_psnr_db','foreground_ssim','foreground_pixels'])
        for r in evaluated:writer.writerow([r['n'],r['front_vs_gt']['foreground_psnr_db'],r['front_vs_gt']['foreground_ssim'],r['front_vs_gt']['foreground_pixels']])
    base.atomic_json(OUT/'run_manifest.json',{**manifest,'status':'complete'})
    print('[done]',OUT,flush=True)

if __name__=='__main__':main()
