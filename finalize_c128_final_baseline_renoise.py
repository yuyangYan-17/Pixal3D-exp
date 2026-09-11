"""Audit final-baseline re-noise outputs and prepare the experiment report."""
import json,math,hashlib
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/c128_final_baseline_renoise_visibility_gated_2048_cuda4'
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    s=json.loads((OUT/'summary.json').read_text());assert s['status']=='complete' and len(s['results'])==12
    e=load(Path(s['final_endpoint_path']))['normalized_features'].float()
    eps=load(OUT/'fixed_noise.pt')['epsilon'].float()
    max_error=0.;best=max(s['results'],key=lambda r:r['front_vs_gt']['foreground_psnr_db'])
    for r in s['results']:
        n=r['n'];p=load(OUT/f'noised_states/n_{n:02d}.pt')
        expected=(1-r['start_t'])*e+r['noise_weight']*eps
        err=float((p['normalized_features']-expected).abs().max());max_error=max(max_error,err)
        assert err<2e-6
        assert r['final_endpoint_sha256']==s['final_endpoint_sha256']
        assert len(r['steps'])==12-n
        assert [a['step'] for a in r['steps']]==list(range(n+1,13))
        state=load(Path(r['path']))['normalized_features']
        assert torch.isfinite(state).all()
        for view in ('front','back'):
            with Image.open(r[f'{view}_render']) as im:assert im.size==(1024,1024)
        if n==12:assert torch.equal(state,p['normalized_features'])
    # Side-by-side baseline comparison at the same render camera/light.
    tile=1024;bar=110
    canvas=Image.new('RGB',(3*tile,2*(tile+bar)),(18,20,25));draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',32)
    for row,view in enumerate(('front','back')):
        entries=[('Baseline 1024',OUT/f'baseline1024_reference/{view}/render.png'),(f"Best input PSNR: n={best['n']}",Path(best[f'{view}_render'])),('n=12 / no Flow',Path(s['results'][-1][f'{view}_render']))]
        for col,(title,path) in enumerate(entries):
            x=col*tile;y=row*(tile+bar)
            draw.text((x+20,y+20),f'{view.title()} | {title}',font=font,fill='white')
            with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((tile,tile),Image.Resampling.LANCZOS),(x,y+bar))
    canvas.save(OUT/'baseline_comparison_front_back.png')
    rows=['# Final baseline re-noise → C128 visibility-gated flow → 2048','', 'Completed on physical CUDA 4. Steps 1–7 reused the validated completed 1024 baseline (12 uniform texture steps) and official encoder caches. All variants use its final clean predicted material endpoint E12, never the time-dependent endpoint En. The saved original predicted material SLat is `outputs/baseline1024_uniform_texture_endpoints_c1282048_cuda4/texture_endpoint_latents/step_12.pt`.','',f"Encoded endpoint: `{s['final_endpoint_path']}`",'', 'The same fixed Gaussian epsilon is used for all n. No prefix model updates or sequential endpoint guidance are executed.','', '`t_n = 1 - n/12`','', '`x(t_n) = (1-t_n)*E_final + [sigma_min+(1-sigma_min)*t_n]*epsilon`','',f"sigma_min={s['sigma_min']}. At n=12, t=0, there are no Flow steps; the official convention retains the tiny sigma_min*epsilon residual.",'','Steps n+1…12: blocks 1/3/5/7 (>10% visible active cells) use full-image conditional tokens; blocks 0/2/4/6 use zero global/projected image tokens. Shape concat remains fixed. Context=64, stride=64, configured batch limit=44, effective batch=8.','', 'All variants decode to 2048 geometry/materials. Front and back renders are native 1024x1024. Front reference: original canonical 1024 input and fixed original baseline alpha. Back reference: original baseline mesh/material freshly rendered at 1024 from the back with the same camera/light. This is a baseline comparison, not back-view GT. Metrics use saved RGB PNG values and fixed foreground masks. SSIM uses a 7x7 window at the 1024 evaluation resolution.','', '| n | t_n | Flow steps | Front PSNR dB | Front SSIM | Back PSNR dB | Back SSIM |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['results']:
        a,b=r['front_vs_gt'],r['back_vs_baseline'];rows.append(f"| {r['n']} | {r['start_t']:.6f} | {r['suffix_steps']} | {a['foreground_psnr_db']:.4f} | {a['foreground_ssim']:.4f} | {b['foreground_psnr_db']:.4f} | {b['foreground_ssim']:.4f} |")
    b=s['baseline_front_vs_gt'];rows+=['',f"Baseline front vs input under the same evaluation: PSNR {b['foreground_psnr_db']:.4f} dB; SSIM {b['foreground_ssim']:.4f}.",'',f"Validation: 12 noised states independently recomputed; max error={max_error}; suffix step indices/counts correct; final states finite; n12 final equals its noised input; source endpoint hash shared; shape hash checked unchanged by runner; all 24 render images are native 1024x1024.",'', 'Outputs: `front_renders_3x4.png`, `back_renders_3x4.png`, `front_error_maps_3x4.png`, `back_error_maps_3x4.png`, `foreground_metrics.png`, `front_back_foreground_metrics.csv`, `baseline_comparison_front_back.png`, `summary.json`, `noised_states/`, `flow/`, `variants/`.']
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(rows)+'\n')
    print('Validated max noise reconstruction error:',max_error)
    print('Baseline:',s['baseline_front_vs_gt'])
    for r in s['results']:print(r['n'],r['front_vs_gt']['foreground_psnr_db'],r['front_vs_gt']['foreground_ssim'],r['back_vs_baseline']['foreground_psnr_db'],r['back_vs_baseline']['foreground_ssim'])
if __name__=='__main__':main()
