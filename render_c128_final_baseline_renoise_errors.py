#!/usr/bin/env python3
"""Recompute saved-render foreground errors and metrics with fixed reference masks."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/c128_final_baseline_renoise_visibility_gated_2048_cuda4'
BASE=ROOT/'outputs/baseline1024_raw_ovoxel_cuda4_0_img'
LIMIT=.25
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def rgb(path):
    return np.asarray(Image.open(path).convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS),np.float32)/255

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    records=summary['results'];assert len(records)==12
    refs={'front':rgb(BASE/'canonical_1024.png'),'back':rgb(OUT/'baseline1024_reference/back/render.png')}
    masks={'front':np.asarray(Image.open(BASE/'raw_ovoxel_render/alpha.png').convert('L').resize((1024,1024),Image.Resampling.NEAREST))>127,
           'back':np.asarray(Image.open(OUT/'baseline1024_reference/back/mask.png').convert('L').resize((1024,1024),Image.Resampling.NEAREST))>127}
    rows=[]
    for view in ('front','back'):
        reference=refs[view];mask=masks[view];assert mask.any()
        gap=32;size=1024;label=180;header=160
        sheet=Image.new('RGB',(4*size+5*gap,header+3*(size+label+gap)+gap+130),(18,20,25))
        draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,38);small=ImageFont.truetype(FONT,30)
        refname='canonical input' if view=='front' else 'baseline back render (not GT)'
        draw.text((30,15),f'{view.title()} error | reference: {refname}',font=font,fill='white')
        draw.text((30,72),'Per-pixel mean absolute RGB error | fixed foreground mask | same color scale for both views',font=small,fill='white')
        for index,record in enumerate(records):
            pred=rgb(record[f'{view}_render']);assert pred.shape==reference.shape==(1024,1024,3)
            error=np.mean(np.abs(pred-reference),axis=2)
            mse=float(np.mean((pred[mask]-reference[mask])**2))
            psnr=10*np.log10(1/max(mse,1e-12))
            _,ssim_map=structural_similarity(reference,pred,data_range=1.,channel_axis=2,full=True)
            ssim=float(ssim_map[mask].mean())
            row={'n':record['n'],'view':view,'foreground_psnr_db':float(psnr),'foreground_ssim':ssim,'foreground_mae':float(error[mask].mean()),'foreground_pixels':int(mask.sum())};rows.append(row)
            colors=(matplotlib.colormaps['magma'](np.clip(error/LIMIT,0,1))[...,:3]*255).astype(np.uint8);colors[~mask]=0
            heat=Image.fromarray(colors)
            variant=OUT/f"variants/n_{record['n']:02d}"
            heat.save(variant/f'{view}_foreground_error_map.png')
            np.save(variant/f'{view}_foreground_absolute_error.npy',np.where(mask,error,0))
            x=gap+(index%4)*(size+gap);y=header+gap+(index//4)*(size+label+gap)
            draw.text((x+12,y+8),f"n={record['n']:02d} | noise t={record['start_t']:.4f} / Flow {12-record['n']} steps",font=font,fill='white')
            draw.text((x+12,y+65),f'FG PSNR {psnr:.4f} dB | SSIM {ssim:.4f}',font=small,fill=(110,205,255))
            sheet.paste(heat,(x,y+label))
        x=500;y=sheet.height-92;width=3000
        gradient=matplotlib.colormaps['magma'](np.linspace(0,1,width))[None,:,:3]
        sheet.paste(Image.fromarray(np.uint8(np.repeat(gradient,26,axis=0)*255)),(x,y))
        for i in range(6):draw.text((x+i*width//5-20,y+33),f'{LIMIT*i/5:.2f}'+('+' if i==5 else ''),font=small,fill='white')
        draw.text((30,y),'RGB MAE',font=small,fill='white')
        sheet.save(OUT/f'{view}_error_maps_3x4.png')
    with (OUT/'front_back_foreground_metrics.csv').open('w') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,2,figsize=(15,9),dpi=160)
    for col,view in enumerate(('front','back')):
        data=[r for r in rows if r['view']==view];ns=[r['n'] for r in data]
        for axis,key,ylabel in zip(axes[:,col],('foreground_psnr_db','foreground_ssim'),('PSNR (dB)','SSIM')):
            values=[r[key] for r in data]
            axis.plot(ns,values,'o-',color='#169b92' if view=='front' else '#ad63db',linewidth=2)
            best=int(np.argmax(values));axis.scatter(ns[best],values[best],s=85,facecolors='none',edgecolors='#e89e21',linewidths=2,zorder=3)
            axis.annotate(f'best n={ns[best]}: {values[best]:.4f}',(ns[best],values[best]),xytext=(8,-20) if best==0 else (-160,-20),textcoords='offset points')
            axis.set(xlabel='Noise start index n: t=1-n/12; remaining Flow steps=12-n',ylabel=ylabel,xticks=ns)
            axis.grid(alpha=.25)
        axes[0,col].set_title('Input view vs canonical input' if view=='front' else 'Back view vs baseline back (not GT)')
    fig.suptitle('Final baseline re-noise + C128 visibility-gated flow (1024 renders)',fontsize=17)
    fig.tight_layout();fig.savefig(OUT/'foreground_metrics.png');fig.savefig(OUT/'foreground_metrics.svg');plt.close(fig)
    metadata={'error':'mean(abs(pred_rgb-reference_rgb), axis=RGB), linear display of error in saved RGB values','color_map':'magma','color_range':[0,LIMIT],'saturates_above':LIMIT,'outside_foreground':'black / excluded from all foreground metrics','metrics':'saved 8-bit RGB PNG / 255; PSNR from foreground RGB MSE; skimage default 7x7 SSIM map averaged over fixed foreground','references':{'front':str(BASE/'canonical_1024.png'),'back':str(OUT/'baseline1024_reference/back/render.png')},'results':rows}
    (OUT/'error_metrics_summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Done:',OUT)
    for view in ('front','back'):
        best=max((r for r in rows if r['view']==view),key=lambda r:r['foreground_psnr_db']);print(best)

if __name__=='__main__':main()
