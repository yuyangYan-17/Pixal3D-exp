#!/usr/bin/env python3
"""Transfer baseline triangle visibility to nearest-surface C128 cells and render."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')
import json, math
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import nvdiffrast.torch as dr
import utils3d
from PIL import Image, ImageDraw, ImageFont
from pixal3d.representations import Mesh
from pixal3d.renderers import MeshRenderer
from pixal3d.renderers.pbr_mesh_renderer import intrinsics_to_projection

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/c128_baseline_visibility'
SOURCE=ROOT/'outputs/baseline1024_uniform_texture_endpoints_c1282048_cuda4/ovoxel1024/shared_geometry.pt'
SUPPORT=ROOT/'outputs/c128_baseline_endpoint_renoise_uncond_2048_cuda4/support/fixed_c128_shape_slat.pt'
CAMERA=ROOT/'outputs/baseline1024_raw_ovoxel_cuda4_0_img/global_camera.json'
VISIBLE=np.array([45,211,171],np.float32)/255
HIDDEN=np.array([229,94,121],np.float32)/255
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def raster_visibility(v,f,camera,res=4096):
    device='cuda'; d=camera['distance']*camera.get('mesh_scale',1)
    ext=utils3d.torch.extrinsics_look_at(torch.tensor([0.,0.,d],device=device),torch.zeros(3,device=device),torch.tensor([0.,1.,0.],device=device))
    angle=torch.tensor(camera['camera_angle_x'],device=device)
    intr=utils3d.torch.intrinsics_from_fov_xy(angle,angle)
    vg=torch.as_tensor(v,device=device); fg=torch.as_tensor(f.astype(np.int32),device=device)
    clip=torch.cat([vg,torch.ones_like(vg[:,:1])],1)@(intrinsics_to_projection(intr,.01,10)@ext).T
    ctx=dr.RasterizeCudaContext(device=device)
    best=torch.full((res,res),float('inf'),device=device); ids=torch.full((res,res),-1,device=device,dtype=torch.int32)
    for start in range(0,len(f),2000000):
        rast,_=dr.rasterize(ctx,clip[None],fg[start:start+2000000],(res,res))
        valid=(rast[0,:,:,3]>0)&(rast[0,:,:,2]<best)
        ids[valid]=rast[0,:,:,3][valid].int()-1+start;best[valid]=rast[0,:,:,2][valid]
    ids=ids.cpu().numpy(); visible=np.zeros(len(f),bool);visible[np.unique(ids[ids>=0])]=True
    Image.fromarray(np.uint8((ids>=0)*255)).save(OUT/'baseline_camera_foreground.png')
    torch.save(torch.from_numpy(ids),OUT/'camera_triangle_ids_4096.pt')
    return visible

def make_voxels(coords,visible,offset=None):
    centers=(coords+.5)/128-.5
    if offset is not None: centers=centers+offset
    # Separate faces provide flat directional shading and visible voxel gaps.
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],np.float32)-.5
    quads=np.array([[0,3,2,1],[4,5,6,7],[0,1,5,4],[3,7,6,2],[0,4,7,3],[1,2,6,5]])
    template=corners[quads].reshape(24,3)*(.9/128)
    vertices=(centers[:,None,:]+template[None]).reshape(-1,3)
    faces=(np.array([[0,1,2],[0,2,3]])[None]+np.arange(6)[:,None,None]*4).reshape(-1,3)
    faces=(faces[None]+np.arange(len(coords))[:,None,None]*24).reshape(-1,3)
    colors=np.where(visible[:,None],VISIBLE,HIDDEN)
    shades=np.repeat(np.array([.61,1.,.65,.96,.72,.86]),4)
    attrs=(colors[:,None,:]*shades[None,:,None]).reshape(-1,3)
    return Mesh(torch.tensor(vertices,device='cuda'),torch.tensor(faces,dtype=torch.int32,device='cuda'),torch.tensor(attrs,dtype=torch.float32,device='cuda'))

def box_lines(lo,hi,color):
    verts=[];faces=[];attrs=[]
    for axis in range(3):
        fixed=[a for a in range(3) if a!=axis]
        for u in (0,1):
            for w in (0,1):
                a=np.array(lo,dtype=np.float32);b=a.copy();b[axis]=hi[axis]
                a[fixed[0]]=b[fixed[0]]=(lo,hi)[u][fixed[0]]
                a[fixed[1]]=b[fixed[1]]=(lo,hi)[w][fixed[1]]
                # Thin rectangular 3D rod.
                low=np.minimum(a,b)-.0012;high=np.maximum(a,b)+.0012
                cr=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
                fs=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
                faces.append(fs+len(verts)*8);verts.append(cr);attrs.append(np.tile(color,(8,1)))
    return np.concatenate(verts),np.concatenate(faces),np.concatenate(attrs)

def render(coords,visible,owner,exploded=False,cube=None,res=1400):
    selected=np.ones(len(coords),bool) if cube is None else owner==cube
    c=coords[selected];vis=visible[selected]
    signs=(c//64)*2-1
    offset=signs*.15 if exploded else None
    if cube is not None: offset=-((c//64)*.5-.25)
    vox=make_voxels(c,vis,offset)
    lines=[]
    for i in (range(8) if cube is None else [cube]):
        start=np.array([i//4,(i//2)%2,i%2])*.5-.5
        off=(start+.25)*.6 if exploded else np.zeros(3)
        if cube is not None: off=-(start+.25)
        lines.append(box_lines(start+off,start+.5+off,np.array([.75,.80,.91])))
    vv=[];ff=[];aa=[];n=0
    for v,f,a in lines:vv.append(v);ff.append(f+n);aa.append(a);n+=len(v)
    wire=Mesh(torch.tensor(np.concatenate(vv),dtype=torch.float32,device='cuda'),torch.tensor(np.concatenate(ff),dtype=torch.int32,device='cuda'),torch.tensor(np.concatenate(aa),dtype=torch.float32,device='cuda'))
    direction=torch.tensor([1.35,1.15,1.8],device='cuda');direction/=direction.norm()
    distance=3.5 if exploded else (1.35 if cube is not None else 2.7)
    ext=utils3d.torch.extrinsics_look_at(direction*distance,torch.zeros(3,device='cuda'),torch.tensor([0.,1.,0.],device='cuda'))
    angle=torch.tensor(math.radians(40),device='cuda');intr=utils3d.torch.intrinsics_from_fov_xy(angle,angle)
    renderer=MeshRenderer({'resolution':res,'near':.01,'far':10,'ssaa':1,'antialias':True,'chunk_size':2000000},device='cuda')
    buf=renderer.render(vox,ext,intr,return_types=['attr','depth','mask'])
    wb=renderer.render(wire,ext,intr,return_types=['attr','depth','mask'])
    rgb=buf['attr'].permute(1,2,0);mask=buf['mask'];rgb=rgb*mask[:,:,None]+torch.tensor([.035,.045,.07],device='cuda')*(1-mask[:,:,None])
    front=(buf['mask']<.5)|(wb['depth']<=buf['depth']+.0001)
    alpha=wb['mask']*torch.where(front,.8,.14)
    rgb=rgb*(1-alpha[:,:,None])+wb['attr'].permute(1,2,0)*alpha[:,:,None]
    return Image.fromarray((rgb.clamp(0,1).cpu().numpy()*255).astype(np.uint8))

@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True,parents=True)
    camera=json.loads(CAMERA.read_text())
    payload=torch.load(SOURCE,map_location='cpu',weights_only=False)
    v=payload['vertices'].float().numpy();f=payload['faces'].long().numpy()
    coords=torch.load(SUPPORT,map_location='cpu',weights_only=False)['coords'][:,1:].numpy()
    centers=(coords.astype(np.float32)+.5)/128-.5
    print(f'baseline {len(v):,} vertices / {len(f):,} faces; C128 {len(coords):,}',flush=True)
    visible=raster_visibility(v,f,camera)
    raster_count=int(visible.sum());print(f'4096 z-buffer visible faces {raster_count:,}',flush=True)
    scene=o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(v),o3d.core.Tensor(f.astype(np.uint32)))
    # Pixel rasterization can miss subpixel triangles: also test every remaining
    # triangle centroid with an exact first-hit ray from the SAME source camera.
    origin=np.array([0,0,camera['distance']*camera.get('mesh_scale',1)],np.float32)
    focal=.5/math.tan(camera['camera_angle_x']/2)
    for start in range(0,len(f),200000):
        face_ids=np.arange(start,min(start+200000,len(f)))
        face_ids=face_ids[~visible[face_ids]]
        pts=v[f[face_ids]].mean(1);rays=pts-origin
        uv=np.stack([focal*rays[:,0]/(-rays[:,2])+.5,-focal*rays[:,1]/(-rays[:,2])+.5],1)
        inside=(uv>=0).all(1)&(uv<1).all(1)&(rays[:,2]<0)
        face_ids=face_ids[inside];rays=rays[inside]
        if len(rays):
            hit=scene.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(origin,rays.shape),rays],1).astype(np.float32)))
            visible[face_ids]=hit['primitive_ids'].numpy()==face_ids
    closest=scene.compute_closest_points(o3d.core.Tensor(centers))
    nearest=closest['primitive_ids'].numpy().astype(np.int64)
    nearest_points=closest['points'].numpy();dist=np.linalg.norm(centers-nearest_points,axis=1)
    inherited=visible[nearest];owner=(coords//64)@np.array([4,2,1])
    assert np.isfinite(dist).all() and (nearest>=0).all() and (nearest<len(f)).all()
    stats=[]
    for i in range(8):
        m=owner==i;stats.append({'cube_id':i,'start':[i//4*64,(i//2)%2*64,i%2*64],'tokens':int(m.sum()),'visible':int(inherited[m].sum()),'hidden':int((~inherited[m]).sum()),'visible_fraction':float(inherited[m].mean())})
    summary={'baseline_mesh':str(SOURCE),'support':str(SUPPORT),'camera_source':str(CAMERA),'camera':camera,'baseline_faces':len(f),'raster_visible_faces':raster_count,'visible_faces':int(visible.sum()),'visibility_definition':'4096x4096 first-hit triangle-ID raster, union exact first-hit rays at in-frame centroids of remaining faces; two-sided, finite sampling','inheritance':'C128 cell center -> exact closest point on baseline triangle surface via Open3D BVH -> that face binary visibility; no nearest-centroid approximation','voxel_geometry':'[(i)/128-.5,(i+1)/128-.5]; displayed at 90% width for gaps','display_camera':'independent observer from normalized (1.35,1.15,1.8), looking at origin; visibility remains fixed to generation camera +Z','colors':{'visible':'teal','hidden':'rose'},'tokens':len(coords),'visible_tokens':int(inherited.sum()),'hidden_tokens':int((~inherited).sum()),'nearest_distance_quantiles':np.quantile(dist,[0,.5,.95,.99,1]).tolist(),'cubes':stats,'exploded_view':'only display: cube center offsets = sign(center)*0.15; assembled view uses original generation coordinates'}
    torch.save({'coords':torch.from_numpy(coords),'face_visible':torch.from_numpy(visible),'nearest_face_id':torch.from_numpy(nearest),'nearest_point':torch.from_numpy(nearest_points),'surface_distance':torch.from_numpy(dist),'visible':torch.from_numpy(inherited),'cube_id':torch.from_numpy(owner)},OUT/'visibility.pt')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    panels=[]
    for exploded,name in [(False,'assembled'),(True,'exploded')]:
        im=render(coords,inherited,owner,exploded=exploded);im.save(OUT/f'{name}.png');panels.append(im)
    sheet=Image.new('RGB',(2800,1510),(9,12,18));draw=ImageDraw.Draw(sheet)
    for i,im in enumerate(panels):sheet.paste(im,(i*1400,110))
    font=ImageFont.truetype(FONT,30)
    draw.text((35,18),'C128 in generation space | upper oblique view',font=font,fill='white')
    draw.text((1435,18),'8 C64 blocks | exploded display',font=font,fill='white')
    draw.text((35,64),f'Teal: source-camera visible ({inherited.sum():,})   Rose: hidden ({(~inherited).sum():,})   Source camera: +Z',font=font,fill='white')
    sheet.save(OUT/'c128_visibility_overview.png')
    sheet=Image.new('RGB',(2400,1320),(9,12,18));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,21)
    for i,row in enumerate(stats):
        x=(i%4)*600;y=(i//4)*660
        sheet.paste(render(coords,inherited,owner,cube=i,res=600),(x,y+60))
        draw.text((x+15,y+6),f'C64 #{i} start={row["start"]}',font=font,fill='white')
        draw.text((x+15,y+32),f'Visible {row["visible"]:,} / {row["tokens"]:,} ({row["visible_fraction"]:.1%})',font=font,fill=tuple((VISIBLE*255).astype(int)))
    sheet.save(OUT/'c64_visibility_details.png')
    print('[done]',OUT,flush=True)

if __name__=='__main__':main()
