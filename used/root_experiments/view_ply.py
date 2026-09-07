import open3d as o3d
import numpy as np
from PIL import Image
import argparse
import os


def normalize_points(points):

    center = points.mean(axis=0)

    points = points - center

    scale = np.max(
        np.linalg.norm(points, axis=1)
    )

    points = points / scale

    return points



def look_at(
    camera,
    target=np.array([0,0,0])
):

    forward = target-camera
    forward /= np.linalg.norm(forward)

    up=np.array([0,1,0])

    right=np.cross(
        forward,
        up
    )

    right/=np.linalg.norm(right)

    up=np.cross(
        right,
        forward
    )

    return forward,right,up



def render_pointcloud(
        points,
        colors,
        camera,
        size=512
):

    renderer=o3d.visualization.rendering.OffscreenRenderer(
        size,
        size
    )


    mat=o3d.visualization.rendering.MaterialRecord()

    mat.shader="defaultUnlit"

    mat.point_size=2.0


    pcd=o3d.geometry.PointCloud()

    pcd.points=o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors=o3d.utility.Vector3dVector(colors)



    renderer.scene.add_geometry(
        "pcd",
        pcd,
        mat
    )


    center=np.array([0,0,0])


    renderer.setup_camera(
        60,
        center,
        camera,
        [0,1,0]
    )


    img=renderer.render_to_image()


    img=np.asarray(img)

    return Image.fromarray(img)



def make_grid(imgs,cols=4):

    w,h=imgs[0].size

    rows=int(np.ceil(len(imgs)/cols))

    canvas=Image.new(
        "RGB",
        (cols*w,rows*h)
    )

    for i,img in enumerate(imgs):

        canvas.paste(
            img,
            (
                i%cols*w,
                i//cols*h
            )
        )

    return canvas



def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--ply",
        required=True
    )

    parser.add_argument(
        "--out",
        default="views.png"
    )

    parser.add_argument(
        "--points",
        type=int,
        default=10000000
    )

    args=parser.parse_args()



    print("load",args.ply)


    pcd=o3d.io.read_point_cloud(
        args.ply
    )


    points=np.asarray(
        pcd.points
    )


    colors=None

    if pcd.has_colors():
        colors=np.asarray(
            pcd.colors
        )


    print(
        "points:",
        points.shape
    )



    # 下采样
    if len(points)>args.points:

        idx=np.random.choice(
            len(points),
            args.points,
            replace=False
        )

        points=points[idx]

        if colors is not None:
            colors=colors[idx]



    points=normalize_points(points)



    views=[

        [0,0,3],
        [0,0,-3],

        [3,0,0],
        [-3,0,0],

        [0,3,0],
        [0,-3,0],

        [2,2,2],
        [-2,2,2],

    ]


    imgs=[]


    for v in views:

        print("render",v)

        img=render_pointcloud(
            points,
            colors,
            np.array(v),
            512
        )

        imgs.append(img)



    out=make_grid(
        imgs
    )


    out.save(args.out)

    print(
        "saved",
        args.out
    )



if __name__=="__main__":
    main()