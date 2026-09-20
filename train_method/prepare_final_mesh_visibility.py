#!/usr/bin/env python3
"""Rasterize the final decoded mesh and save a vertex visibility mask.

The older bridge visibility mask is computed on the intermediate mesh used to
encode C256.  This utility recomputes visibility on the actual final mesh so a
material-field merge can use the same surface that is rendered at 1024.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="4")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/sr_0_img_trilinear_face_n3_20260919/fixed"),
    )
    p.add_argument("--resolution", type=int, default=4096)
    p.add_argument("--face-chunk", type=int, default=1_000_000)
    p.add_argument("--dilation-radii", default="0,1,2")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    from sr_tools import common, visibility

    root = args.root.resolve()
    out = root / "shared/final_mesh/final_visibility"
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / "visible_vertices.pt"
    meta_path = out / "visibility.json"
    mesh = common.load_payload(root / "shared/final_mesh/geometry_mesh.pt")["mesh"]
    if mask_path.exists() and meta_path.exists():
        cached = common.load_payload(mask_path)
        face_ids = cached["face_ids"]
        visible_vertices = cached["visible_vertices"].bool()
        print(f"[cached raster] {mask_path}", flush=True)
    else:
        camera = json.loads((root / "baseline/camera.json").read_text())
        print(
            f"[mesh] vertices={len(mesh.vertices):,} faces={len(mesh.faces):,} "
            f"resolution={args.resolution}",
            flush=True,
        )
        face_ids = visibility.visible_faces(
            mesh,
            camera,
            resolution=args.resolution,
            face_chunk=args.face_chunk,
        )
        visible_vertices = torch.zeros(len(mesh.vertices), dtype=torch.bool)
        visible_vertices[mesh.faces[face_ids].reshape(-1).long().unique()] = True
        common.save(mask_path, visible_vertices=visible_vertices, face_ids=face_ids)
        common.js(
            meta_path,
            dict(
                status="COMPLETE",
                resolution=int(args.resolution),
                face_chunk=int(args.face_chunk),
                total_faces=len(mesh.faces),
                visible_faces=len(face_ids),
                total_vertices=len(mesh.vertices),
                visible_vertices=int(visible_vertices.sum()),
                visible_vertex_fraction=float(visible_vertices.float().mean()),
                rule="vertices incident to z-buffer-visible final-mesh faces",
            ),
        )

    radii = sorted({int(v.strip()) for v in args.dilation_radii.split(",") if v.strip()})
    if any(v < 0 for v in radii):
        raise ValueError("dilation radii must be non-negative")
    active = visible_vertices
    for radius in range(max(radii) + 1):
        if radius in radii:
            target = out / f"visible_vertices_d{radius}.pt"
            if not target.exists():
                common.save(target, visible_vertices=active)
                print(
                    f"[save] radius={radius} vertices={int(active.sum()):,}/"
                    f"{len(active):,}",
                    flush=True,
                )
        if radius == max(radii):
            break
        expanded = active.clone()
        for start in range(0, len(mesh.faces), int(args.face_chunk)):
            faces = mesh.faces[start : start + int(args.face_chunk)].long()
            hit = active[faces].any(dim=1)
            if bool(hit.any()):
                expanded[faces[hit].reshape(-1).unique()] = True
        active = expanded
    print(
        f"[done] faces={len(face_ids):,} visible_vertices="
        f"{int(visible_vertices.sum()):,}/{len(visible_vertices):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
