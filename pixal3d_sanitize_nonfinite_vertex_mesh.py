#!/usr/bin/env python3
"""Remove faces incident to non-finite vertices and make unused vertices finite."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch

import pixal3d_global_c256_cube_owner_flow_singleview as cube_flow
from pixal3d.representations import MeshWithVertexPbr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    mesh = payload.get("mesh", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(mesh, MeshWithVertexPbr):
        raise TypeError(type(mesh))
    bad = ~torch.isfinite(mesh.vertices).all(1)
    bad_ids = torch.where(bad)[0]
    face_keep = ~bad[mesh.faces.long()].any(1)
    vertices = mesh.vertices.clone()
    vertices[bad] = 0
    clean = MeshWithVertexPbr(
        vertices, mesh.faces[face_keep].int(), mesh.vertex_attrs,
        layout=dict(mesh.layout),
    )
    cube_flow.atomic_save(args.output, {
        "format": "pixal3d_sanitized_nonfinite_vertex_mesh_v1",
        "mesh": clean,
        "source": str(args.input.resolve()),
        "nonfinite_vertices_replaced": int(bad.sum()),
        "incident_faces_removed": int((~face_keep).sum()),
        "vertices_retained": int(len(vertices)),
        "faces_retained": int(face_keep.sum()),
        "bad_vertex_ids": bad_ids,
    })
    print(
        f"[sanitize] bad_vertices={int(bad.sum())} "
        f"removed_faces={int((~face_keep).sum())} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
