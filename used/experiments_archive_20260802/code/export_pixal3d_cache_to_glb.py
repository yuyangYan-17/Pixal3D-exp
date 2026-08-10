#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Pixal3D postprocess_cache vertices.pt + faces.pt to a geometry-only GLB.

The output contains only:
  - POSITION
  - triangle indices

It intentionally contains no vertex colors, normals, UVs, textures, or materials.
The GLB is written directly, avoiding trimesh's potentially large intermediate copies.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK_TYPE = b"JSON"
BIN_CHUNK_TYPE = b"BIN\x00"

ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

FLOAT = 5126
UNSIGNED_INT = 5125
TRIANGLES = 4


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def _find_tensor(obj: Any, preferred_keys: Iterable[str], source: Path) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    if isinstance(obj, dict):
        for key in preferred_keys:
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)

        tensor_values = [
            value for value in obj.values()
            if isinstance(value, (torch.Tensor, np.ndarray))
        ]
        if len(tensor_values) == 1:
            value = tensor_values[0]
            return value if isinstance(value, torch.Tensor) else torch.from_numpy(value)

        raise ValueError(
            f"{source} is a dict, but no unique tensor was found. "
            f"Available keys: {list(obj.keys())}"
        )

    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return _find_tensor(obj[0], preferred_keys, source)

    raise TypeError(f"Unsupported data type in {source}: {type(obj)!r}")


def _as_n_by_3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()

    # Remove only trivial singleton dimensions.
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.ndim == 1:
        if tensor.numel() % 3 != 0:
            raise ValueError(f"{name} has {tensor.numel()} values, not divisible by 3")
        tensor = tensor.reshape(-1, 3)
    elif tensor.ndim == 2:
        if tensor.shape[1] == 3:
            pass
        elif tensor.shape[0] == 3:
            tensor = tensor.transpose(0, 1)
        else:
            raise ValueError(f"{name} must have shape [N,3] or [3,N], got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"{name} must be one- or two-dimensional, got {tuple(tensor.shape)}")

    return tensor


def _pad4(n: int) -> int:
    return (4 - (n % 4)) % 4


def _write_rows(
    fp,
    tensor: torch.Tensor,
    numpy_dtype: np.dtype,
    chunk_rows: int,
) -> None:
    total_rows = int(tensor.shape[0])
    for start in range(0, total_rows, chunk_rows):
        end = min(start + chunk_rows, total_rows)
        chunk = tensor[start:end].contiguous().numpy()
        chunk = chunk.astype(numpy_dtype, copy=False)
        chunk.tofile(fp)


def export_geometry_only_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_path: Path,
    chunk_rows: int,
    validate: bool,
) -> None:
    vertices = _as_n_by_3(vertices, "vertices")
    faces = _as_n_by_3(faces, "faces")

    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])

    if vertex_count == 0:
        raise ValueError("vertices is empty")
    if face_count == 0:
        raise ValueError("faces is empty")

    if not vertices.dtype.is_floating_point:
        vertices = vertices.to(torch.float32)

    # Compute POSITION bounds required/recommended by glTF viewers.
    vertex_min = vertices.amin(dim=0).to(torch.float64)
    vertex_max = vertices.amax(dim=0).to(torch.float64)

    if not torch.isfinite(vertex_min).all() or not torch.isfinite(vertex_max).all():
        raise ValueError("vertices contains NaN or Inf")

    if validate:
        face_min = int(faces.amin().item())
        face_max = int(faces.amax().item())
        if face_min < 0:
            raise ValueError(f"faces contains a negative vertex index: {face_min}")
        if face_max >= vertex_count:
            raise ValueError(
                f"faces references vertex {face_max}, but vertex_count={vertex_count}"
            )

    position_nbytes = vertex_count * 3 * np.dtype("<f4").itemsize
    position_padding = _pad4(position_nbytes)

    index_offset = position_nbytes + position_padding
    index_nbytes = face_count * 3 * np.dtype("<u4").itemsize
    index_padding = _pad4(index_nbytes)

    bin_nbytes = index_offset + index_nbytes + index_padding

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Pixal3D geometry-only cache exporter",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": TRIANGLES,
            }]
        }],
        "buffers": [{"byteLength": bin_nbytes}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": position_nbytes,
                "target": ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_nbytes,
                "target": ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": FLOAT,
                "count": vertex_count,
                "type": "VEC3",
                "min": [float(x) for x in vertex_min.tolist()],
                "max": [float(x) for x in vertex_max.tolist()],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": UNSIGNED_INT,
                "count": face_count * 3,
                "type": "SCALAR",
            },
        ],
    }

    json_bytes = json.dumps(
        gltf, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_bytes += b" " * _pad4(len(json_bytes))

    total_length = (
        12
        + 8 + len(json_bytes)
        + 8 + bin_nbytes
    )

    # GLB header/chunk sizes are uint32.
    if total_length > 0xFFFFFFFF:
        raise ValueError(
            f"Output GLB would be {total_length / 1024**3:.2f} GiB, exceeding "
            "the GLB uint32 file-size limit. Split the mesh before export."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] vertices: {tuple(vertices.shape)} {vertices.dtype}")
    print(f"[load] faces:    {tuple(faces.shape)} {faces.dtype}")
    print(f"[write] estimated GLB size: {total_length / 1024**3:.2f} GiB")
    print(f"[write] output: {output_path}")

    with output_path.open("wb") as fp:
        fp.write(struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length))

        fp.write(struct.pack("<I4s", len(json_bytes), JSON_CHUNK_TYPE))
        fp.write(json_bytes)

        fp.write(struct.pack("<I4s", bin_nbytes, BIN_CHUNK_TYPE))

        _write_rows(fp, vertices, np.dtype("<f4"), chunk_rows)
        if position_padding:
            fp.write(b"\x00" * position_padding)

        _write_rows(fp, faces, np.dtype("<u4"), chunk_rows)
        if index_padding:
            fp.write(b"\x00" * index_padding)

    actual_size = output_path.stat().st_size
    if actual_size != total_length:
        raise RuntimeError(
            f"GLB size mismatch: expected {total_length}, wrote {actual_size}"
        )

    print(f"[done] geometry-only GLB written: {output_path}")
    print("[done] no colors, materials, normals, UVs, or textures were exported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Pixal3D postprocess_cache geometry to a colorless GLB."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing vertices.pt and faces.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .glb path; defaults to <cache-dir>/geometry_only.glb",
    )
    parser.add_argument(
        "--vertices-file",
        default="vertices.pt",
        help="Vertex tensor filename inside --cache-dir",
    )
    parser.add_argument(
        "--faces-file",
        default="faces.pt",
        help="Face tensor filename inside --cache-dir",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows converted/written per chunk to limit temporary memory",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checking whether face indices are within the vertex range",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    cache_dir = args.cache_dir.expanduser().resolve()
    vertices_path = cache_dir / args.vertices_file
    faces_path = cache_dir / args.faces_file
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else cache_dir / "geometry_only.glb"
    )

    if output_path.suffix.lower() != ".glb":
        raise ValueError("--output must end with .glb")
    if not vertices_path.is_file():
        raise FileNotFoundError(vertices_path)
    if not faces_path.is_file():
        raise FileNotFoundError(faces_path)

    vertices_obj = _torch_load(vertices_path)
    faces_obj = _torch_load(faces_path)

    vertices = _find_tensor(
        vertices_obj,
        preferred_keys=("vertices", "verts", "points", "xyz"),
        source=vertices_path,
    )
    faces = _find_tensor(
        faces_obj,
        preferred_keys=("faces", "triangles", "indices"),
        source=faces_path,
    )

    export_geometry_only_glb(
        vertices=vertices,
        faces=faces,
        output_path=output_path,
        chunk_rows=args.chunk_rows,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
