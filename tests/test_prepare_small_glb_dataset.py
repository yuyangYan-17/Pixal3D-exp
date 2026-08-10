import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_small_glb_dataset as prep


def test_glb_material_detection(tmp_path):
    def write_glb(path, document):
        payload = json.dumps(document).encode("utf-8")
        payload += b" " * ((-len(payload)) % 4)
        total = 12 + 8 + len(payload)
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(payload), b"JSON") + payload)

    with_material = tmp_path / "yes.glb"
    without_material = tmp_path / "no.glb"
    write_glb(with_material, {"asset": {"version": "2.0"}, "materials": [{}]})
    write_glb(without_material, {"asset": {"version": "2.0"}})
    assert prep.glb_has_material(with_material)
    assert not prep.glb_has_material(without_material)


def test_tile_layout_is_7_by_7():
    boxes = prep.tile_boxes()
    assert len(boxes) == 49
    assert boxes[0] == (0, 0, 1024, 1024)
    assert boxes[-1] == (3072, 3072, 4096, 4096)


def test_projection_and_overlapping_tile_counts():
    coords = np.array([[31, 31, 31], [32, 32, 32]], dtype=np.uint8)
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[2, 3] = 2.0
    intrinsics = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    crop = {"square_extent_source": [0, 0, 4096, 4096]}
    uv, valid, indices, counts = prep.project_global_support(coords, extrinsics, intrinsics, crop)
    assert uv.shape == (2, 2)
    assert valid.all()
    assert len(indices) == 49
    assert counts.shape == (49,)
    # Central points belong to four overlapping 1024 tiles each.
    assert counts.sum() == 8


def test_selection_count_rule():
    assert max(1, int(np.ceil(1 / 3))) == 1
    assert max(1, int(np.ceil(10 / 3))) == 4


def test_discovery_includes_staged_glb_for_resume(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    staged = output_root / "0123456789abcdef" / "0123456789abcdef.glb"
    input_root.mkdir()
    staged.parent.mkdir(parents=True)
    payload = json.dumps({"asset": {"version": "2.0"}, "materials": [{}]}).encode("utf-8")
    payload += b" " * ((-len(payload)) % 4)
    staged.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20 + len(payload))
        + struct.pack("<I4s", len(payload), b"JSON")
        + payload
    )
    records = prep.discover_glbs(input_root, output_root, workers=1)
    assert len(records) == 1
    assert records[0]["sources"] == [staged]


def test_mark_skipped_removes_partial_artifacts_but_keeps_glb(tmp_path):
    object_id = "0123456789abcdef"
    full_hash = object_id + "0" * 48
    object_dir = tmp_path / object_id
    object_dir.mkdir()
    glb = object_dir / f"{object_id}.glb"
    glb.write_bytes(b"glb")
    for name in (".work", ".candidate_tmp", "slat", "views"):
        partial = object_dir / name
        partial.mkdir()
        (partial / "partial.bin").write_bytes(b"partial")
    prep.mark_object_skipped(object_dir, object_id, full_hash, "pbr", "unsupported")
    assert glb.exists()
    assert prep.object_is_skipped(object_dir, full_hash)
    assert not any((object_dir / name).exists() for name in (".work", ".candidate_tmp", "slat", "views"))
