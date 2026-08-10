import pytest

from pixal3d.renderers.pbr_mesh_renderer import (
    NVDIFFRAST_MAX_SUBTRIANGLES,
    _validate_nvdiffrast_face_chunking,
)


def test_full_mesh_below_nvdiffrast_limit_is_allowed() -> None:
    _validate_nvdiffrast_face_chunking(
        NVDIFFRAST_MAX_SUBTRIANGLES - 1,
        0,
    )


def test_oversized_mesh_requires_safe_face_chunks() -> None:
    with pytest.raises(RuntimeError, match="fixed.*work-buffer limit"):
        _validate_nvdiffrast_face_chunking(
            NVDIFFRAST_MAX_SUBTRIANGLES + 1,
            0,
        )

    with pytest.raises(RuntimeError, match="4,000,000 is recommended"):
        _validate_nvdiffrast_face_chunking(
            NVDIFFRAST_MAX_SUBTRIANGLES + 1,
            NVDIFFRAST_MAX_SUBTRIANGLES,
        )

    _validate_nvdiffrast_face_chunking(
        NVDIFFRAST_MAX_SUBTRIANGLES + 1,
        4_000_000,
    )


def test_negative_chunk_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="face_chunk_size"):
        _validate_nvdiffrast_face_chunking(1, -1)
