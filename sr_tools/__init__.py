"""Single-image, globally synchronized shape and material super-resolution."""

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("PIXAL3D_LOW_MEMORY_DECODER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).resolve().parents[1] / "autotune_cache.json"),
)
os.environ["FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE"] = "0"
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")
