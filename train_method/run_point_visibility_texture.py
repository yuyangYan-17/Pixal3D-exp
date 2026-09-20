#!/usr/bin/env python3
"""Entry point for the point-level visibility experiment.

The implementation is kept at the repository root for compatibility with the
existing sr_tools imports; this wrapper makes the experiment runnable from the
requested train_method workspace.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "sr_point_visibility_texture.py"),
        run_name="__main__",
    )
