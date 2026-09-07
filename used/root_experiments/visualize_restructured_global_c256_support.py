#!/usr/bin/env python3
"""Render the restructured Global C256 support, colored by its C64 block."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--support", type=Path, default=Path("outputs/global_c256_restructured_blocks_cuda5/global_support/global_c256_support.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/global_c256_restructured_blocks_cuda5/global_support/global_c256_support_block_colors_multiview.png"))
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    xyz = torch.load(args.support, map_location="cpu", weights_only=False)["coords"][:, 1:4].numpy()
    blocks_per_axis = 256 // args.block_size
    block = xyz // args.block_size
    ids = block[:, 0] * blocks_per_axis ** 2 + block[:, 1] * blocks_per_axis + block[:, 2]
    num_blocks = blocks_per_axis ** 3
    cmap = plt.get_cmap("turbo", num_blocks)
    colors = cmap(ids / max(1, num_blocks - 1))
    views = ((18, -55), (18, 35), (18, 125), (18, 215))
    fig = plt.figure(figsize=(20.48, 5.12), dpi=200, facecolor="black")
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d", facecolor="black")
        # Native world-up is SLAT y: matplotlib axes are (x,z,y), with no y flip.
        ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], c=colors, s=.08, alpha=.82,
                   linewidths=0, rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.set_title(f"azimuth {azim}°", color="white", fontsize=9)
    title = args.title or f"Global C256 support · colors identify non-overlap C{args.block_size} blocks"
    fig.suptitle(title,
                 color="white", fontsize=13)
    fig.subplots_adjust(left=.005, right=.995, top=.92, bottom=.01, wspace=.01)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="black")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
