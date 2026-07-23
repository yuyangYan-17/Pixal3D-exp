#!/usr/bin/env python3
import os
import shutil
import argparse


def collect_comparison_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file != "comparison.png":
                continue

            src_path = os.path.join(root, file)

            # root 示例:
            # .../Avocado/view_000/r1024
            parts = os.path.normpath(root).split(os.sep)

            try:
                # 找 r1024
                res_dir = parts[-1]
                # 找 view_000
                view_dir = parts[-2]
                # 找 mesh 名
                mesh_name = parts[-3]

                if not res_dir.startswith("r") or not view_dir.startswith("view"):
                    print(f"[Skip] unexpected path: {src_path}")
                    continue

            except Exception:
                print(f"[Skip] cannot parse: {src_path}")
                continue

            new_name = f"{mesh_name}{view_dir}_{res_dir}_comparison.png"

            dst_path = os.path.join(output_dir, new_name)

            shutil.copy2(src_path, dst_path)

            print(f"[COPY] {src_path}")
            print(f"    -> {dst_path}")

            count += 1

    print(f"\nDone. Copied {count} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect all comparison.png recursively"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing comparison.png files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store collected images"
    )

    args = parser.parse_args()

    collect_comparison_images(
        args.input_dir,
        args.output_dir
    )