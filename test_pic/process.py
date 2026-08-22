import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


# ============================================================
# 配置
# ============================================================

# ------------------------------------------------------------
# 所有输入图片统一 resize 到该尺寸
#
# 注意：
#     TARGET_H = 高
#     TARGET_W = 宽
#
# OpenCV resize 使用顺序为：
#     (width, height)
#
# 当前三视图目标尺寸：
#     3072 x 1024
# ------------------------------------------------------------

TARGET_H = 1024
TARGET_W = 3072


# ------------------------------------------------------------
# 黑背景阈值
#
# B/G/R 三个通道都 <= BLACK_THRESHOLD
# 时认为是黑色背景。
#
# 如果背景严格为纯黑，可以设为 0。
# ------------------------------------------------------------

BLACK_THRESHOLD = 10


# 普通 mask 半透明程度
ALPHA = 0.45


# 非重叠区域标记透明度
RESIDUAL_ALPHA = 0.60


# ============================================================
# 图片读取 + Resize
# ============================================================

def load_and_resize(path):
    """
    读取图片，并统一 resize 到：

        TARGET_W x TARGET_H

    后续所有比较都在统一尺寸下进行。
    """

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise FileNotFoundError(
            f"无法读取图片: {path}"
        )

    original_h, original_w = img.shape[:2]

    print(f"读取图片: {path}")
    print(
        f"  原始尺寸 : "
        f"{original_w} x {original_h}"
    )

    # --------------------------------------------------------
    # 根据是放大还是缩小自动选择 interpolation
    #
    # 缩小：
    #     INTER_AREA
    #
    # 放大：
    #     INTER_CUBIC
    # --------------------------------------------------------

    if (
        TARGET_W <= original_w
        and TARGET_H <= original_h
    ):
        interpolation = cv2.INTER_AREA
        interpolation_name = "INTER_AREA"

    else:
        interpolation = cv2.INTER_CUBIC
        interpolation_name = "INTER_CUBIC"

    img = cv2.resize(
        img,
        (
            TARGET_W,
            TARGET_H,
        ),
        interpolation=interpolation,
    )

    print(
        f"  Resize后  : "
        f"{TARGET_W} x {TARGET_H}"
    )

    print(
        f"  插值方式  : "
        f"{interpolation_name}"
    )

    return img


# ============================================================
# Foreground mask
# ============================================================

def get_object_mask(
    img,
    threshold=BLACK_THRESHOLD,
):
    """
    黑色背景 -> False
    非黑色物体 -> True

    只要 B/G/R 中至少一个通道高于 threshold，
    就认为该像素属于前景。
    """

    return (
        np.max(
            img,
            axis=2,
        )
        > threshold
    )


def make_binary_mask(mask):
    """
    将 bool mask 保存成：

        前景 = 255
        背景 = 0
    """

    result = np.zeros(
        mask.shape,
        dtype=np.uint8,
    )

    result[mask] = 255

    return result


# ============================================================
# 去黑背景
# ============================================================

def remove_black_background(
    img,
    mask,
):
    """
    将背景变成透明，
    保留原始 RGB。
    """

    bgra = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2BGRA,
    )

    bgra[:, :, 3] = np.where(
        mask,
        255,
        0,
    ).astype(np.uint8)

    return bgra


# ============================================================
# Foreground PSNR / SSIM
# ============================================================

def compute_foreground_metrics(
    img1,
    img2,
    mask1,
    mask2,
):
    """
    只在两张图共同前景区域计算 PSNR / SSIM。

    common_mask:
        mask1 & mask2

    PSNR:
        只使用共同前景像素。

    SSIM:
        SSIM 使用 7x7 local window。

        为了尽量避免 object/background 边缘处
        的黑背景进入 SSIM window，
        将 common mask 向内腐蚀。
    """

    common_mask = (
        mask1
        & mask2
    )

    num_common = int(
        np.sum(
            common_mask
        )
    )

    if num_common == 0:
        print(
            "没有共同 foreground，"
            "无法计算 PSNR / SSIM。"
        )

        return None, None

    # ========================================================
    # PSNR
    # ========================================================

    img1_float = img1.astype(
        np.float64
    )

    img2_float = img2.astype(
        np.float64
    )

    pixels1 = img1_float[
        common_mask
    ]

    pixels2 = img2_float[
        common_mask
    ]

    mse = np.mean(
        (
            pixels1
            - pixels2
        )
        ** 2
    )

    if mse == 0:

        psnr = float(
            "inf"
        )

    else:

        psnr = (
            10.0
            * np.log10(
                (255.0 ** 2)
                / mse
            )
        )

    # ========================================================
    # SSIM
    # ========================================================

    rgb1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2RGB,
    )

    rgb2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2RGB,
    )

    _, ssim_map = structural_similarity(
        rgb1,
        rgb2,
        channel_axis=2,
        data_range=255,
        win_size=7,
        full=True,
    )

    # 某些 skimage 版本可能返回：
    #
    # H x W x C
    #
    # 转成：
    #
    # H x W
    if ssim_map.ndim == 3:

        ssim_map = np.mean(
            ssim_map,
            axis=2,
        )

    # --------------------------------------------------------
    # SSIM 使用 7x7 window。
    #
    # 对 common foreground 向内腐蚀，
    # 避免边缘的背景参与 SSIM。
    # --------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        dtype=np.uint8,
    )

    valid_ssim_mask = cv2.erode(
        common_mask.astype(
            np.uint8
        ),
        kernel,
        iterations=1,
    ).astype(bool)

    num_ssim_pixels = int(
        np.sum(
            valid_ssim_mask
        )
    )

    if num_ssim_pixels > 0:

        ssim = float(
            np.mean(
                ssim_map[
                    valid_ssim_mask
                ]
            )
        )

    else:

        print(
            "Warning: "
            "SSIM mask 腐蚀后为空，"
            "退化为 common mask。"
        )

        ssim = float(
            np.mean(
                ssim_map[
                    common_mask
                ]
            )
        )

    print(
        "\n"
        "========== Foreground Image Metrics =========="
    )

    print(
        f"Comparison resolution   : "
        f"{img1.shape[1]} x {img1.shape[0]}"
    )

    print(
        f"Common foreground pixels : "
        f"{num_common}"
    )

    print(
        f"SSIM valid pixels        : "
        f"{num_ssim_pixels}"
    )

    print(
        f"Foreground MSE           : "
        f"{mse:.6f}"
    )

    print(
        f"Foreground PSNR          : "
        f"{psnr:.6f} dB"
    )

    print(
        f"Foreground SSIM          : "
        f"{ssim:.6f}"
    )

    print(
        "=============================================="
        "\n"
    )

    return psnr, ssim


# ============================================================
# Mask overlay
# ============================================================

def overlay_mask(
    img,
    mask,
    color_bgr,
    alpha=ALPHA,
):
    """
    将指定 mask 以纯色半透明方式覆盖到原图。
    """

    result = img.astype(
        np.float32
    ).copy()

    color = np.array(
        color_bgr,
        dtype=np.float32,
    )

    result[mask] = (
        result[mask]
        * (1.0 - alpha)
        + color
        * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# Mask 差异
# ============================================================

def create_comparison(
    mask1,
    mask2,
    alpha=ALPHA,
):
    """
    Mask 差异图：

        红色：
            只有 image1 有

        蓝色：
            只有 image2 有

        绿色：
            两者重合

        背景：
            透明
    """

    only1 = (
        mask1
        & (~mask2)
    )

    only2 = (
        mask2
        & (~mask1)
    )

    overlap = (
        mask1
        & mask2
    )

    h, w = mask1.shape

    result = np.zeros(
        (
            h,
            w,
            4,
        ),
        dtype=np.uint8,
    )

    alpha_value = int(
        255 * alpha
    )

    # OpenCV 使用 BGRA

    # image1 独有 -> 红色
    result[only1] = [
        0,
        0,
        255,
        alpha_value,
    ]

    # image2 独有 -> 蓝色
    result[only2] = [
        255,
        0,
        0,
        alpha_value,
    ]

    # 两者重叠 -> 绿色
    result[overlap] = [
        0,
        255,
        0,
        alpha_value,
    ]

    return result


# ============================================================
# 两个 mask 半透明叠加
# ============================================================

def create_two_mask_overlay(
    mask1,
    mask2,
    alpha=ALPHA,
):
    """
    两个半透明 mask 直接叠加：

        image1 mask -> 红
        image2 mask -> 蓝

    重叠区域会自然混色。
    """

    h, w = mask1.shape

    canvas = np.zeros(
        (
            h,
            w,
            4,
        ),
        dtype=np.float32,
    )

    def alpha_composite(
        canvas,
        mask,
        bgr_color,
        a,
    ):

        color = np.array(
            bgr_color,
            dtype=np.float32,
        )

        current_rgb = (
            canvas[
                :,
                :,
                :3
            ]
        )

        current_a = (
            canvas[
                :,
                :,
                3
            ]
            / 255.0
        )

        src_a = np.zeros(
            (
                h,
                w,
            ),
            dtype=np.float32,
        )

        src_a[mask] = a

        out_a = (
            src_a
            + current_a
            * (
                1.0
                - src_a
            )
        )

        valid = (
            out_a
            > 1e-8
        )

        for c in range(3):

            src_c = np.zeros(
                (
                    h,
                    w,
                ),
                dtype=np.float32,
            )

            src_c[mask] = (
                color[c]
            )

            numerator = (
                src_c
                * src_a
                + current_rgb[
                    :,
                    :,
                    c
                ]
                * current_a
                * (
                    1.0
                    - src_a
                )
            )

            current_rgb[
                :,
                :,
                c
            ][valid] = (
                numerator[valid]
                / out_a[valid]
            )

        canvas[
            :,
            :,
            3
        ] = (
            out_a
            * 255.0
        )

    # image1 -> 红
    alpha_composite(
        canvas,
        mask1,
        [
            0,
            0,
            255,
        ],
        alpha,
    )

    # image2 -> 蓝
    alpha_composite(
        canvas,
        mask2,
        [
            255,
            0,
            0,
        ],
        alpha,
    )

    return np.clip(
        canvas,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# 非重叠区域 residual
# ============================================================

def extract_nonoverlap_rgb(
    img,
    nonoverlap_mask,
):
    """
    只保留非重叠区域的原始 RGB。

    非重叠区域：
        保留原图颜色

    其他区域：
        完全透明
    """

    result = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2BGRA,
    )

    # 默认全部透明
    result[
        :,
        :,
        3
    ] = 0

    # residual 区域不透明
    result[
        nonoverlap_mask,
        3
    ] = 255

    return result


def overlay_nonoverlap(
    img,
    nonoverlap_mask,
    color_bgr,
    alpha=RESIDUAL_ALPHA,
):
    """
    将非重叠区域以颜色半透明形式
    mask 回原图。

    方便直接观察 residual
    位于物体的哪个位置。
    """

    result = img.astype(
        np.float32
    ).copy()

    color = np.array(
        color_bgr,
        dtype=np.float32,
    )

    result[
        nonoverlap_mask
    ] = (
        result[
            nonoverlap_mask
        ]
        * (
            1.0
            - alpha
        )
        + color
        * alpha
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


def keep_only_nonoverlap_on_original(
    img,
    nonoverlap_mask,
):
    """
    只保留原图中的非重叠区域。

    其他区域全部变黑。
    """

    result = np.zeros_like(
        img
    )

    result[
        nonoverlap_mask
    ] = img[
        nonoverlap_mask
    ]

    return result


def create_nonoverlap_rgb_combined(
    img1,
    img2,
    mask1,
    mask2,
):
    """
    将两张图各自独有区域
    放到同一张透明图片。

    image1-only：
        使用 image1 原始 RGB

    image2-only：
        使用 image2 原始 RGB

    overlap：
        透明
    """

    only1 = (
        mask1
        & (~mask2)
    )

    only2 = (
        mask2
        & (~mask1)
    )

    h, w = mask1.shape

    result = np.zeros(
        (
            h,
            w,
            4,
        ),
        dtype=np.uint8,
    )

    # image1 独有
    result[
        only1,
        :3
    ] = img1[
        only1
    ]

    result[
        only1,
        3
    ] = 255

    # image2 独有
    result[
        only2,
        :3
    ] = img2[
        only2
    ]

    result[
        only2,
        3
    ] = 255

    return result


def create_nonoverlap_overlay_both(
    img1,
    img2,
    mask1,
    mask2,
    alpha=RESIDUAL_ALPHA,
):
    """
    创建统一 residual 位置图。

    底图：
        img1 和 img2 平均

    image1 独有：
        红色

    image2 独有：
        蓝色
    """

    only1 = (
        mask1
        & (~mask2)
    )

    only2 = (
        mask2
        & (~mask1)
    )

    base = (
        0.5
        * img1.astype(
            np.float32
        )
        + 0.5
        * img2.astype(
            np.float32
        )
    )

    # image1 residual -> 红
    red = np.array(
        [
            0,
            0,
            255,
        ],
        dtype=np.float32,
    )

    base[
        only1
    ] = (
        base[
            only1
        ]
        * (
            1.0
            - alpha
        )
        + red
        * alpha
    )

    # image2 residual -> 蓝
    blue = np.array(
        [
            255,
            0,
            0,
        ],
        dtype=np.float32,
    )

    base[
        only2
    ] = (
        base[
            only2
        ]
        * (
            1.0
            - alpha
        )
        + blue
        * alpha
    )

    return np.clip(
        base,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# Mask metrics
# ============================================================

def print_mask_metrics(
    mask1,
    mask2,
):
    """
    打印 silhouette / foreground mask 重叠情况。
    """

    intersection = int(
        np.sum(
            mask1
            & mask2
        )
    )

    union = int(
        np.sum(
            mask1
            | mask2
        )
    )

    area1 = int(
        np.sum(
            mask1
        )
    )

    area2 = int(
        np.sum(
            mask2
        )
    )

    only1 = int(
        np.sum(
            mask1
            & (~mask2)
        )
    )

    only2 = int(
        np.sum(
            mask2
            & (~mask1)
        )
    )

    iou = (
        intersection
        / union
        if union > 0
        else 1.0
    )

    coverage1 = (
        intersection
        / area1
        if area1 > 0
        else 0.0
    )

    coverage2 = (
        intersection
        / area2
        if area2 > 0
        else 0.0
    )

    residual_ratio1 = (
        only1
        / area1
        if area1 > 0
        else 0.0
    )

    residual_ratio2 = (
        only2
        / area2
        if area2 > 0
        else 0.0
    )

    print(
        "\n"
        "========== Mask Comparison =========="
    )

    print(
        f"Image 1 object pixels : "
        f"{area1}"
    )

    print(
        f"Image 2 object pixels : "
        f"{area2}"
    )

    print(
        f"Overlap pixels        : "
        f"{intersection}"
    )

    print(
        f"Only image 1          : "
        f"{only1}"
    )

    print(
        f"Only image 2          : "
        f"{only2}"
    )

    print(
        f"Union pixels          : "
        f"{union}"
    )

    print()

    print(
        f"IoU                   : "
        f"{iou:.6f}"
    )

    print(
        f"Mask1 overlap ratio   : "
        f"{coverage1:.6f}"
    )

    print(
        f"Mask2 overlap ratio   : "
        f"{coverage2:.6f}"
    )

    print(
        f"Image1 residual ratio : "
        f"{residual_ratio1:.6f}"
    )

    print(
        f"Image2 residual ratio : "
        f"{residual_ratio2:.6f}"
    )

    print(
        "====================================="
        "\n"
    )

    return {
        "area1": area1,
        "area2": area2,
        "intersection": intersection,
        "union": union,
        "only1": only1,
        "only2": only2,
        "iou": iou,
        "coverage1": coverage1,
        "coverage2": coverage2,
        "residual_ratio1": residual_ratio1,
        "residual_ratio2": residual_ratio2,
    }


# ============================================================
# Main
# ============================================================

def main(
    image1_path,
    image2_path,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. 读取 + Resize
    # ========================================================

    print(
        "\n"
        "========== Load & Resize =========="
    )

    img1 = load_and_resize(
        image1_path
    )

    print()

    img2 = load_and_resize(
        image2_path
    )

    print(
        "===================================\n"
    )

    # --------------------------------------------------------
    # sanity check
    # --------------------------------------------------------

    if img1.shape != img2.shape:

        raise RuntimeError(
            "\nResize 后尺寸仍然不同：\n"
            f"Image 1: "
            f"{img1.shape[1]} x "
            f"{img1.shape[0]}\n"
            f"Image 2: "
            f"{img2.shape[1]} x "
            f"{img2.shape[0]}"
        )

    height, width = (
        img1.shape[:2]
    )

    print(
        "========== Comparison Setup =========="
    )

    print(
        f"Image 1 : "
        f"{image1_path}"
    )

    print(
        f"Image 2 : "
        f"{image2_path}"
    )

    print(
        f"Target resolution : "
        f"{TARGET_W} x {TARGET_H}"
    )

    print(
        f"Actual resolution : "
        f"{width} x {height}"
    )

    print(
        f"Black threshold   : "
        f"{BLACK_THRESHOLD}"
    )

    print(
        "======================================\n"
    )

    # --------------------------------------------------------
    # 保存 resize 后的输入
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            output_dir
            / "image1_resized.png"
        ),
        img1,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_resized.png"
        ),
        img2,
    )

    # ========================================================
    # 2. Foreground masks
    # ========================================================

    mask1 = get_object_mask(
        img1
    )

    mask2 = get_object_mask(
        img2
    )

    cv2.imwrite(
        str(
            output_dir
            / "mask1.png"
        ),
        make_binary_mask(
            mask1
        ),
    )

    cv2.imwrite(
        str(
            output_dir
            / "mask2.png"
        ),
        make_binary_mask(
            mask2
        ),
    )

    # ========================================================
    # 3. Foreground PSNR / SSIM
    # ========================================================

    psnr, ssim = (
        compute_foreground_metrics(
            img1,
            img2,
            mask1,
            mask2,
        )
    )

    # ========================================================
    # 4. 去背景
    # ========================================================

    transparent1 = (
        remove_black_background(
            img1,
            mask1,
        )
    )

    transparent2 = (
        remove_black_background(
            img2,
            mask2,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_bg_removed.png"
        ),
        transparent1,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_bg_removed.png"
        ),
        transparent2,
    )

    # ========================================================
    # 5. Foreground mask overlay
    # ========================================================

    overlay1 = overlay_mask(
        img1,
        mask1,
        color_bgr=(
            0,
            0,
            255,
        ),
        alpha=ALPHA,
    )

    overlay2 = overlay_mask(
        img2,
        mask2,
        color_bgr=(
            255,
            0,
            0,
        ),
        alpha=ALPHA,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_mask_overlay.png"
        ),
        overlay1,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_mask_overlay.png"
        ),
        overlay2,
    )

    # ========================================================
    # 6. Mask difference
    # ========================================================

    comparison = create_comparison(
        mask1,
        mask2,
        alpha=ALPHA,
    )

    cv2.imwrite(
        str(
            output_dir
            / "mask_difference.png"
        ),
        comparison,
    )

    # ========================================================
    # 7. 两个 mask 半透明叠加
    # ========================================================

    two_overlay = (
        create_two_mask_overlay(
            mask1,
            mask2,
            alpha=ALPHA,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "two_masks_overlay.png"
        ),
        two_overlay,
    )

    # ========================================================
    # 8. Non-overlap residual
    # ========================================================

    only1 = (
        mask1
        & (~mask2)
    )

    only2 = (
        mask2
        & (~mask1)
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_only_mask.png"
        ),
        make_binary_mask(
            only1
        ),
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_only_mask.png"
        ),
        make_binary_mask(
            only2
        ),
    )

    # ========================================================
    # 9. Residual RGB
    # ========================================================

    residual1_transparent = (
        extract_nonoverlap_rgb(
            img1,
            only1,
        )
    )

    residual2_transparent = (
        extract_nonoverlap_rgb(
            img2,
            only2,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_only_residual_transparent.png"
        ),
        residual1_transparent,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_only_residual_transparent.png"
        ),
        residual2_transparent,
    )

    # --------------------------------------------------------
    # 黑底版本
    # --------------------------------------------------------

    residual1_black = (
        keep_only_nonoverlap_on_original(
            img1,
            only1,
        )
    )

    residual2_black = (
        keep_only_nonoverlap_on_original(
            img2,
            only2,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_only_residual.png"
        ),
        residual1_black,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_only_residual.png"
        ),
        residual2_black,
    )

    # ========================================================
    # 10. Residual overlay 到 resize 后原图
    # ========================================================

    # image1 独有 -> 红色
    image1_only_overlay = (
        overlay_nonoverlap(
            img1,
            only1,
            color_bgr=(
                0,
                0,
                255,
            ),
            alpha=RESIDUAL_ALPHA,
        )
    )

    # image2 独有 -> 蓝色
    image2_only_overlay = (
        overlay_nonoverlap(
            img2,
            only2,
            color_bgr=(
                255,
                0,
                0,
            ),
            alpha=RESIDUAL_ALPHA,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "image1_only_overlay.png"
        ),
        image1_only_overlay,
    )

    cv2.imwrite(
        str(
            output_dir
            / "image2_only_overlay.png"
        ),
        image2_only_overlay,
    )

    # ========================================================
    # 11. 两张图 residual 原始颜色合并
    # ========================================================

    combined_residual = (
        create_nonoverlap_rgb_combined(
            img1,
            img2,
            mask1,
            mask2,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "nonoverlap_rgb_combined.png"
        ),
        combined_residual,
    )

    # ========================================================
    # 12. 两边 residual 在统一底图显示
    # ========================================================

    residual_overlay_both = (
        create_nonoverlap_overlay_both(
            img1,
            img2,
            mask1,
            mask2,
            alpha=RESIDUAL_ALPHA,
        )
    )

    cv2.imwrite(
        str(
            output_dir
            / "nonoverlap_overlay_both.png"
        ),
        residual_overlay_both,
    )

    # ========================================================
    # 13. Mask metrics
    # ========================================================

    mask_metrics = (
        print_mask_metrics(
            mask1,
            mask2,
        )
    )

    # ========================================================
    # 14. 保存 metrics.txt
    # ========================================================

    metrics_path = (
        output_dir
        / "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "========== Input ==========\n"
        )

        f.write(
            f"Image 1 : "
            f"{image1_path}\n"
        )

        f.write(
            f"Image 2 : "
            f"{image2_path}\n"
        )

        f.write(
            f"Target width  : "
            f"{TARGET_W}\n"
        )

        f.write(
            f"Target height : "
            f"{TARGET_H}\n"
        )

        f.write(
            f"Target resolution : "
            f"{TARGET_W} x {TARGET_H}\n"
        )

        f.write(
            "Resize : enabled\n"
        )

        f.write(
            f"Black threshold : "
            f"{BLACK_THRESHOLD}\n"
        )

        f.write(
            "\n"
            "========== Foreground Image Metrics ==========\n"
        )

        if psnr is not None:

            f.write(
                f"Foreground PSNR : "
                f"{psnr:.6f} dB\n"
            )

            f.write(
                f"Foreground SSIM : "
                f"{ssim:.6f}\n"
            )

        else:

            f.write(
                "Foreground PSNR : N/A\n"
            )

            f.write(
                "Foreground SSIM : N/A\n"
            )

        f.write(
            "\n"
            "========== Mask Metrics ==========\n"
        )

        f.write(
            f"Image 1 object pixels : "
            f"{mask_metrics['area1']}\n"
        )

        f.write(
            f"Image 2 object pixels : "
            f"{mask_metrics['area2']}\n"
        )

        f.write(
            f"Overlap pixels        : "
            f"{mask_metrics['intersection']}\n"
        )

        f.write(
            f"Only image 1          : "
            f"{mask_metrics['only1']}\n"
        )

        f.write(
            f"Only image 2          : "
            f"{mask_metrics['only2']}\n"
        )

        f.write(
            f"Union pixels          : "
            f"{mask_metrics['union']}\n"
        )

        f.write(
            f"IoU                   : "
            f"{mask_metrics['iou']:.6f}\n"
        )

        f.write(
            f"Mask1 overlap ratio   : "
            f"{mask_metrics['coverage1']:.6f}\n"
        )

        f.write(
            f"Mask2 overlap ratio   : "
            f"{mask_metrics['coverage2']:.6f}\n"
        )

        f.write(
            f"Image1 residual ratio : "
            f"{mask_metrics['residual_ratio1']:.6f}\n"
        )

        f.write(
            f"Image2 residual ratio : "
            f"{mask_metrics['residual_ratio2']:.6f}\n"
        )

    # ========================================================
    # 完成
    # ========================================================

    print(
        f"结果已保存到: "
        f"{output_dir.resolve()}"
    )

    print(
        f"指标文件: "
        f"{metrics_path.resolve()}"
    )

    print(
        "\n"
        "重点查看以下结果："
    )

    print(
        "  image1_resized.png"
        "  -> resize 后图1"
    )

    print(
        "  image2_resized.png"
        "  -> resize 后图2"
    )

    print(
        "  mask_difference.png"
        "  -> 两张图 silhouette 差异"
    )

    print(
        "  image1_only_overlay.png"
        "  -> 图1独有区域标红"
    )

    print(
        "  image2_only_overlay.png"
        "  -> 图2独有区域标蓝"
    )

    print(
        "  image1_only_residual.png"
        "  -> 图1 residual 原始 RGB"
    )

    print(
        "  image2_only_residual.png"
        "  -> 图2 residual 原始 RGB"
    )

    print(
        "  nonoverlap_overlay_both.png"
        "  -> 两侧 residual 统一显示"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Resize two input images to a user-defined "
            "TARGET_W x TARGET_H resolution, "
            "compare foreground masks, calculate "
            "foreground PSNR/SSIM, and visualize "
            "non-overlapping residual regions."
        )
    )

    parser.add_argument(
        "image1",
        help="第一张图片路径",
    )

    parser.add_argument(
        "image2",
        help="第二张图片路径",
    )

    parser.add_argument(
        "--output",
        default="mask_compare_output",
        help="输出文件夹",
    )

    args = parser.parse_args()

    main(
        args.image1,
        args.image2,
        args.output,
    )