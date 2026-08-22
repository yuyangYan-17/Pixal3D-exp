import argparse
from pathlib import Path

import cv2
from flask import Flask, Response


# ============================================================
# Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# 全局图像
# ============================================================

IMAGE1 = None
IMAGE2 = None

IMAGE_WIDTH = None
IMAGE_HEIGHT = None


# ============================================================
# 读取图片
# ============================================================

def load_image(path):
    """
    直接读取图片。

    不进行任何 resize。
    保持文件本身的原始像素尺寸。
    """

    img = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise FileNotFoundError(
            f"无法读取图片: {path}"
        )

    return img


# ============================================================
# PNG 编码
# ============================================================

def encode_png(img):
    """
    OpenCV image -> PNG bytes
    """

    ok, buffer = cv2.imencode(
        ".png",
        img,
    )

    if not ok:
        raise RuntimeError(
            "PNG 编码失败"
        )

    return buffer.tobytes()


# ============================================================
# 图片接口
# ============================================================

@app.route("/image1")
def get_image1():
    return Response(
        encode_png(IMAGE1),
        mimetype="image/png",
    )


@app.route("/image2")
def get_image2():
    return Response(
        encode_png(IMAGE2),
        mimetype="image/png",
    )


# ============================================================
# 网页
# ============================================================

@app.route("/")
def index():

    html = """
<!DOCTYPE html>

<html lang="zh-CN">

<head>

<meta charset="UTF-8">

<title>Image Overlay Viewer</title>

<style>

    * {
        box-sizing: border-box;
    }


    body {
        margin: 0;
        padding: 24px;

        background: #171717;
        color: #eeeeee;

        font-family:
            Arial,
            Helvetica,
            sans-serif;
    }


    h1 {
        margin-top: 0;
        margin-bottom: 20px;

        font-size: 24px;
    }


    .main {
        width: 100%;
        max-width: 1800px;

        margin: auto;
    }


    .controls {
        padding: 18px;

        margin-bottom: 20px;

        background: #252525;

        border:
            1px solid #444;

        border-radius: 8px;
    }


    .row {
        display: flex;

        align-items: center;

        gap: 15px;

        margin-bottom: 14px;
    }


    .row:last-child {
        margin-bottom: 0;
    }


    .label {
        width: 160px;

        font-weight: bold;
    }


    input[type="range"] {
        width: 500px;
        max-width: 50vw;
    }


    .value {
        width: 60px;

        font-family: monospace;

        font-size: 16px;
    }


    button {
        padding: 8px 14px;

        background: #333;
        color: white;

        border:
            1px solid #555;

        border-radius: 5px;

        cursor: pointer;
    }


    button:hover {
        background: #444;
    }


    .viewer-container {
        display: flex;

        justify-content: center;

        width: 100%;
    }


    .viewer {
        position: relative;

        /*
        不再写死 1024x1024。

        宽度根据浏览器自动缩放，
        高宽比使用实际输入图片比例。
        */
        width:
            min(1600px, 96vw);

        aspect-ratio:
            __IMAGE_WIDTH__ / __IMAGE_HEIGHT__;

        overflow: hidden;

        border:
            1px solid #555;

        background: black;
    }


    .layer {
        position: absolute;

        left: 0;
        top: 0;

        width: 100%;
        height: 100%;

        /*
        两张图尺寸完全相同，因此直接严格重叠。
        */
        object-fit: fill;

        pointer-events: none;

        user-select: none;
        -webkit-user-drag: none;
    }


    #img1 {
        z-index: 1;

        opacity: 0.5;
    }


    #img2 {
        z-index: 2;

        opacity: 0.5;
    }


    .info {
        margin-top: 12px;

        text-align: center;

        font-family: monospace;

        color: #bbbbbb;
    }


    .resolution {
        margin-bottom: 16px;

        font-family: monospace;

        color: #aaaaaa;
    }

</style>

</head>


<body>

<div class="main">

    <h1>
        Image Overlay Viewer
    </h1>


    <div class="resolution">
        Image resolution:
        __IMAGE_WIDTH__ x __IMAGE_HEIGHT__
        &nbsp; | &nbsp;
        No resize
    </div>


    <div class="controls">


        <!-- Image 1 opacity -->

        <div class="row">

            <div class="label">
                Image 1 opacity
            </div>

            <input
                id="opacity1"
                type="range"
                min="0"
                max="100"
                value="50"
            >

            <div
                id="value1"
                class="value"
            >
                50%
            </div>

        </div>


        <!-- Image 2 opacity -->

        <div class="row">

            <div class="label">
                Image 2 opacity
            </div>

            <input
                id="opacity2"
                type="range"
                min="0"
                max="100"
                value="50"
            >

            <div
                id="value2"
                class="value"
            >
                50%
            </div>

        </div>


        <!-- 快捷按钮 -->

        <div class="row">

            <div class="label">
                Quick view
            </div>

            <button id="half">
                50 / 50
            </button>

            <button id="mostly1">
                80 / 20
            </button>

            <button id="mostly2">
                20 / 80
            </button>

            <button id="only1">
                Only image 1
            </button>

            <button id="only2">
                Only image 2
            </button>

        </div>


        <!-- Layer order -->

        <div class="row">

            <div class="label">
                Layer order
            </div>

            <button id="swap">
                Image 2 on top
            </button>

        </div>

    </div>


    <!-- ===================================================
         两张图严格同坐标重叠
         =================================================== -->

    <div class="viewer-container">

        <div
            id="viewer"
            class="viewer"
        >

            <img
                id="img1"
                class="layer"
                src="/image1"
            >

            <img
                id="img2"
                class="layer"
                src="/image2"
            >

        </div>

    </div>


    <div
        id="info"
        class="info"
    >
        image1 = 50% | image2 = 50%
    </div>

</div>


<script>

const img1 =
    document.getElementById(
        "img1"
    );

const img2 =
    document.getElementById(
        "img2"
    );


const opacity1 =
    document.getElementById(
        "opacity1"
    );

const opacity2 =
    document.getElementById(
        "opacity2"
    );


const value1 =
    document.getElementById(
        "value1"
    );

const value2 =
    document.getElementById(
        "value2"
    );


const info =
    document.getElementById(
        "info"
    );


function updateOpacity() {

    const v1 =
        Number(
            opacity1.value
        );

    const v2 =
        Number(
            opacity2.value
        );


    img1.style.opacity =
        v1 / 100.0;

    img2.style.opacity =
        v2 / 100.0;


    value1.textContent =
        v1 + "%";

    value2.textContent =
        v2 + "%";


    info.textContent =
        "image1 = "
        + v1
        + "% | image2 = "
        + v2
        + "%";
}


opacity1.addEventListener(
    "input",
    updateOpacity
);


opacity2.addEventListener(
    "input",
    updateOpacity
);


function setOpacity(
    v1,
    v2
) {

    opacity1.value =
        v1;

    opacity2.value =
        v2;

    updateOpacity();
}


document
    .getElementById(
        "half"
    )
    .addEventListener(
        "click",
        () => setOpacity(
            50,
            50
        )
    );


document
    .getElementById(
        "mostly1"
    )
    .addEventListener(
        "click",
        () => setOpacity(
            80,
            20
        )
    );


document
    .getElementById(
        "mostly2"
    )
    .addEventListener(
        "click",
        () => setOpacity(
            20,
            80
        )
    );


document
    .getElementById(
        "only1"
    )
    .addEventListener(
        "click",
        () => setOpacity(
            100,
            0
        )
    );


document
    .getElementById(
        "only2"
    )
    .addEventListener(
        "click",
        () => setOpacity(
            0,
            100
        )
    );


let image2OnTop =
    true;


document
    .getElementById(
        "swap"
    )
    .addEventListener(
        "click",
        function() {

            image2OnTop =
                !image2OnTop;


            if (
                image2OnTop
            ) {

                img1.style.zIndex =
                    1;

                img2.style.zIndex =
                    2;

                this.textContent =
                    "Image 2 on top";

            } else {

                img1.style.zIndex =
                    2;

                img2.style.zIndex =
                    1;

                this.textContent =
                    "Image 1 on top";
            }
        }
    );


updateOpacity();

</script>

</body>

</html>
"""

    # --------------------------------------------------------
    # 使用真实图片尺寸设置网页 viewer 比例
    # --------------------------------------------------------

    html = html.replace(
        "__IMAGE_WIDTH__",
        str(IMAGE_WIDTH),
    )

    html = html.replace(
        "__IMAGE_HEIGHT__",
        str(IMAGE_HEIGHT),
    )

    return html


# ============================================================
# Main
# ============================================================

def main():

    global IMAGE1
    global IMAGE2

    global IMAGE_WIDTH
    global IMAGE_HEIGHT


    parser = argparse.ArgumentParser(
        description=(
            "Overlay two already-aligned images "
            "in a browser with independently "
            "adjustable opacity. "
            "No image resizing is performed."
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
        "--port",
        type=int,
        default=8000,
        help="网页服务端口，默认 8000",
    )


    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址，默认 0.0.0.0",
    )


    args = parser.parse_args()


    image1_path = Path(
        args.image1
    )

    image2_path = Path(
        args.image2
    )


    # ========================================================
    # 读取 image1
    # ========================================================

    print(
        "\nLoading image 1:"
    )

    print(
        image1_path
    )

    IMAGE1 = load_image(
        image1_path
    )


    # ========================================================
    # 读取 image2
    # ========================================================

    print(
        "\nLoading image 2:"
    )

    print(
        image2_path
    )

    IMAGE2 = load_image(
        image2_path
    )


    # ========================================================
    # 检查尺寸
    # ========================================================

    h1, w1 = (
        IMAGE1.shape[:2]
    )

    h2, w2 = (
        IMAGE2.shape[:2]
    )


    print(
        "\n========== Image Info =========="
    )

    print(
        f"Image 1: "
        f"{w1} x {h1}"
    )

    print(
        f"Image 2: "
        f"{w2} x {h2}"
    )


    # --------------------------------------------------------
    # 不允许网页程序自行 resize。
    #
    # 如果不一致，
    # 应该回到之前的 process.py 先生成统一尺寸结果。
    # --------------------------------------------------------

    if (
        w1 != w2
        or h1 != h2
    ):

        raise ValueError(
            "\n两张图片尺寸不同，"
            "无法严格逐像素重叠。\n"
            "\n"
            f"Image 1: "
            f"{w1} x {h1}\n"
            f"Image 2: "
            f"{w2} x {h2}\n"
            "\n"
            "本程序不会 resize。\n"
            "请读取之前 process.py "
            "生成的统一尺寸结果，例如：\n"
            "\n"
            "  mask_compare_output/"
            "image1_resized.png\n"
            "  mask_compare_output/"
            "image2_resized.png\n"
        )


    IMAGE_WIDTH = w1
    IMAGE_HEIGHT = h1


    print(
        "Resize : disabled"
    )

    print(
        "Pixel alignment : 1-to-1"
    )

    print(
        "================================\n"
    )


    # ========================================================
    # Server
    # ========================================================

    print(
        "Server:"
    )

    print(
        f"http://127.0.0.1:"
        f"{args.port}"
    )

    print()

    print(
        "如果在远程服务器上，"
        "请使用 SSH port forwarding。"
    )

    print()


    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()