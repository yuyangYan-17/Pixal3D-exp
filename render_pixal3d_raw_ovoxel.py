#!/usr/bin/env python3
"""Render a Pixal3D ``MeshWithVoxel`` with the official nvdiffrast renderer.

This module keeps the decoder output in its native representation:

    MeshWithVoxel
      -> pixal3d.utils.render_utils.render_frames
      -> PbrMeshRenderer
      -> sparse O-Voxel trilinear lookup
      -> PNG images

There is no GLB export, UV unwrap, texture atlas, Blender process, or Cycles
shader reconstruction.  The public functions are intended to be called
directly after ``pipeline.decode_latent(...)``.  The CLI is provided for
previously saved ``MeshWithVoxel`` PyTorch checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pixal3d.renderers import EnvMap
from pixal3d.representations import MeshWithVoxel
from pixal3d.utils import render_utils


DEFAULT_RENDER_MODES = (
    "shaded",
    "base_color",
    "normal",
    "metallic",
    "roughness",
    "alpha",
)
DEFAULT_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON without exposing a partially written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def composite_on_black(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def image_to_tensor(
    image: Image.Image,
    size: Tuple[int, int],
) -> torch.Tensor:
    rgb = image.convert("RGB")
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def psnr_metric(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    mse = F.mse_loss(prediction, reference).item()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def _gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device)
    coords = coords - (window_size - 1) / 2.0
    gaussian = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim_metric(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    channels = int(reference.shape[1])
    kernel = _gaussian_kernel(
        window_size,
        sigma,
        channels,
        reference.dtype,
        reference.device,
    )
    padding = window_size // 2
    mu_ref = F.conv2d(reference, kernel, padding=padding, groups=channels)
    mu_pred = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_ref_sq = mu_ref.square()
    mu_pred_sq = mu_pred.square()
    mu_cross = mu_ref * mu_pred
    sigma_ref = (
        F.conv2d(reference.square(), kernel, padding=padding, groups=channels)
        - mu_ref_sq
    )
    sigma_pred = (
        F.conv2d(prediction.square(), kernel, padding=padding, groups=channels)
        - mu_pred_sq
    )
    sigma_cross = (
        F.conv2d(reference * prediction, kernel, padding=padding, groups=channels)
        - mu_cross
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    ) / (
        (mu_ref_sq + mu_pred_sq + c1)
        * (sigma_ref + sigma_pred + c2)
    )
    return float(score.mean().item())


class LPIPSEvaluator:
    def __init__(self, network: str, device: torch.device):
        try:
            import lpips
        except Exception as exc:
            raise RuntimeError(
                "LPIPS was requested but the lpips package is unavailable"
            ) from exc
        self.device = device
        self.model = lpips.LPIPS(net=network).eval().to(device)

    @torch.no_grad()
    def evaluate(
        self,
        reference: torch.Tensor,
        prediction: torch.Tensor,
    ) -> float:
        reference = reference.to(self.device) * 2.0 - 1.0
        prediction = prediction.to(self.device) * 2.0 - 1.0
        return float(self.model(reference, prediction).item())


def _resolve_envmap_path(envmap: Union[str, Path]) -> Path:
    candidate = Path(envmap).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    name = str(envmap)
    if name.lower().endswith(".exr"):
        filename = name
    else:
        filename = f"{name}.exr"
    path = Path(__file__).resolve().parent / "assets" / "hdri" / filename
    if not path.is_file():
        available = sorted(item.stem for item in path.parent.glob("*.exr"))
        raise FileNotFoundError(
            f"environment map {envmap!r} was not found; "
            f"available bundled maps: {', '.join(available)}"
        )
    return path


def load_envmap(
    envmap: Union[str, Path] = "studio",
    *,
    device: Union[str, torch.device] = "cuda",
) -> EnvMap:
    """Load a bundled-name or explicit EXR path as Pixal3D's ``EnvMap``."""
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to read Pixal3D EXR envmaps") from exc

    path = _resolve_envmap_path(envmap)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        # The pip OpenCV wheels used by Pixal3D may be built without OpenEXR.
        # imageio-ffmpeg is already an official repository dependency and its
        # bundled ffmpeg preserves the EXR's planar float32 HDR values.
        try:
            import imageio_ffmpeg
        except Exception as exc:
            raise RuntimeError(
                f"OpenCV cannot decode {path} and imageio-ffmpeg is unavailable"
            ) from exc
        capture = cv2.VideoCapture(str(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width < 1 or height < 1:
            raise RuntimeError(f"failed to inspect environment map: {path}")
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gbrpf32le",
            "pipe:1",
        ]
        process = subprocess.run(command, capture_output=True, check=False)
        if process.returncode != 0:
            error = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"failed to decode environment map {path}: {error}")
        planar = np.frombuffer(process.stdout, dtype="<f4")
        expected = 3 * width * height
        if planar.size != expected:
            raise RuntimeError(
                f"unexpected decoded envmap size: {planar.size} vs {expected}"
            )
        # gbrpf32le stores one G plane, then B, then R.
        planes = planar.reshape(3, height, width)
        image_rgb = np.ascontiguousarray(
            planes[[2, 0, 1]].transpose(1, 2, 0)
        )
        print(f"[envmap] OpenCV EXR unavailable; decoded HDR via ffmpeg: {path}")
    tensor = torch.as_tensor(
        np.ascontiguousarray(image_rgb),
        dtype=torch.float32,
        device=device,
    )
    result = EnvMap(tensor)
    result.source_path = str(path)
    return result


def _validate_mesh(mesh: Any) -> MeshWithVoxel:
    if not isinstance(mesh, MeshWithVoxel):
        raise TypeError(
            "official O-Voxel rendering requires pixal3d.representations."
            f"MeshWithVoxel, got {type(mesh)!r}"
        )
    required = (
        "vertices",
        "faces",
        "coords",
        "attrs",
        "origin",
        "voxel_size",
        "voxel_shape",
        "layout",
    )
    missing = [name for name in required if not hasattr(mesh, name)]
    if missing:
        raise ValueError(f"MeshWithVoxel is missing: {', '.join(missing)}")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must have shape [N, 3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("mesh.faces must have shape [M, 3]")
    if mesh.coords.ndim != 2 or mesh.coords.shape[1] != 3:
        raise ValueError("mesh.coords must have shape [L, 3]")
    if mesh.attrs.ndim != 2 or mesh.attrs.shape[0] != mesh.coords.shape[0]:
        raise ValueError("mesh.attrs must have shape [L, C] aligned with coords")
    return mesh


@torch.no_grad()
def render_static_ovoxel(
    mesh: MeshWithVoxel,
    *,
    camera_angle_x: float,
    distance: float,
    resolution: int = 1024,
    envmap: Union[EnvMap, str, Path] = "studio",
    near: Optional[float] = None,
    far: Optional[float] = None,
    ssaa: int = 2,
    peel_layers: int = 8,
    face_chunk_size: int = 0,
    use_envmap_bg: bool = False,
    verbose: bool = True,
) -> Dict[str, Sequence[np.ndarray]]:
    """Render one projection-aligned camera through Pixal3D's official API."""
    mesh = _validate_mesh(mesh)
    if (
        resolution < 1
        or ssaa < 1
        or peel_layers < 1
        or face_chunk_size < 0
    ):
        raise ValueError(
            "resolution, ssaa, and peel_layers must be positive and "
            "face_chunk_size must be non-negative"
        )
    if camera_angle_x <= 0.0 or distance <= 0.0:
        raise ValueError("camera_angle_x and distance must be positive")
    if mesh.device.type != "cuda":
        mesh = mesh.cuda()
    if not isinstance(envmap, EnvMap):
        envmap = load_envmap(envmap, device=mesh.device)
    elif envmap.image.device != mesh.device:
        source_path = getattr(envmap, "source_path", None)
        envmap = EnvMap(envmap.image.to(mesh.device))
        if source_path is not None:
            envmap.source_path = source_path

    near_value = max(0.01, float(distance) - 2.0) if near is None else float(near)
    far_value = float(distance) + 10.0 if far is None else float(far)
    if not (0.0 < near_value < far_value):
        raise ValueError(f"invalid clipping range: near={near_value}, far={far_value}")

    with torch.cuda.device(mesh.device):
        extrinsics, intrinsics = render_utils.proj_camera_to_render_params(
            camera_angle_x=float(camera_angle_x),
            distance=float(distance),
        )
        return render_utils.render_frames(
            mesh,
            extrinsics=[extrinsics],
            intrinsics=[intrinsics],
            options={
                "resolution": int(resolution),
                "near": near_value,
                "far": far_value,
                "ssaa": int(ssaa),
                "peel_layers": int(peel_layers),
                "face_chunk_size": int(face_chunk_size),
            },
            verbose=verbose,
            envmap=envmap,
            use_envmap_bg=bool(use_envmap_bg),
        )


def save_render_outputs(
    renders: Mapping[str, Sequence[np.ndarray]],
    output_dir: Path,
    *,
    modes: Sequence[str] = DEFAULT_RENDER_MODES,
) -> Dict[str, str]:
    """Save the first (and only) frame for each requested official render mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for mode in modes:
        frames = renders.get(mode)
        if not frames:
            continue
        path = output_dir / f"{mode}.png"
        Image.fromarray(np.asarray(frames[0])).save(path)
        paths[mode] = str(path)
    if "shaded" not in paths:
        available = ", ".join(sorted(renders))
        raise KeyError(f"official renderer did not return 'shaded'; got: {available}")
    render_path = output_dir / "render.png"
    Image.open(paths["shaded"]).convert("RGB").save(render_path)
    paths["render"] = str(render_path)
    return paths


def save_comparison(
    reference: Image.Image,
    rendered: Image.Image,
    output_path: Path,
) -> None:
    if reference.size != rendered.size:
        raise ValueError("reference and rendered images must have equal sizes")
    canvas = Image.new("RGB", (reference.width * 2, reference.height))
    canvas.paste(reference.convert("RGB"), (0, 0))
    canvas.paste(rendered.convert("RGB"), (reference.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_and_evaluate_mesh(
    mesh: MeshWithVoxel,
    *,
    camera_angle_x: float,
    distance: float,
    output_dir: Path,
    reference_image: Optional[Path] = None,
    resolution: int = 1024,
    metric_resolution: int = 1024,
    envmap: Union[EnvMap, str, Path] = "studio",
    envmap_name: Optional[str] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
    ssaa: int = 2,
    peel_layers: int = 8,
    face_chunk_size: int = 0,
    use_envmap_bg: bool = False,
    lpips_net: str = "vgg",
    metric_device: str = "cuda",
    skip_lpips: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Render a live decoder output, save maps, and optionally compute metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    renders = render_static_ovoxel(
        mesh,
        camera_angle_x=camera_angle_x,
        distance=distance,
        resolution=resolution,
        envmap=envmap,
        near=near,
        far=far,
        ssaa=ssaa,
        peel_layers=peel_layers,
        face_chunk_size=face_chunk_size,
        use_envmap_bg=use_envmap_bg,
        verbose=verbose,
    )
    finished.record()
    torch.cuda.synchronize()
    render_seconds = float(started.elapsed_time(finished) / 1000.0)
    output_paths = save_render_outputs(renders, output_dir)

    row: Dict[str, Any] = {
        "status": "success",
        "renderer": "pixal3d.utils.render_utils.render_frames",
        "sample_type": "MeshWithVoxel",
        "material_source": "official sparse O-Voxel surface-position lookup",
        "sampling_mode": "grid_sample_3d trilinear",
        "envmap": (
            envmap_name
            if envmap_name is not None
            else getattr(envmap, "source_path", str(envmap))
        ),
        "render_resolution": int(resolution),
        "metric_resolution": int(metric_resolution),
        "camera_angle_x": float(camera_angle_x),
        "distance": float(distance),
        "near": max(0.01, float(distance) - 2.0) if near is None else float(near),
        "far": float(distance) + 10.0 if far is None else float(far),
        "ssaa": int(ssaa),
        "peel_layers": int(peel_layers),
        "face_chunk_size": int(face_chunk_size),
        "use_envmap_bg": bool(use_envmap_bg),
        "decoder_vertices": int(mesh.vertices.shape[0]),
        "decoder_faces": int(mesh.faces.shape[0]),
        "active_voxels": int(mesh.coords.shape[0]),
        "render_seconds": render_seconds,
        "render_png": output_paths["render"],
        "render_outputs": output_paths,
        "original_png": None,
        "comparison_png": None,
        "psnr_db": None,
        "ssim": None,
        "lpips": None,
        "error": None,
    }

    if reference_image is not None:
        reference_path = Path(reference_image)
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        with Image.open(reference_path) as source:
            reference = composite_on_black(source)
        target_size = (int(resolution), int(resolution))
        if reference.size != target_size:
            reference = reference.resize(target_size, Image.Resampling.LANCZOS)
        rendered = Image.open(output_paths["render"]).convert("RGB")
        original_path = output_dir / "original.png"
        comparison_path = output_dir / "comparison.png"
        reference.save(original_path)
        save_comparison(reference, rendered, comparison_path)

        metric_size = (int(metric_resolution), int(metric_resolution))
        reference_tensor = image_to_tensor(reference, metric_size)
        prediction_tensor = image_to_tensor(rendered, metric_size)
        lpips_value: Optional[float] = None
        if not skip_lpips:
            device = torch.device(
                metric_device
                if metric_device.startswith("cuda") and torch.cuda.is_available()
                else "cpu"
            )
            evaluator = LPIPSEvaluator(lpips_net, device)
            lpips_value = evaluator.evaluate(reference_tensor, prediction_tensor)
            evaluator.model.cpu()
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row.update(
            {
                "original_png": str(original_path),
                "comparison_png": str(comparison_path),
                "psnr_db": psnr_metric(reference_tensor, prediction_tensor),
                "ssim": ssim_metric(reference_tensor, prediction_tensor),
                "lpips": lpips_value,
            }
        )

    atomic_json(output_dir / "metrics.json", row)
    return row


def _normalize_layout(layout: Any) -> Dict[str, slice]:
    if not isinstance(layout, Mapping):
        return dict(DEFAULT_ATTR_LAYOUT)
    result: Dict[str, slice] = {}
    for name, default in DEFAULT_ATTR_LAYOUT.items():
        value = layout.get(name, default)
        if isinstance(value, slice):
            result[name] = value
        elif isinstance(value, Sequence) and len(value) == 2:
            result[name] = slice(int(value[0]), int(value[1]))
        else:
            raise ValueError(f"invalid attribute layout for {name!r}: {value!r}")
    return result


def load_mesh_checkpoint(
    path: Path,
    *,
    device: Union[str, torch.device] = "cuda",
) -> MeshWithVoxel:
    """Load a ``MeshWithVoxel`` object or a mapping of its native tensors."""
    path = Path(path).expanduser().resolve()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and isinstance(
        payload.get("mesh"),
        MeshWithVoxel,
    ):
        payload = payload["mesh"]
    if isinstance(payload, MeshWithVoxel):
        return payload.to(device)
    if not isinstance(payload, Mapping):
        raise TypeError(
            "checkpoint must contain MeshWithVoxel or a mapping of native fields"
        )
    required = (
        "vertices",
        "faces",
        "origin",
        "voxel_size",
        "coords",
        "attrs",
        "voxel_shape",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"mesh checkpoint is missing: {', '.join(missing)}")
    origin = torch.as_tensor(payload["origin"]).reshape(-1).tolist()
    voxel_size_tensor = torch.as_tensor(payload["voxel_size"]).reshape(-1)
    voxel_size: Any
    if voxel_size_tensor.numel() == 1:
        voxel_size = float(voxel_size_tensor.item())
    elif voxel_size_tensor.numel() == 3:
        voxel_size = voxel_size_tensor.to(device)
    else:
        raise ValueError("voxel_size must be scalar or length three")
    mesh = MeshWithVoxel(
        vertices=torch.as_tensor(payload["vertices"]),
        faces=torch.as_tensor(payload["faces"]),
        origin=origin,
        voxel_size=voxel_size,
        coords=torch.as_tensor(payload["coords"]),
        attrs=torch.as_tensor(payload["attrs"]),
        voxel_shape=torch.Size(payload["voxel_shape"]),
        layout=_normalize_layout(payload.get("layout")),
    )
    return mesh.to(device)


def _load_camera(args: argparse.Namespace) -> Tuple[float, float]:
    camera_angle_x = args.camera_angle_x
    distance = args.distance
    if args.camera_json is not None:
        payload = json.loads(args.camera_json.read_text(encoding="utf-8"))
        if isinstance(payload.get("camera"), Mapping):
            payload = payload["camera"]
        if camera_angle_x is None:
            camera_angle_x = payload.get("camera_angle_x")
        if distance is None:
            distance = payload.get("distance")
    if camera_angle_x is None or distance is None:
        raise ValueError(
            "provide --camera-angle-x and --distance, or a --camera-json "
            "containing those fields"
        )
    return float(camera_angle_x), float(distance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, default=None)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--camera-angle-x", type=float, default=None)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="visible CUDA device index; omitted respects the current CUDA environment",
    )
    parser.add_argument(
        "--envmap",
        default="studio",
        help="bundled HDRI name (for example studio) or an EXR path",
    )
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--metric-resolution", type=int, default=1024)
    parser.add_argument("--ssaa", type=int, default=2)
    parser.add_argument("--peel-layers", type=int, default=8)
    parser.add_argument(
        "--face-chunk-size",
        type=int,
        default=0,
        help=(
            "maximum faces per nvdiffrast call; zero uses one full-mesh call"
        ),
    )
    parser.add_argument(
        "--use-envmap-bg",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
    )
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument(
        "--skip-lpips",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.render_resolution < 1
        or args.metric_resolution < 1
        or args.ssaa < 1
        or args.peel_layers < 1
        or args.face_chunk_size < 0
    ):
        raise ValueError(
            "render resolution, metric resolution, ssaa, and peel layers "
            "must be positive; face chunk size must be non-negative"
        )
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            raise ValueError("--cuda-device must be non-negative")
        torch.cuda.set_device(int(args.cuda_device))
    camera_angle_x, distance = _load_camera(args)
    mesh = load_mesh_checkpoint(args.mesh_checkpoint, device="cuda")
    envmap = load_envmap(args.envmap, device="cuda")
    try:
        result = render_and_evaluate_mesh(
            mesh,
            camera_angle_x=camera_angle_x,
            distance=distance,
            output_dir=args.output_dir.expanduser().resolve(),
            reference_image=(
                None
                if args.reference_image is None
                else args.reference_image.expanduser().resolve()
            ),
            resolution=int(args.render_resolution),
            metric_resolution=int(args.metric_resolution),
            envmap=envmap,
            envmap_name=str(args.envmap),
            ssaa=int(args.ssaa),
            peel_layers=int(args.peel_layers),
            face_chunk_size=int(args.face_chunk_size),
            use_envmap_bg=bool(args.use_envmap_bg),
            lpips_net=str(args.lpips_net),
            metric_device=str(args.metric_device),
            skip_lpips=bool(args.skip_lpips),
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(
        f"[done] renderer={result['renderer']} "
        f"render={result['render_png']} metrics={args.output_dir / 'metrics.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
