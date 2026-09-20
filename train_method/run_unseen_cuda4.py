"""Detached, sequential experiment from the original turtle image."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
GPU = "GPU-5a01b63c-14ed-235f-7936-8043e91e88a5"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONUNBUFFERED="1",
               PIXAL3D_LOW_MEMORY_DECODER="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", HF_HUB_OFFLINE="1",
               PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if not a.worker:
        if (out / "launch.json").exists() and not a.resume:
            raise RuntimeError("output already launched; choose a new directory")
        if a.resume and (out / "launch.json").exists():
            old_pid = json.loads((out / "launch.json").read_text())["pid"]
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"previous worker {old_pid} is still alive")
        with (out / "driver.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      "--output", str(out), "--worker"],
                                     cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=child.pid, resumed=a.resume, gpu=GPU, image=str(REPO / "assets/images/0_img.png"),
                      threshold=1.0, seeds=[48], n_values=[1, 2, 3, 4], output=str(out))
        (out / "launch.json").write_text(json.dumps(launch, indent=2))
        print(json.dumps(launch, indent=2))
        return
    assets = out / "input"
    assets.mkdir(exist_ok=True)
    link = assets / "0_img.png"
    if not link.exists():
        link.symlink_to(REPO / "assets/images/0_img.png")
    prepared = out / "prepared"
    root = prepared / "0_img/texture_explore"
    texture = out / "texture_experiments"
    core = [sys.executable, str(REPO / "sr_point_visibility_texture.py"),
            "--gpu", GPU, "--root", str(root), "--output-dir", str(texture),
            "--seeds", "48", "--visibility-threshold", "1.0", "--batch-size", "8",
            "--render-resolution", "1024"]
    stages = [
        ("prepare", [sys.executable, str(REPO / "sr_batch_experiment.py"),
                     "--phase", "prepare", "--assets", str(assets),
                     "--output-root", str(prepared), "--gpu", GPU]),
        ("texture_n1_n4", core + ["--n-values", "1,2,3,4"]),
        ("conditional_control", core + ["--n-values", "0", "--all-conditional"]),
    ]
    records = []
    start = time.time()
    try:
        for name, command in stages:
            print("START", name, command, flush=True)
            (out / "status.json").write_text(json.dumps(dict(status="RUNNING", stage=name, pid=os.getpid())))
            before = time.time()
            with (out / f"{name}.log").open("a") as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            records.append(dict(stage=name, seconds=time.time()-before))
            (out / "timing.json").write_text(json.dumps(records, indent=2))
        status = dict(status="COMPLETE", seconds=time.time()-start, stages=records)
    except Exception as exc:
        status = dict(status="FAILED", stage=name, error=repr(exc), stages=records)
        raise
    finally:
        (out / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
