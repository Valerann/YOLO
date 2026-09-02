"""Profile yolo.compat's load() and detect_cpu() to find where time actually
goes. Run this on the real deployment target (e.g. inside the ECS Fargate
task) -- timing on a dev machine won't transfer 1:1 to different CPU
hardware, but this script is meant to be dropped in and run wherever the
slowness is actually observed.

Usage:
    python3 profile_compat.py <weights_path> <image_path> [img_size] [model_variant] [n_iters]

Example:
    python3 profile_compat.py /weights/v7x_converted.pt cars.jpg 416 v7x 5
"""
import cProfile
import io
import pstats
import sys
import time

import numpy as np
from PIL import Image

import yolo.compat as yolov7


def report(profiler: cProfile.Profile, label: str, top_n: int = 25) -> None:
    print(f"\n{'=' * 20} {label}: by cumulative time {'=' * 20}")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)
    print(stream.getvalue())

    print(f"{'=' * 20} {label}: by internal (self) time {'=' * 20}")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(top_n)
    print(stream.getvalue())


def main() -> None:
    weights_path = sys.argv[1]
    image_path = sys.argv[2]
    img_size = int(sys.argv[3]) if len(sys.argv) > 3 else 416
    model_variant = sys.argv[4] if len(sys.argv) > 4 else "v7x"
    n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    print(f"torch threads: {__import__('torch').get_num_threads()}", file=sys.stderr)

    load_profiler = cProfile.Profile()
    load_profiler.enable()
    model = yolov7.load(weights_path, model_variant=model_variant, size=img_size)
    load_profiler.disable()
    report(load_profiler, "load()")

    image = np.array(Image.open(image_path).convert("RGB"))

    # Warmup call, excluded from profiling -- first call pays one-off costs
    # (lazy imports, cudnn/mkldnn algo selection, etc.) that a warm service
    # process wouldn't repeat per-request.
    yolov7.detect_cpu(model, image, img_size=img_size, nms_agnostic=True)

    detect_profiler = cProfile.Profile()
    wall_times = []
    for _ in range(n_iters):
        t0 = time.time()
        detect_profiler.enable()
        det = yolov7.detect_cpu(model, image, img_size=img_size, nms_agnostic=True)
        detect_profiler.disable()
        wall_times.append(time.time() - t0)

    print(f"\ndetect_cpu() wall times over {n_iters} calls: "
          f"{[f'{t * 1000:.1f}ms' for t in wall_times]}")
    print(f"mean: {sum(wall_times) / len(wall_times) * 1000:.1f}ms, "
          f"detections: {det.shape[0]}")
    report(detect_profiler, f"detect_cpu() [{n_iters} calls combined]")


if __name__ == "__main__":
    main()
