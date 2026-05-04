"""
pipeline_workers.py
===================
Multi-stage image processing pipeline with a dedicated worker pool at every
stage.  This script brings together every concept covered in the course:

  ① Multi-stage pipeline             (task4.py pattern)
  ② Worker pool per stage            (task2.py pattern)
  ③ Producer-Consumer with Queue     (task3.py / Producer-Consumer.py)
  ④ Sequential vs Parallel timing   (main.py)

Architecture
────────────
                    Q_paths           Q_loaded          Q_gray
  [image paths] ──► [Stage-1 Workers] ──► [Stage-2 Workers] ──► [Stage-3 Workers] ──► [disk]
                     Load (I/O)             Resize+Gray+Blur       Canny+Save

Each stage runs WORKERS_PER_STAGE processes in parallel.
Shutdown uses a "poison-pill" (None sentinel) that each worker forwards to
the next queue, so the chain shuts down automatically.
"""

import os
import time
import multiprocessing

from main import generate_dataset
from pipeline import (
    process_image as _seq_process,
    stage_load, stage_resize, stage_grayscale,
    stage_blur, stage_canny, stage_save,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

WORKERS_PER_STAGE = 2   # processes at each pipeline stage
SENTINEL = None         # poison-pill: signals a worker to stop


# ── Stage workers ─────────────────────────────────────────────────────────────

def stage1_worker(q_in, q_out):
    """
    Stage 1 – I/O bound: read an image file from disk.

    Receives  : file path (str)
    Forwards  : (filename, BGR ndarray)
    """
    while True:
        path = q_in.get()
        if path is SENTINEL:
            q_out.put(SENTINEL)   # propagate one shutdown token downstream
            break
        item = stage_load(path)
        if item is not None:
            q_out.put(item)


def stage2_worker(q_in, q_out):
    """
    Stage 2 – CPU bound: resize → grayscale → Gaussian blur.

    Receives  : (filename, BGR ndarray)
    Forwards  : (filename, blurred grayscale ndarray)
    """
    while True:
        item = q_in.get()
        if item is SENTINEL:
            q_out.put(SENTINEL)   # propagate one shutdown token downstream
            break
        item = stage_resize(item)
        item = stage_grayscale(item)
        item = stage_blur(item)
        q_out.put(item)


def stage3_worker(q_in, output_dir):
    """
    Stage 3 – CPU + I/O: Canny edge detection → save result to disk.

    Receives  : (filename, blurred grayscale ndarray)
    Writes    : edge image to output_dir
    """
    while True:
        item = q_in.get()
        if item is SENTINEL:
            break
        item = stage_canny(item)
        stage_save(item, output_dir)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_workers(image_paths, workers_per_stage=WORKERS_PER_STAGE):
    """
    Spin up all three stages (workers_per_stage processes each), feed image
    paths, wait for every worker to finish.  Returns elapsed wall-clock time.

    Shutdown chain:
      Producer puts N sentinels → each Stage-1 worker receives 1, forwards 1
      → each Stage-2 worker receives 1, forwards 1
      → each Stage-3 worker receives 1 and exits.
    """
    q_paths  = multiprocessing.Queue()
    q_loaded = multiprocessing.Queue()
    q_gray   = multiprocessing.Queue()

    # Build worker pools for each stage
    stage1 = [
        multiprocessing.Process(target=stage1_worker, args=(q_paths, q_loaded))
        for _ in range(workers_per_stage)
    ]
    stage2 = [
        multiprocessing.Process(target=stage2_worker, args=(q_loaded, q_gray))
        for _ in range(workers_per_stage)
    ]
    stage3 = [
        multiprocessing.Process(target=stage3_worker, args=(q_gray, OUTPUT_DIR))
        for _ in range(workers_per_stage)
    ]

    # Start all workers before feeding work so every stage is ready
    for w in stage1 + stage2 + stage3:
        w.start()

    start = time.time()

    # Feed image paths into Stage 1
    for path in image_paths:
        q_paths.put(path)

    # One sentinel per Stage-1 worker kicks off the shutdown chain
    for _ in range(workers_per_stage):
        q_paths.put(SENTINEL)

    # Wait for all stages to drain and exit
    for w in stage1 + stage2 + stage3:
        w.join()

    return time.time() - start


# ── Sequential baseline ───────────────────────────────────────────────────────

def run_sequential(image_paths):
    """Process every image in a single loop (no concurrency)."""
    start = time.time()
    for path in image_paths:
        _seq_process(path, OUTPUT_DIR)
    return time.time() - start


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    generate_dataset()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_paths:
        print("No images found in dataset/. Run main.py first.")
        return

    total_workers = WORKERS_PER_STAGE * 3   # 3 stages

    print(f"\n{'='*58}")
    print(f"  Total Images      : {len(image_paths)}")
    print(f"  Pipeline Stages   : 3")
    print(f"    Stage 1 (I/O)   : Load image from disk")
    print(f"    Stage 2 (CPU)   : Resize → Grayscale → Blur")
    print(f"    Stage 3 (CPU+IO): Canny edges → Save to disk")
    print(f"  Workers / Stage   : {WORKERS_PER_STAGE}  ({total_workers} total workers)")
    print(f"{'='*58}")

    seq_time = run_sequential(image_paths)
    print(f"  Sequential        : {seq_time:.4f} sec")

    par_time = run_pipeline_workers(image_paths, workers_per_stage=WORKERS_PER_STAGE)
    print(f"  Pipeline Workers  : {par_time:.4f} sec  (workers/stage={WORKERS_PER_STAGE})")

    speedup    = seq_time / par_time if par_time > 0 else 0
    efficiency = (speedup / total_workers) * 100

    print(f"{'='*58}")
    print(f"  Speedup           : {speedup:.2f}x")
    print(f"  Efficiency        : {efficiency:.1f}%  (vs {total_workers} total workers)")
    print(f"  Results saved to  : '{OUTPUT_DIR}/'")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
