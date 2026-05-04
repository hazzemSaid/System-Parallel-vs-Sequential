"""
pipeline.py
===========
Reusable image-processing pipeline shared across all demo scripts.

Each stage is a pure function that operates on a (filename, img) tuple, so
stages can be composed freely or dropped into any parallel / pipeline
architecture.

Stage chain (default IMAGE_PIPELINE):
  stage_resize → stage_grayscale → stage_blur → stage_canny
"""

import cv2
import os


# ---------------------------------------------------------------------------
# Stage functions  –  (filename, img) → (filename, transformed_img)
# ---------------------------------------------------------------------------

def stage_load(path):
    """Stage 0 – load an image from disk. Returns (filename, img) or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    return (os.path.basename(path), img)


def stage_resize(item, size=(800, 800)):
    """Stage 1 – resize to a fixed resolution."""
    filename, img = item
    return (filename, cv2.resize(img, size))


def stage_grayscale(item):
    """Stage 2 – convert BGR image to grayscale."""
    filename, img = item
    return (filename, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def stage_blur(item, ksize=(9, 9), sigma=0):
    """Stage 3 – apply Gaussian blur."""
    filename, img = item
    return (filename, cv2.GaussianBlur(img, ksize, sigma))


def stage_canny(item, low=100, high=200):
    """Stage 4 – detect edges with the Canny algorithm."""
    filename, img = item
    return (filename, cv2.Canny(img, low, high))


def stage_save(item, output_dir):
    """Stage 5 – write the result image to *output_dir*."""
    filename, img = item
    cv2.imwrite(os.path.join(output_dir, filename), img)


# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------

IMAGE_PIPELINE = [stage_resize, stage_grayscale, stage_blur, stage_canny]


def run_pipeline(item, stages=None):
    """
    Apply each stage in *stages* (default: IMAGE_PIPELINE) to *item*.

    Parameters
    ----------
    item   : (filename: str, img: np.ndarray)
    stages : list of callables, each (item) -> item

    Returns
    -------
    (filename, processed_img)
    """
    if stages is None:
        stages = IMAGE_PIPELINE
    for stage in stages:
        item = stage(item)
    return item


def process_image(path, output_dir):
    """
    Convenience wrapper: load → run_pipeline → save.

    Safe to use with multiprocessing.Pool via functools.partial:
        from functools import partial
        fn = partial(process_image, output_dir=OUTPUT_DIR)
        pool.map(fn, image_paths)
    """
    item = stage_load(path)
    if item is None:
        print(f"[WARN] Could not read: {path}")
        return
    result = run_pipeline(item)
    stage_save(result, output_dir)
