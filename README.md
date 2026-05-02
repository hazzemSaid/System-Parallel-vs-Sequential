# Parallel Image Processing Demo

This project demonstrates sequential vs parallel image processing in Python using OpenCV and the multiprocessing module.

## What It Does

The script in [main.py](main%20%282%29.py) does the following:

- Reads images from the `dataset/` folder.
- Resizes each image to `800 x 800`.
- Converts each image to grayscale.
- Repeats a blur + edge-detection workload several times to simulate heavier processing.
- Saves the processed edge image into the `results/` folder.
- Measures how long the work takes in sequential mode and in parallel mode.

If `dataset/` is empty, the script can generate synthetic test images automatically.

## Folder Layout

- `dataset/` - Input images.
- `results/` - Output images created by the script.
- `main.py` - Main script.

## Requirements

- Python 3
- `opencv-python`
- `numpy`

Install dependencies with:

```bash
pip install opencv-python numpy
```

## How To Use

Run the script from the project root:

```bash
python "main.py"
```

You do not need to move into `dataset/` or `results/`. The script now resolves both folders relative to the script location.

## What Output To Expect

When you run it, the script prints a short performance report similar to this:

```text
Dataset already exists: 200 images found, skipping generation.

=============================================
  Total Images : 200
  CPU Cores    : 12
  Workload     : 10 blur/canny passes per image
=============================================
  Sequential   : 5.5189 sec
  Parallel     : 2.8377 sec  (processes=4, chunksize=5)
=============================================
  Speedup      : 1.94x
  Efficiency   : 48.6%  (vs 4 processes)
  Results saved to 'e:\dev\paraller\results/'
=============================================
```

Each line comes from a specific part of the script:

- `Dataset already exists...` comes from `generate_dataset()`, which skips synthetic image creation if `dataset/` already has images.
- `Total Images` is the number of image files found in `dataset/`.
- `CPU Cores` comes from `cpu_count()`.
- `Workload` shows the value of `WORKLOAD_PASSES`, the number of blur/Canny passes applied to each image.
- `Sequential` is the time for processing all images one by one in `run_sequential()`.
- `Parallel` is the time for processing the same images with `Pool.map()` in `run_parallel()`.
- `Speedup` is calculated as `sequential_time / parallel_time`.
- `Efficiency` is calculated as `(speedup / PARALLEL_PROCESSES) * 100`.
- `Results saved...` confirms that the processed edge images were written to `results/`.

The files in `results/` are the final edge-detected images, one output image for each input image.

## Notes

- The script uses `4` parallel worker processes by default.
- The workload is intentionally repeated several times to make the timing difference easier to see.
- If you add your own images to `dataset/`, they will be processed the next time you run the script.
