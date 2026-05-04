import cv2
import os
import time
import numpy as np
import random
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
NUM_IMAGES = 200
PARALLEL_PROCESSES = 4
CHUNKSIZE = 10
WORKLOAD_PASSES = 10


# ---------- Dataset Generator ----------
def generate_dataset(n=NUM_IMAGES, seed=42):
    """Generate synthetic images if dataset folder is empty."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(INPUT_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
    if existing:
        print(f"Dataset already exists: {len(existing)} images found, skipping generation.")
        return

    random.seed(seed)
    np.random.seed(seed)

    for i in range(n):
        h = random.randint(400, 1200)
        w = random.randint(400, 1200)
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        for _ in range(random.randint(3, 10)):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), color, -1)
            cv2.circle(
                img,
                (random.randint(0, w - 1), random.randint(0, h - 1)),
                random.randint(20, 100),
                color,
                -1,
            )

        cv2.imwrite(os.path.join(INPUT_DIR, f"img_{i:03d}.jpg"), img)

    print(f"Generated {n} synthetic images in '{INPUT_DIR}/'")

x=1
# ---------- Image Processing ----------
def process_image(path):
    filename = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Could not read: {path}")
        return

    img = cv2.resize(img, (800, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for _ in range(WORKLOAD_PASSES):
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blur, 100, 200)

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), edges)


# ---------- Sequential ----------
def run_sequential(image_paths):
    start = time.time()
    for path in image_paths:
        process_image(path)
    return time.time() - start


# ---------- Parallel ----------
def run_parallel(image_paths, processes=None, chunksize=10):
    if processes is None:
        processes = cpu_count()
    start = time.time()
    with Pool(processes=processes) as p:
        p.map(process_image, image_paths, chunksize=chunksize)
    return time.time() - start


# ---------- Main ----------
def main():
    generate_dataset()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    if not image_paths:
        print("No images found in dataset/. Add images and re-run.")
        return

    cores = cpu_count()
    print(f"\n{'='*45}")
    print(f"  Total Images : {len(image_paths)}")
    print(f"  CPU Cores    : {cores}")
    print(f"  Workload     : {WORKLOAD_PASSES} blur/canny passes per image")
    print(f"{'='*45}")

    # Sequential
    seq_time = run_sequential(image_paths)
    print(f"  Sequential   : {seq_time:.4f} sec")

    # Parallel (all cores)
    par_time = run_parallel(image_paths, processes=PARALLEL_PROCESSES, chunksize=CHUNKSIZE)
    print(
        f"  Parallel     : {par_time:.4f} sec  "
        f"(processes={PARALLEL_PROCESSES}, chunksize={CHUNKSIZE})"
    )

    # Speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    efficiency = (speedup / PARALLEL_PROCESSES) * 100

    print(f"{'='*45}")
    print(f"  Speedup      : {speedup:.2f}x")
    print(f"  Efficiency   : {efficiency:.1f}%  (vs {PARALLEL_PROCESSES} processes)")
    print(f"  Results saved to '{OUTPUT_DIR}/'")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()