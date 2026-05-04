# Measure parallel overhead for a trivial task
from ctypes import c_int
import time
from multiprocessing import Pool, cpu_count

def trivial_task(x):
	# A very small task (e.g., increment)
	return x + 1

def run_sequential(n):
	start = time.time()
	results = [trivial_task(i) for i in range(n)]
	elapsed = time.time() - start
	return elapsed, results

def run_parallel(n, processes=None, chunksize=10):
	if processes is None:
		processes = cpu_count()
	start = time.time()
	with Pool(processes=processes) as p:
		results = p.map(trivial_task, range(n), chunksize=chunksize)
	elapsed = time.time() - start
	return elapsed, results

def main():
	N = 300 # Small task count
	PROCESSES = max(4, cpu_count())
	CHUNKSIZE = 30

	print(f"\n{'='*40}")
	print(f"  Measuring Parallel Overhead (N={N})")
	print(f"  CPU Cores: {cpu_count()}")
	print(f"  Processes: {PROCESSES}")
	print(f"  Chunk Size: {CHUNKSIZE}")
	print(f"{'='*40}")

	seq_time, _ = run_sequential(N)
	print(f"  Sequential   : {seq_time:.6f} sec")

	par_time, _ = run_parallel(N, processes=PROCESSES, chunksize=CHUNKSIZE)
	print(f"  Parallel     : {par_time:.6f} sec  (processes={PROCESSES}, chunksize={CHUNKSIZE})")

	overhead = par_time - seq_time
	speedup = seq_time / par_time if par_time > 0 else 0
	efficiency = (speedup / PROCESSES) * 100

	print(f"{'='*40}")
	print(f"  Overhead      : {overhead:.6f} sec")
	print(f"  Speedup       : {speedup:.2f}x")
	print(f"  Efficiency    : {efficiency:.1f}%  (vs {PROCESSES} processes)")
	print(f"{'='*40}\n")

if __name__ == "__main__":
	main()
