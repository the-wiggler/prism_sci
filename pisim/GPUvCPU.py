import cupy as cp # if not using nvidia GPU with proper dependencies please convert all "cp" to "np". It should work the same
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


batches = 30 # number of times that sets of pi should be estimated, increasing by a factor of 10 for each permutation
pi_per_batch = 1 # how many estimates of pi should be output per batch

chunk_size = 10**7

calc_pi = []

# monte carlo method to calculate pi in a 2*2 box with a unit circle inscribed
def pi_estimateGPU(n):
    total_inside_circle = 0
    total_points = 0
    num_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks) as pbar:
        while n > 0:
            current_chunk = min(n, chunk_size)
            x = cp.random.uniform(-1, 1, current_chunk)
            y = cp.random.uniform(-1, 1, current_chunk)
            inside_circle = cp.sum((x ** 2 + y ** 2) <= 1)
            total_inside_circle += inside_circle
            total_points += current_chunk
            n -= current_chunk
            pbar.update(1)
        estimate_output = (4 * (total_inside_circle / total_points))
        return estimate_output

def pi_estimateCPU(n):
    total_inside_circle = 0
    total_points = 0
    num_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks) as pbar:
        while n > 0:
            current_chunk = min(n, chunk_size)
            x = np.random.uniform(-1, 1, current_chunk)
            y = np.random.uniform(-1, 1, current_chunk)
            inside_circle = np.sum((x**2 + y**2) <= 1)
            total_inside_circle += inside_circle
            total_points += current_chunk
            n -= current_chunk
            pbar.update(1)
        estimate_output = (4 * (total_inside_circle / total_points))
        return estimate_output


# loop that runs the pi_estimate function over a number of permutations and organizes them into a matrix
# GPU VERSION
histories = 2
history_count_listGPU = []
elapsed_timeGPU = []
for i in range(batches):
    start = time.time()
    num = []
    for i in range(pi_per_batch):
        num.append(pi_estimateGPU(histories))
    history_count_listGPU.append(histories)
    calc_pi.append(num)
    histories *= 2
    end = time.time()
    elapsed_timeGPU.append(end - start)

histories = 2
history_count_listCPU = []
elapsed_timeCPU = []
for i in range(batches):
    start = time.time()
    num = []
    for i in range(pi_per_batch):
        num.append(pi_estimateCPU(histories))
    history_count_listCPU.append(histories)
    calc_pi.append(num)
    histories *= 2
    end = time.time()
    elapsed_timeCPU.append(end - start)

print(elapsed_timeGPU)
print(elapsed_timeCPU)


# convert to np array
# np_avg_calc_pi = avg_calc_pi.get()



# pi accuracy v history count
plt.title("GPU Time v. History Count")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("History Count (10^x)")
plt.yscale('log')
plt.axhline(y=cp.pi, color='gray', linestyle='dotted', linewidth=1)
plt.plot(elapsed_timeGPU, history_count_listGPU, marker='o', linestyle='-', color='r', label='GPU')
plt.plot(elapsed_timeCPU, history_count_listCPU, marker='o', linestyle='-', color='b', label="CPU")
plt.legend()
plt.show()
