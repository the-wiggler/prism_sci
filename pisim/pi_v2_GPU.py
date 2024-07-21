import cupy as cp # if not using nvidia GPU with proper dependencies please convert all "cp" to "np". It should work the same
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

start = time.time() # just times how long the process takes

batches = 11 # number of times that sets of pi should be estimated, increasing by a factor of 10 for each permutation
pi_per_batch = 2 # how many estimates of pi should be output per batch

chunk_size = 10**7

calc_pi = []

# monte carlo method to calculate pi in a 2*2 box with a unit circle inscribed
def pi_estimate(n):
    total_inside_circle = 0
    total_points = 0
    num_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks) as pbar:
        while n > 0:
            current_chunk = min(n, chunk_size)
            x = cp.random.uniform(-1, 1, current_chunk)
            y = cp.random.uniform(-1, 1, current_chunk)
            inside_circle = cp.sum((x ** 2 + y ** 2) <= 1, dtype=cp.int64)
            total_inside_circle += inside_circle
            total_points += current_chunk
            n -= current_chunk
            pbar.update(1)
        estimate_output = (4 * (total_inside_circle / total_points))
        return estimate_output


# loop that runs the pi_estimate function over a number of permutations and organizes them into a matrix
histories = 10
history_count_list = []
for i in range(batches):
    num = []
    for i in range(pi_per_batch):
        num.append(pi_estimate(histories))
    history_count_list.append(histories)
    calc_pi.append(num)
    histories *= 10

# print the pi values
print(f"PI OUTPUT ARRAY: [{batches} BATCHES WITH {pi_per_batch} ESTIMATES]")
calc_pi = cp.array(calc_pi)
print(calc_pi)


# average the matrix of pi values
print("PI BATCH AVERAGE VALUES")
avg_calc_pi = []
for i in range(len(calc_pi)):
    avg_calc_pi_y = []
    avg_calc_pi_y.append(cp.average(calc_pi[i]))
    avg_calc_pi.append(avg_calc_pi_y)
avg_calc_pi = cp.array(avg_calc_pi)
print(avg_calc_pi)

# take standard deviations of calc_pi array
std_list = []
for i in range(batches):
    std_y = []
    std_y.append(cp.std(calc_pi[i]))
    std_list.append(std_y)
std_list = cp.array(std_list)
print("STANDARD DEVIATION FOR PI OUTPUT ARRAY")
print(std_list)

# convert to np array
np_avg_calc_pi = avg_calc_pi.get()


# pi accuracy v history count
plt.title("Pi Estimation v. History Count")
plt.xlabel("History Count")
plt.ylabel("Pi Estimate")
plt.xscale('log')
plt.axhline(y=cp.pi, color='gray', linestyle='dotted', linewidth=1)
plt.plot(history_count_list, np_avg_calc_pi, marker='o', linestyle='-', color='r')

print(f'EXECUTION TIME: {time.time() - start} SECONDS') # just times how long the process takes

plt.show()
