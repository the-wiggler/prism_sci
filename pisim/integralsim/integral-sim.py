import cupy as cp # if not using nvidia GPU with proper dependencies please convert all "cp" to "np". It should work the same
import numpy as np
from tqdm import tqdm
import time
import os

start = time.time() # just times how long the process takes
chunk_size = 10**7
batches = 1 # number of times that sets of int should be estimated, increasing by a factor of 10 for each permutation
int_per_batch = 5 # how many estimates of int should be output per batch
histories = 100000000


a = cp.pi # lower bound
b = cp.pi * 3 # upper bound
# THE FUNCTION TO INTEGRATE
def f(x):
    return cp.sin(x) * cp.tan(cp.sqrt(2 * x))

# monte carlo method to calculate int
def int_estimate(n):
    total_inside = 0
    total_points = 0
    num_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks) as pbar:
        while n > 0:
            current_chunk = min(n, chunk_size)
            x = cp.random.uniform(a, b, current_chunk)
            inside_func = cp.sum(f(x))
            total_inside += inside_func
            total_points += current_chunk
            n -= current_chunk
            pbar.update(1)
        estimate_output = (total_inside / total_points)
        return estimate_output

# loop that runs the int_estimate function over a number of permutations and organizes them into a matrix
calc_int = []
history_count_list = []
for i in range(batches):
    num = []
    for i in range(int_per_batch):
        num.append(int_estimate(histories))
    history_count_list.append(histories)
    calc_int.append(num)
    # histories *= 10

# print the int values
print(f"int OUTPUT ARRAY: [{batches} BATCHES WITH {int_per_batch} ESTIMATES]")
calc_int = cp.array(calc_int)
print(calc_int)


# average the matrix of int values
print("int BATCH AVERAGE VALUES")
avg_calc_int = []
for i in range(len(calc_int)):
    avg_calc_int_y = []
    avg_calc_int_y.append(cp.average(calc_int[i]))
    avg_calc_int.append(avg_calc_int_y)
avg_calc_int = cp.array(avg_calc_int)
print(avg_calc_int)

# take standard deviations of calc_int array
std_list = []
for i in range(batches):
    std_y = []
    std_y.append(cp.std(calc_int[i]))
    std_list.append(std_y)
std_list = cp.array(std_list)
print("STANDARD DEVIATION FOR int OUTPUT ARRAY")
print(std_list)

# convert to np array and save
np.save(os.path.join('np_store', 'calc_int.npy'), calc_int.get())
np.save(os.path.join('np_store', 'avg_calc_int.npy'), avg_calc_int.get())
np.save(os.path.join('np_store', 'std_list.npy'), std_list.get())
np.save(os.path.join('np_store', 'history_count_list.npy'), history_count_list)


print(f'EXECUTION TIME: {time.time() - start} SECONDS') # just times how long the process takes

