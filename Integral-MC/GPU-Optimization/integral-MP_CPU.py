# import cupy as cp
import numpy as np
from tqdm import tqdm
import time
import os

start = time.time()  # Start timing the process

batches = 5  # Number of times that sets of integral should be estimated ( y in array )
int_per_batch = 2  # How many estimates of integral should be output per batch ( x in array )

chunk_size = 10**7 # for RAM management, decrease for less ram (current system uses 32gb) -- untested at higher values

calc_int = [] # list that the calculated integral estimations are put into

# INPUT FUNCTION HERE
def f(x):
    return np.sqrt(9-(x-3)**2)

# Monte Carlo method to calculate the integral over (a, b)
def int_estimate(n, a, b):
    total_func = 0 # sum of the height values taken of the function values
    total_points = 0 # total number of point (x) estimates taken
    num_chunks = (n + chunk_size - 1) // chunk_size
    with tqdm(total=num_chunks) as pbar: # defines progress bar
        while n > 0: # continues monte carlo until permutations are finished
            current_chunk = min(n, chunk_size)
            x = np.random.uniform(a, b, current_chunk) # takes a random number x within the bounds of the integral
            total_func += np.sum(f(x)) # substitutes each x value into the function to get their y value, or vertical distance from y=0
            total_points += current_chunk # adds total amount of points
            n -= current_chunk # lets n approach 0
            pbar.update(1) # updates progress bar
        estimate_output = (b - a) * (total_func / total_points) # (b-a) is the width of the integral----(avg/points) calculates the average height of the function----its like finding the area of a square
        return estimate_output

# Integration bounds
a = 0  # Lower bound of the integral
b = 5  # Upper bound of the integral

# Loop that runs the int_estimate function over a number of permutations and organizes them into a matrix
histories = 10000 # number of permutations of integral estimate to perform
history_count_list = [] # records number of histories performed per batch
for i in range(batches):
    num = []
    for i in range(int_per_batch):
        num.append(int_estimate(histories, a, b))
    history_count_list.append(histories)
    calc_int.append(num)
    histories *= 10


# Print the integral values
print(f"INTEGRAL OUTPUT ARRAY: [{batches} BATCHES WITH {int_per_batch} ESTIMATES]")
calc_int = np.array(calc_int)
print(calc_int)

# Average the matrix of integral values
print("INTEGRAL BATCH AVERAGE VALUES")
avg_calc_int = []
for i in range(len(calc_int)):
    avg_calc_int_y = []
    avg_calc_int_y.append(np.average(calc_int[i]))
    avg_calc_int.append(avg_calc_int_y)
avg_calc_int = np.array(avg_calc_int)
print(avg_calc_int)

# Take standard deviations of calc_int array
std_list = []
for i in range(batches):
    std_y = []
    std_y.append(np.std(calc_int[i]))
    std_list.append(std_y)
std_list = np.array(std_list)
print("STANDARD DEVIATION FOR INTEGRAL OUTPUT ARRAY")
print(std_list)

# Convert to numpy array and save
np.save(os.path.join('np_store', 'calc_int.npy'), calc_int)
np.save(os.path.join('np_store', 'avg_calc_int.npy'), avg_calc_int)
np.save(os.path.join('np_store', 'std_list.npy'), std_list)
np.save(os.path.join('np_store', 'history_count_list.npy'), history_count_list)

print(f'EXECUTION TIME: {time.time() - start} SECONDS')  # Output the execution time