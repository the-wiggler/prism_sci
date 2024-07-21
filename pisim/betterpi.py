import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

batches = 6 # number of times that sets of pi should be estimated, increasing by a factor of 10 for each permutation
pi_per_batch = 10 # how many estimates of pi should be output per batch

chunk_size = 10**7

calc_pi = []

def pi_estimate(n):
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


histories = 10
history_count_list = []
for i in range(batches):
    num = []
    for i in range(pi_per_batch):
        num.append(pi_estimate(histories))
    history_count_list.append(histories)
    calc_pi.append(num)
    histories *= 10

print(f"PI OUTPUT ARRAY: [{batches} BATCHES WITH {pi_per_batch} ESTIMATES]")
calc_pi = np.array(calc_pi)
print(calc_pi)

std_list = []
for i in range(batches):
    std_y = []
    std_y.append(np.std(calc_pi[i]))
    std_list.append(std_y)
std_list = np.array(std_list)
print("STANDARD DEVIATION FOR PI OUTPUT ARRAY")
print(std_list)


# pi accuracy v history count
print("PI BATCH AVERAGE VALUES")
avg_calc_pi = []
for i in range(len(calc_pi)):
    avg_calc_pi_y = []
    avg_calc_pi_y.append(np.average(calc_pi[i]))
    avg_calc_pi.append(avg_calc_pi_y)
print(np.array(avg_calc_pi))
plt.title("Pi Estimation v. History Count")
plt.xlabel("History Count")
plt.ylabel("Pi Estimate")
plt.xscale('log')
plt.axhline(y=np.pi, color='gray', linestyle='dotted', linewidth=1)
plt.plot(history_count_list, avg_calc_pi, marker='o', linestyle='-', color='r')

plt.show()
