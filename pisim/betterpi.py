import numpy as np
import time

start = time.time()

repetition_factor = 1
chunk_size = 10**7

calc_pi = []

def pi_estimate(n):
    total_inside_circle = 0
    total_points = 0
    while n > 0:
        current_chunk = min(n, chunk_size)
        x = np.random.uniform(-1, 1, current_chunk)
        y = np.random.uniform(-1, 1, current_chunk)
        inside_circle = np.sum((x**2 + y**2) <= 1)
        total_inside_circle += inside_circle
        total_points += current_chunk
        n -= current_chunk
    estimate_output = (4 * (total_inside_circle / total_points))
    return estimate_output

histories = 100000000
for i in range(repetition_factor):
    calc_pi.append(pi_estimate(histories))
    histories *= 10

final_values = [str(n) for n in calc_pi]
print(final_values)

end = time.time()
print(f"Runtime: {end - start}")
