import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
start = time.time() # just times how long the process takes

# Change these to alter how the neutrons behave!
total_neutrons = 500
cross_section = 1

# Initialize an array with dimensions: total_neutrons * 4
n_positions = np.zeros((total_neutrons, 4))

def random_coord():
    for i in tqdm(range(total_neutrons)):
        rand_value = np.random.uniform(0, 10)
        x = np.random.uniform(-10 / (rand_value * cross_section), 10 / (rand_value * cross_section))
        y = np.random.uniform(-10 / (rand_value * cross_section), 10 / (rand_value * cross_section))
        z = np.random.uniform(-10 / (rand_value * cross_section), 10 / (rand_value * cross_section))
        dist = np.sqrt(x**2 + y**2 + z**2)
        n_positions[i] = [x, y, z, dist]

# Generate random coordinates for neutrons
random_coord()

# Print the generated positions
print(n_positions)

# Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the random neutron points
for i in tqdm(range(total_neutrons)):
    ax.plot([0, n_positions[i, 0]], [0, n_positions[i, 1]], [0, n_positions[i, 2]], 'b')

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_zlim([-10, 10])
plt.title('Neutron Paths Simulation')

print(f'EXECUTION TIME: {time.time() - start} SECONDS') # just times how long the process takes

plt.show()
