import matplotlib.pyplot as plt
import numpy as np
import os

calc_int = np.load(os.path.join('np_store', 'calc_int.npy'))
avg_calc_int = np.load(os.path.join('np_store', 'avg_calc_int.npy'))
std_list = np.load(os.path.join('np_store', 'std_list.npy'))
history_count_list = np.load(os.path.join('np_store', 'history_count_list.npy'))
batch_times = np.load(os.path.join('np_store', 'batch_times.npy'))


# # int history count v avg calc int
x = history_count_list
y = avg_calc_int
plt.figure(1)
plt.suptitle("Integral Calculation Value v. History Count")
plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
plt.xlabel("History Count")
plt.ylabel("Integral Calculation Value")
plt.xscale('linear')
plt.scatter(x , y, s=0.05, alpha=1, color='r')


# # history count v error
x = history_count_list
y = std_list
plt.figure(2)
plt.suptitle("Standard Deviation v. History Count")
plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
plt.xlabel("History Count")
plt.xscale('linear')
plt.ylabel("Standard Deviation")
ax = plt.gca()
ax.set_ylim([0, 0.02])
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.scatter(x , y, s=0.9, alpha=1, color='b')

# time taken per batch v history count
x = history_count_list
y = batch_times
plt.figure(3)
plt.suptitle("Batch Times v. History Count")
plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
plt.xlabel("History Count")
plt.xscale('linear')
plt.ylabel("Batch Times")
ax = plt.gca()
ax.set_ylim([0, 0.01])
# ax.set_xlim([0, 900000])
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.scatter(x , y, s=0.9, alpha=1, color='b')


# show graphs
plt.show()
