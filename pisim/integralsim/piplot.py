import matplotlib.pyplot as plt
import numpy as np
import os

calc_int = np.load(os.path.join('np_store', 'calc_int.npy'))
avg_calc_int = np.load(os.path.join('np_store', 'avg_calc_int.npy'))
std_list = np.load(os.path.join('np_store', 'std_list.npy'))
history_count_list = np.load(os.path.join('np_store', 'history_count_list.npy'))


# int history count v avg calc int
plt.figure(1)
plt.title("int Estimation v. History Count")
plt.xlabel("History Count")
plt.ylabel("int Estimate")
plt.xscale('log')
plt.axhline(y=1/3, color='gray', linestyle='dotted', linewidth=1)
plt.plot(history_count_list, avg_calc_int, marker='o', linestyle='-', color='r')


# history count v error
plt.figure(2)
plt.title("Error v. History Count")
plt.xlabel("History Count")
plt.xscale('log')
plt.ylabel("Error")
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.plot(history_count_list, std_list)

# show graphs
plt.show()

