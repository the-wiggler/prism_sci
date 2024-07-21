import matplotlib.pyplot as plt
import numpy as np
import os

calc_pi = np.load(os.path.join('np_store', 'calc_pi.npy'))
avg_calc_pi = np.load(os.path.join('np_store', 'avg_calc_pi.npy'))
std_list = np.load(os.path.join('np_store', 'std_list.npy'))
history_count_list = np.load(os.path.join('np_store', 'history_count_list.npy'))


# pi history count v avg calc pi
plt.figure(1)
plt.title("Pi Estimation v. History Count")
plt.xlabel("History Count")
plt.ylabel("Pi Estimate")
plt.xscale('log')
plt.axhline(y=np.pi, color='gray', linestyle='dotted', linewidth=1)
plt.plot(history_count_list, avg_calc_pi, marker='o', linestyle='-', color='r')


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

