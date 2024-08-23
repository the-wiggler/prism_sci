import matplotlib.pyplot as plt
import numpy as np
import os

calc_int = np.load(os.path.join('np_store', 'calc_int.npy'))
avg_calc_int = np.load(os.path.join('np_store', 'avg_calc_int.npy'))
std_list = np.load(os.path.join('np_store', 'std_list.npy'))
history_count_list = np.load(os.path.join('np_store', 'history_count_list.npy'))


# int history count v avg calc int
plt.figure(1)
plt.suptitle("Integral Calculation Value v. History Count")
plt.title(r'$\int_{0}^{5} f(x) = \sqrt{9 - (x - 3)^2}\, dx$', fontsize=10)
plt.xlabel("History Count")
plt.ylabel("Integral Calculation Value")
plt.xscale('log')
plt.plot(history_count_list, avg_calc_int, marker='o', linestyle='-', color='r')


# history count v error
plt.figure(2)
plt.suptitle("Standard Deviation v. History Count")
plt.title(r'$\int_{0}^{5} f(x) = \sqrt{9 - (x - 3)^2}\, dx$', fontsize=10)
plt.xlabel("History Count")
plt.xscale('log')
plt.ylabel("Standard Deviation")
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.plot(history_count_list, std_list)

# show graphs
plt.show()

