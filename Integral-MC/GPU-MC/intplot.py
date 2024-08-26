import os
import matplotlib.pyplot as plt
import numpy as np
from integral_MC_GPU import IntegralMC

integral_calc = IntegralMC(
    batches=5,  # Number of times that sets of integral should be estimated ( y in calc_int array )
    int_per_batch=4,  # How many estimates of integral should be output per batch ( x in calc_int array )
    chunk_size=10 ** 7,
    # for VRAM management, chunks history count into "manageable bites" for your VRAM. decrease for less VRAM (current system uses 12GB) -- untested at higher values
    a=0,  # Lower bound of the integral
    b=5,  # Upper bound of the integral
    histories=1000,  # number of histories to take per permutation
    hist_factor=10,  # factor in which histories are multiplied in order to create a gradient for data analysis
    f=lambda x: x ** 2  # the function to integrate
)
integral_calc.calc()


def plot():
    calc_int = np.load(os.path.join('np_store', 'calc_int.npy'))
    avg_calc_int = np.load(os.path.join('np_store', 'avg_calc_int.npy'))
    std_list = np.load(os.path.join('np_store', 'std_list.npy'))
    history_count_list = np.load(os.path.join('np_store', 'history_count_list.npy'))
    batch_times = np.load(os.path.join('np_store', 'batch_times.npy'))

    # Average Calculated Integral Value v. Histories
    x = history_count_list
    y = avg_calc_int
    plt.figure(1)
    plt.suptitle("Integral Calculation Value v. History Count")
    plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
    plt.xlabel("History Count")
    plt.ylabel("Integral Calculation Value")
    plt.xscale('linear')
    plt.scatter(x, y, s=0.05, alpha=1, color='r')

    # Standard Deviation v. Histories
    x = history_count_list
    y = std_list
    plt.figure(2)
    plt.suptitle("Standard Deviation v. History Count")
    plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
    plt.xlabel("History Count")
    plt.xscale('linear')
    plt.ylabel("Standard Deviation")
    ax = plt.gca()
    plt.scatter(x, y, s=0.9, alpha=1, color='b')

    # time taken per batch v. history count
    x = history_count_list
    y = batch_times
    plt.figure(3)
    plt.suptitle("Batch Times v. History Count")
    plt.title(r'$\int_{0}^{5} f(x) = {f}, dx$', fontsize=10)
    plt.xlabel("History Count")
    plt.xscale('linear')
    plt.ylabel("Batch Times (s)")
    # ax = plt.gca()
    # ax.set_ylim([0, 0.003])
    # ax.set_xlim([0, 900000])
    plt.scatter(x, y, s=0.9, alpha=1, color='b')

    # batch size (int_per_batch) v. accuracy (# of batches constant)
    # x = int_per_batch
    # y = avg_calc_int
    # plt.figure(1)
    # plt.suptitle("Integral Calculation Value v. History Count")
    # plt.title(r'$\int_{0}^{5} f(x) = \sin{(x)}\, dx$', fontsize=10)
    # plt.xlabel("History Count")
    # plt.ylabel("Integral Calculation Value")
    # plt.xscale('linear')
    # plt.scatter(x , y, s=0.05, alpha=1, color='r')

    # chunk_size v. execution time (all else constant)
    # different functions v. execution time
    # CPU v. GPU
    plt.show()


plot()
