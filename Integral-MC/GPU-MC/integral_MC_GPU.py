# ** THIS PROGRAM ONLY FUNCTIONS ON NVIDIA GPUS WITH PROPER DEPENDENCIES INSTALLED ** #
# CUPY IS A NUMPY LIBRARY OPTIMIZED TO FUNCTION WITH CUDA #
import cupy as cp
import numpy as np
from tqdm import tqdm
import time
import os


class IntegralMC:
    def __init__(self, batches, int_per_batch, chunk_size, a, b, histories, f, hist_factor):
        self.batches = batches
        self.int_per_batch = int_per_batch
        self.chunk_size = chunk_size
        self.a = a
        self.b = b
        self.histories = histories
        self.f = f
        self.hist_factor = hist_factor

    # Monte Carlo method to calculate the integral over (a, b)
    def calc(self):  # calc is short for calculator for those who are just now viewing the program
        start = time.time()  # Start timing the process
        calc_int, history_count_list, batch_times = self.int_loop(self.batches, self.histories, self.int_per_batch,
                                                                  self.a, self.b, self.chunk_size, self.f,
                                                                  self.hist_factor)  # int_loop performs a loop of MC calculation as defined by int_estimate
        self.int_print(self.batches, history_count_list, calc_int)  # takes calc_int array from int_loop and prints it
        avg_calc_int = self.avg_matrix(calc_int)  # averages each row of multiple columns in calc_int into one column
        std_list = self.std_calc_int(self.batches,
                                     calc_int)  # calculates standard deviation of each row in calc_int and formats it similarly to avg_calc_int
        self.sv_matrix(calc_int, avg_calc_int, std_list, history_count_list,
                       batch_times)  # exports arrays to np_store folder
        print(f'EXECUTION TIME: {time.time() - start} SECONDS')  # print the execution time

    def int_estimate(self, n, a, b, chunk_size,
                     f):  # function that performs an individual MC calculation on the defined integral function. "n" refers to the number of MC sample points left to take: total_points approaches n's original value as n decreases: n = histories
        sum_function_output = 0  # sum of the y values taken of the function values
        total_points = 0  # total number of point (x) estimates taken, finished when n = 0
        num_chunks = (n + chunk_size - 1) // chunk_size  # defines how many chunks are left based on n
        with tqdm(total=num_chunks) as pbar:  # defines progress bar
            while n > 0:  # continues monte carlo until permutations are finished
                current_chunk = min(n,
                                    chunk_size)  # determines to use chunk size (if n > chunk_size) or n (if total remaining times to perform loop is less than chunk_size)
                x = cp.random.uniform(a, b, current_chunk)  # takes a random number x within the bounds of the integral
                sum_function_output += cp.sum(
                    f(x))  # substitutes each x value into the function to get their y value, or vertical distance from y=0
                total_points += current_chunk  # adds total amount of points to the 1/N term
                n -= current_chunk  # lets n approach 0 (calculation progress)
                pbar.update(1)
            estimate_output = (b - a) * (
                        sum_function_output / total_points)  # I ≈ (b - a) * (1/N) * Σ f(xi) --- (the actual monte carlo equation)
            return estimate_output  # returns calculated integral value

    def int_loop(self, batches, histories, int_per_batch, a, b, chunk_size, f, hist_factor):
        calc_int = []  # list that the calculated integral estimations are put into
        history_count_list = []  # records number of histories performed per batch
        batch_times = []  # records the time taken per batch
        for i in range(batches):
            batch_start_time = time.time()
            num = []
            for i in range(int_per_batch):
                num.append(self.int_estimate(histories, a, b, chunk_size, f))
            history_count_list.append(histories)
            calc_int.append(num)
            histories *= hist_factor  # increase the number of histories for the next batch (for data analysis)
            batch_time = time.time() - batch_start_time  # Calculate the time taken for this batch
            batch_times_y = [batch_time]  # appends batch_time into batch_times_y
            batch_times.append(batch_times_y)
        batch_times = cp.array(batch_times)
        calc_int = cp.array(calc_int)
        return calc_int, history_count_list, batch_times

    def int_print(self, batches, int_per_batch, calc_int):
        # Print the integral values
        print(f"INTEGRAL OUTPUT ARRAY: [{batches} BATCHES WITH {int_per_batch} ESTIMATES]")
        calc_int = cp.array(calc_int)
        print(calc_int)

    def avg_matrix(self, calc_int):
        # Average the matrix of integral values
        print("INTEGRAL BATCH AVERAGE VALUES")
        avg_calc_int = []
        for i in range(len(calc_int)):
            avg_calc_int_y = [cp.average(calc_int[i])]  # calculate average for each batch
            avg_calc_int.append(avg_calc_int_y)
        avg_calc_int = cp.array(avg_calc_int)
        print(avg_calc_int)
        return avg_calc_int

    def std_calc_int(self, batches, calc_int):
        # Take standard deviations of calc_int array
        std_list = []
        for i in range(batches):
            std_y = [cp.std(calc_int[i])]  # calculate standard deviation for each batch
            std_list.append(std_y)
        std_list = cp.array(std_list)
        print("STANDARD DEVIATION FOR INTEGRAL OUTPUT ARRAY")
        print(std_list)
        return std_list

    def sv_matrix(self, calc_int, avg_calc_int, std_list, history_count_list, batch_times):
        # Save results as numpy arrays in the 'np_store' directory
        np.save(os.path.join('np_store', 'calc_int.npy'), calc_int.get())
        np.save(os.path.join('np_store', 'avg_calc_int.npy'), avg_calc_int.get())
        np.save(os.path.join('np_store', 'std_list.npy'), std_list.get())
        np.save(os.path.join('np_store', 'history_count_list.npy'), history_count_list)
        np.save(os.path.join('np_store', 'batch_times.npy'), batch_times.get())
        # .get() is used to convert cupy arrays to numpy arrays before saving
