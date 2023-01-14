import time
import sys
import numpy as np
import pickle
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def throughput_calculate_helper(iterrun, func, model, input_sample):
    '''
    A simple helper to calculate average latency
    '''
    time_list = []
    for i in range(iterrun):
        st = time.perf_counter()
        func(model, input_sample)
        end = time.perf_counter()
        time_list.append(end - st)
    time_list.sort()
    # remove top and least 10% data
    time_list = time_list[int(0.1 * iterrun): int(0.9 * iterrun)]
    return np.mean(time_list) * 1000


if __name__ == "__main__":
    my_input = sys.argv[1:]
    params = pickle.load(open(my_input[0], "rb"))
    latency = throughput_calculate_helper(**params)
    print(latency)
