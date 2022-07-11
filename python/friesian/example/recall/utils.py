import time
import os
import multiprocessing


def open_pickle(pkl_file, USE_SPARK):
    if USE_SPARK:
        from pyspark import SparkFiles
        return open(SparkFiles.get(pkl_file), "rb")
    else:
        return open(pkl_file, "rb")


def set_env(num_threads: int):
    print("multiprocessing.cpu_count():", multiprocessing.cpu_count())
    print("os.cpu_count():", os.cpu_count())
    print("len(os.sched_getaffinity(0):", len(os.sched_getaffinity(0)))

    # parameters settings
    kmp_affinity = "granularity=fine"  # "disabled"
    kmp_blocktime = "200"
    env_update = {
        "OMP_NUM_THREADS": str(num_threads),
        "KMP_AFFINITY": kmp_affinity,
        "KMP_BLOCKTIME": kmp_blocktime,
    }
    os.environ.update(env_update)

    for k in env_update:
        print(k, os.environ[k])


class StopWatch:
    """
    Measure elapesed time via 'with' context.
    """

    def __init__(self, msg=""):
        self.msg = msg
        self.t = None

    def __enter__(self):
        self.t = time.time()
        print(f"ENTER: {self.msg}")
        return self.t

    def __exit__(self, etype, value, traceback):
        elapsed_time = time.time() - self.t
        # used to format the time, returning a readable string.
        stime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"LEAVE: {self.msg} [{stime}]")
