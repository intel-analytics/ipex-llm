import multiprocessing as mp
from multiprocessing.pool import Pool

from stable_diffusion.runner import StableDiffusionRunner

cpu_binding = False
pool = None
var_dict = None

class RunnerProcess(mp.Process):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.runner = StableDiffusionRunner.initialize() # while calling this everything in this line must be serializable
    # can not override run func here otherwise it would loop infinitely between __init__() and run()

class RunnerPool(Pool):
    def __init__(self) -> None:
        super().__init__(1)
    
    @staticmethod
    def Process(ctx, *args, **kwargs):
        return RunnerProcess(*args, **kwargs)
    
def test_func(x):
    print("Running test_func")
    p = mp.current_process()
    y = x * x if p.runner is None else x
    print(y)


if __name__ == '__main__':    
    # test functionality
    
    with RunnerPool() as pool:
        for i in range(3):
            print(f"Applying {i} to pool")
            pool.apply(test_func, (i,))
