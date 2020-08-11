from zoo.serving.client import InputQueue, OutputQueue
import time

def run():
    input_api = InputQueue()
    output_api = OutputQueue()
    output_api.dequeue()

    import numpy as np
    a = np.array([1, 2])
    input_api.enqueue('a', p=a)

    time.sleep(5)
    print(output_api.query('a'))

if __name__ == "__main__":
    run()
