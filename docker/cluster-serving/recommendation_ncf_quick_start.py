from zoo.serving.client import InputQueue, OutputQueue
import os
import cv2
import json
import time
from optparse import OptionParser


def run(path):
    input_api = InputQueue()
    base_path = path

    if not base_path:
        raise EOFError("You have to set your image path")
    output_api = OutputQueue()
    output_api.dequeue()

    import numpy as np
    a = np.array([1, 2])
    input_api.enqueue('a', p=a)

    time.sleep(10)
    print(output_api.query('a'))
