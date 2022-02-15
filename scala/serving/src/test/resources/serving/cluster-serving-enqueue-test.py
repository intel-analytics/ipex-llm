#!/usr/bin/python3
from bigdl.serving.client import InputQueue, OutputQueue
import os
import cv2
import json
import time
from optparse import OptionParser
import base64

output_api = OutputQueue()
output_api.dequeue()

input_api = InputQueue()
path = "./cat1.jpeg"
img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
data = cv2.imencode(".jpg", img)[1]
img_encoded = base64.b64encode(data).decode("utf-8")
input_api.enqueue("my-img", t={"b64": img_encoded})
time.sleep(5)

for i in range(10000):
    input_api.enqueue("my-img", t={"b64": img_encoded})
print("10000 images enqueued")
