#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from zoo.serving.client import InputQueue, OutputQueue
import os
import cv2
import time
from optparse import OptionParser
import base64
import numpy as np


def run(path):
    input_api = InputQueue()
    base_path = path

    if not base_path:
        raise EOFError("You have to set your image path")
    output_api = OutputQueue()
    output_api.dequeue()
    path = [os.path.join(base_path, "cat1.jpeg")]
    for p in path:
        if not p.endswith("jpeg"):
            continue
        img = cv2.imread(p)
        img = cv2.resize(img, (224, 224))
        data = cv2.imencode(".jpg", img)[1]
        img_encoded = base64.b64encode(data).decode("utf-8")
        result = input_api.enqueue("cat", t={"b64": img_encoded})

    time.sleep(10)

    cat_image_prediction = output_api.query("cat")
    print("cat prediction layer shape: ", cat_image_prediction.shape)
    class_idx = np.argmax(cat_image_prediction)
    print("the class index of prediction of cat image result: ", class_idx)

    # get all result and dequeue
    result = output_api.dequeue()
    for k in result.keys():
        prediction = output_api.get_ndarray_from_b64(result[k])
        # this prediction is the same as the cat_image_prediction above
        print(k, "prediction layer shape: ", prediction.shape)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--image_path", dest="path", default="test_image")
    import sys
    (options, args) = parser.parse_args(sys.argv)
    run(options.path)
