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
import json
import time


if __name__ == "__main__":
    input_api = InputQueue()

    base_path = "test_image"

    if not base_path:
        raise EOFError("You have to set your image path")
    output_api = OutputQueue()
    output_api.dequeue()
    path = os.listdir(base_path)
    for p in path:
        if not p.endswith("jpeg"):
            continue
        img = cv2.imread(os.path.join(base_path, p))
        img = cv2.resize(img, (224, 224))
        input_api.enqueue_image(p, img)

    time.sleep(5)

    # get all result and dequeue
    result = output_api.dequeue()
    for k in result.keys():
        output = "image: " + k + ", classification-result:"
        tmp_dict = json.loads(result[k])
        for class_idx in tmp_dict.keys():
            output += "class: " + class_idx + "'s prob: " + tmp_dict[class_idx]
        print(output)
