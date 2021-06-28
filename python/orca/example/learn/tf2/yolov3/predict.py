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
# ===========================================================================
#
# This file is adapted from
# https://github.com/zzh8829/yolov3-tf2/blob/master/train.py,
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py and
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py
#
# MIT License
#
# Copyright (c) 2019 Zihao Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import cv2
import numpy as np
import tensorflow as tf
from yoloV3 import YoloV3, transform_images
import argparse

DEFAULT_IMAGE_SIZE = 416


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint",
                        help="Required. The path where weights locates.")
    parser.add_argument("--names", dest="names",
                        help="Required. The path where classes name locates.")
    parser.add_argument("--class_num", dest="class_num", type=int, default=20,
                        help="Required. class num.")
    parser.add_argument("--image", dest="image",
                        help="Required. image path.")
    parser.add_argument("--output", dest="output", default='./output.jpg',
                        help="Image output path.")

    options = parser.parse_args()

    yolo = YoloV3(classes=options.class_num)

    yolo.load_weights(options.checkpoint).expect_partial()
    print("weights loaded")

    class_names = [c.strip() for c in open(options.names).readlines()]
    print("names loaded")

    img_raw = tf.image.decode_image(
        open(options.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, DEFAULT_IMAGE_SIZE)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                    np.array(scores[0][i]),
                                    np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(options.output, img)
    print('output: {}'.format(options.output))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
