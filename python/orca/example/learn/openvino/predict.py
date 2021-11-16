#
# Copyright 2016 The BigDL Authors.
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

from __future__ import print_function
import argparse
import numpy as np
import time
import cv2
import glob
import math
from bigdl.orca.learn.openvino import Estimator

from bigdl.orca import init_orca_context, stop_orca_context


def crop(img, w, h):
    center = np.array(img.shape) / 2
    x = center[1] - w / 2
    y = center[0] - h / 2
    crop_img = img[int(y):int(y + h), int(x):int(x + w)]
    crop_img = np.transpose(crop_img, (2, 0, 1))
    return crop_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Tensorboard Example')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, or spark-submit.')
    parser.add_argument('--model_path', type=str, default="./model.xml",
                        help="Path to the OpenVINO model file")
    parser.add_argument('--image_folder', type=str, default="./",
                        help="The path to the folder where the images are stored.")
    parser.add_argument('--core_num', type=int, default=4,
                        help="The number of cpu cores you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument('--executor_num', type=int, default=2,
                        help="The number of executors when cluster_mode=yarn.")
    parser.add_argument('--data_num', type=int, default=12, help="The number of dummy data.")
    parser.add_argument('--batch_size', type=int, default=4, help="The batch size of inference.")
    parser.add_argument('--memory', type=str, default="2g", help="The executor memory size.")
    args = parser.parse_args()

    if args.cluster_mode == "local":
        init_orca_context(cores=args.core_num, memory=args.memory)
    elif args.cluster_mode.startswith("yarn"):
        init_orca_context(cluster_mode=args.cluster_mode, cores=args.core_num,
                          num_nodes=args.executor_num, memory=args.memory)
    elif args.cluster_mode == "spark-submit":
        init_orca_context(cluster_mode=args.cluster_mode)

    images = [cv2.imread(file) for file in
              glob.glob(args.image_folder + "/*.jpg")]
    images = [crop(img, 416, 416) for img in images]
    image_num = len(images)
    copy_time = math.ceil(args.data_num/image_num)
    images = images * copy_time
    images = np.array(images[:args.data_num])
    est = Estimator.from_openvino(model_path=args.model_path)
    start = time.time()
    result = est.predict(images, batch_size=args.batch_size)
    end = time.time()
    print("Throughput: ", args.data_num / (end - start))
    assert isinstance(result, list)
    assert result[0].shape == (args.data_num, 255, 13, 13)
    assert result[1].shape == (args.data_num, 255, 26, 26)
    assert result[2].shape == (args.data_num, 255, 52, 52)
    stop_orca_context()

