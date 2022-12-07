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


import time
import os
import sys
import argparse

import torch
from torch import nn

from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch import InferenceOptimizer


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True, include_top=False, freeze=True):
        super().__init__()
        backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
        output_size = backbone.get_output_size()
        head = nn.Linear(output_size, num_classes)
        self.model = nn.Sequential(backbone, head)

    def forward(self, x):
        return self.model(x)


options = list(InferenceOptimizer.ALL_INFERENCE_ACCELERATION_METHOD.keys())
# if option is in sub_options, we can load it without original model,
# otherwise, we have to load it with original model
sub_options = ["openvino_fp32", "openvino_int8", "onnxruntime_fp32",
               "onnxruntime_int8_qlinear", "onnxruntime_int8_integer"]


def run(args):
    save_dir = "models"
    imgs = torch.rand((1000, 1, 3, 128, 128))
    opt = InferenceOptimizer()
    if args.option in options:
        try:
            if args.option not in sub_options:
                model = ResNet18(10)
                model = opt.load(os.path.join(save_dir, args.option), model)
            else:
                model = opt.load(os.path.join(save_dir, args.option))
            with InferenceOptimizer.get_context(model):
                # warmup
                for img in imgs[:10]:
                    _ = model(img)

                st = time.time()
                for img in imgs:
                    _ = model(img)
                end = time.time()
            throughput = len(imgs) / (end - st)
        except Exception:
            throughput = 0
        print(f"Throughput: {throughput}")
    else:
        print(f"unkonwn option: {args.option}")
        sys.exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str)
    args = parser.parse_args()

    run(args)
