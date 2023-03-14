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


import torch
import argparse
import time
import json
from torchvision.models import resnet50
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='PyTorch Inference Benchmark')
parser.add_argument('--name', default='Raw inference benchmark', type=str)
parser.add_argument('--accelerator', default=None, type=str)
parser.add_argument('--precision', default='fp32', type=str)


if __name__ == "__main__":

    args = parser.parse_args()
    model_ft = resnet50(pretrained=False)

    if args.precision == 'fp32' and args.accelerator is None:
        x = torch.rand(2, 3, 224, 224)
        start_time = time.time()
        for _ in range(100):
            y_hat = model_ft(x)
        end_time = time.time()
        output = json.dumps({
            "config": args.name,
            "inference_time": end_time - start_time,
        })

    else:
        from bigdl.nano.pytorch import InferenceOptimizer

        x = torch.rand(2, 3, 224, 224)
        if args.precision != 'fp32':
            optimized_model = InferenceOptimizer.quantize(model_ft,
                                    precision = args.precision,
                                    accelerator = args.accelerator,
                                    calib_dataloader = DataLoader(TensorDataset(torch.rand(10, 3, 224, 224), torch.rand(10))))
        else:
            optimized_model = InferenceOptimizer.trace(model_ft,
                                                accelerator=args.accelerator,
                                                input_sample=torch.rand(1, 3, 224, 224))
        start_time = time.time()
        for _ in range(100):
            y_hat = optimized_model(x)
        end_time = time.time()
        output = json.dumps({
            "config": args.name,
            "inference_time": end_time - start_time,
        })

    print(f'>>>{output}<<<')