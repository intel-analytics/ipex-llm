#!/usr/bin/env bash
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

set -e

bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch,inference --force-reinstall

apt-get update && apt-get install -y libgl1

cd $ANALYTICS_ZOO_ROOT/python/nano/benchmark/resnet_inference/

echo "Nano_Perf: Running PyTorch Inference Baseline"
python pytorch-resnet-inference.py --name "PyTorch Inference Baseline"

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference Default"
python pytorch-resnet-inference.py --name "Nano Inference Baseline"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference Default with int8"
python pytorch-resnet-inference.py --precision "int8" --name "Nano Inference with int8"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference Default with onnxruntime"
python pytorch-resnet-inference.py --accelerator "onnxruntime" --name "Nano Inference with onnxruntime"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference Default with openvino"
python pytorch-resnet-inference.py --accelerator "openvino" --name "Nano Inference with openvino"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference Default with jit"
python pytorch-resnet-inference.py --accelerator "jit" --name "Nano Inference with jit"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference with openvino and int8"
python pytorch-resnet-inference.py --accelerator "openvino" --precision "int8" --name "Nano Inference with openvino and int8"
source bigdl-nano-unset-env

source bigdl-nano-init
echo "Nano_Perf: Running Nano Inference with onnxruntime and int8"
python pytorch-resnet-inference.py --accelerator "onnxruntime" --precision "int8" --name "Nano Inference with onnxruntime and int8"
source bigdl-nano-unset-env
