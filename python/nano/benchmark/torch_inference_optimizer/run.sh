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


export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export NANO_HOME=${ANALYTICS_ZOO_ROOT}/python/nano/src
export NANO_BENCHMARK_DIR=${ANALYTICS_ZOO_ROOT}/python/nano/benchmark
export WORKLOAD_DIR=${NANO_BENCHMARK_DIR}/torch_inference_optimizer

set -e

# install dependencies
bash $ANALYTICS_ZOO_ROOT/python/nano/dev/build_and_install.sh linux default false pytorch,inference

# set nano's environment variables
source bigdl-nano-unset-env
source bigdl-nano-init

# cd workload directory
cd $WORKLOAD_DIR

export OMP_NUM_THREADS=4
taskset -c 0-3 python optimize.py

options="original fp32_channels_last fp32_ipex fp32_ipex_channels_last bf16 \
         bf16_channels_last bf16_ipex bf16_ipex_channels_last int8 int8_ipex \
         jit_fp32 jit_bf16 jit_fp32_ipex jit_fp32_ipex_channels_last \
         jit_bf16_ipex jit_bf16_ipex_channels_last openvino_fp32 openvino_int8 \
         onnxruntime_fp32 onnxruntime_int8_qlinear onnxruntime_int8_integer"

for option in $options;
do
    bash benchmark.sh 1 4 $option
done

for option in $options;
do
    bash benchmark.sh 6 4 $option
done

source bigdl-nano-unset-env
