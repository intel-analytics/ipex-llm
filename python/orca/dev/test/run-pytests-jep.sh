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

cd "`dirname $0`"

echo "Running Jep tests"
set -ex

if [[ ! -z "${DATA_STORE_URI}" ]]; then
    if [[ -d /tmp/datasets/ ]]; then
        rm -rf /tmp/datasets/MNIST/
    fi
    wget  $DATA_STORE_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P /tmp/datasets/MNIST/raw
    wget  $DATA_STORE_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P /tmp/datasets/MNIST/raw
    wget  $DATA_STORE_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P /tmp/datasets/MNIST/raw
    wget  $DATA_STORE_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P /tmp/datasets/MNIST/raw
fi

cd ../../
python -m pytest -v test/bigdl/orca/learn/jep/test_pytorch_estimator_for_spark.py
python -m pytest -v test/bigdl/orca/learn/jep/test_pytorch_estimator_for_dataloader.py
python -m pytest -v test/bigdl/orca/learn/jep/test_pytorch_estimator_for_spark_creator.py
python -m pytest -v test/bigdl/orca/learn/jep/test_pytorch_estimator_for_dataframe.py
python -m pytest -v test/bigdl/orca/learn/jep/test_pytorch_estimator_for_save_load.py
python -m pytest -v test/bigdl/orca/torch/test_torch_estimator.py
python -m pytest -v test/bigdl/orca/torch/test_torch.py
python -m pytest -v test/bigdl/orca/torch/test_torch_optim.py
python -m pytest -v test/bigdl/orca/torch/test_inference_model_torch.py
