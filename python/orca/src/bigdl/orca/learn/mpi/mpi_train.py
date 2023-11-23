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

import os
import argparse
import cloudpickle
from bigdl.dllib.utils.utils import get_node_ip

print(f"Worker on {get_node_ip()} with global rank {os.environ.get('PMI_RANK', 0)}")

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', type=str, default="",
                    help='The directory of the pkl files for mpi training.')
args = parser.parse_args()
pkl_path = args.pkl_path

with open("{}/saved_mpi_estimator.pkl".format(pkl_path), "rb") as f:
    model_creator, optimizer_creator, loss_creator, metrics, \
        scheduler_creator, config, init_func = cloudpickle.load(f)

with open("{}/mpi_train_data.pkl".format(pkl_path), "rb") as f:
    train_data_creator, epochs, batch_size, validation_data_creator,\
        validate_batch_size, train_func, validate_func, train_batches, \
        validate_batches, validate_steps = cloudpickle.load(f)
config["batch_size"] = batch_size
config["validate_batch_size"] = validate_batch_size

if init_func:
    print("Initializing distributed environment")
    init_func(config)

# Wrap DDP should be done by users in model_creator
model = model_creator(config)
optimizer = optimizer_creator(model, config) if optimizer_creator else None
loss = loss_creator if loss_creator else None  # assume it is an instance
if scheduler_creator:
    scheduler = scheduler_creator(optimizer, config)
else:
    scheduler = None
train_ld = train_data_creator(config)
train_batches = train_batches if train_batches else len(train_ld)
print("Batches to train: ", train_batches)
if validation_data_creator:
    valid_ld = validation_data_creator(config)
    validate_batches = validate_batches if validate_batches else len(valid_ld)
    print("Batches to test: ", validate_batches)
else:
    valid_ld = None
    validate_batches = None

train_func(config, epochs, model, train_ld, train_batches, optimizer, loss, scheduler,
           validate_func, valid_ld, metrics, validate_batches, validate_steps)
