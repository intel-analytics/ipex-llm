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
cd ../../script

benchmark-chronos -l 96 -o 720 -m tcn -s train -d nyc_taxi
benchmark-chronos -l 96 -o 720 -m tcn -s latency -d nyc_taxi
benchmark-chronos -l 96 -o 720 -m tcn -s throughput -d nyc_taxi

benchmark-chronos -l 96 -o 720 -m lstm -s train -d nyc_taxi
benchmark-chronos -l 96 -o 720 -m seq2seq -s train -d nyc_taxi
benchmark-chronos -l 96 -o 720 -m autoformer -s train -d nyc_taxi
benchmark-chronos -l 96 -o 720 -m nbeats -s train -d nyc_taxi
