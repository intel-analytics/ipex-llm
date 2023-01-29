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

# Enviroment Settings
export ANALYTICS_ZOO_ROOT=${ANALYTICS_ZOO_ROOT}
export CHRONOS_HOME=${ANALYTICS_ZOO_ROOT}/python/chronos/src
export CHRONOS_BENCHMARK_DIR=${ANALYTICS_ZOO_ROOT}/python/chronos/benchmark

set -e
echo "# Start testing"
start=$(date "+%s")

# Boot-up commands 
# e.g. python benchmark_program.py 
bash $CHRONOS_BENCHMARK_DIR/tsdataset_processing/tsdataset_processing_benchmark.sh
#

now=$(date "+%s")
time=$((now-start))
echo ">> All Benchmark test finished"
echo ">> Time used:$time sec"
