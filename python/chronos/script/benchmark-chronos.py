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

import sys
import getopt
import numpy as np

from utils import CPU_info, check_nano, test_run
from bigdl.nano.utils.log4Error import invalidInputError


if __name__ == "__main__":
    # available options
    model_list = ["LSTM", "Seq2Seq", "TCN", "Autoformer"]
    stage_list = ["latency"]
    opt_list = []

    opts, args = getopt.getopt(sys.argv[1:], "m:s:", ["model=", "stage="])
    for opt, arg in opts:
        opt_list.append(opt)
        if opt in ['-m', '--model']:
            model_name = arg
        elif opt in ['-s', '--stage']:
            stage = arg

    invalidInputError('-m' in opt_list or '--model' in opt_list,
                      "-m/--model option must be provided but get none.")
    invalidInputError(model_name in model_list, f"model argument should be one of {model_list},"
                      f" but get {model_name}.")
    invalidInputError('-s' in opt_list or '--stage' in opt_list,
                      "-s/--stage option must be provided but get none.")
    invalidInputError(stage in stage_list, f"stage argument should be one of {stage_list},"
                      f" but get {stage}.")

    latency1, latency2 = test_run(model_name, stage, lookback=96, horizon=48)

    # output information
    print("-"*50, "short summary", "-"*50, '\n')
    CPU_info()
    check_nano()
    print(">" * 20, "nyc_taxi test-run result", ">" * 20)
    for info in ["p50", "p90", "p95", "p99"]:
        print(info, "latency:", latency1[info], "ms")
    if model_name != "Autoformer":
        for info in ["p50", "p90", "p95", "p99"]:
            print(info, "latency with onnx:", latency2[info], "ms")
    print("<" * 20, "nyc_taxi test-run result", "<" * 20)
