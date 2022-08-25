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

import psutil
import platform
import subprocess
import os
import warnings
# Filter out useless Userwarnings
warnings.filterwarnings('ignore', category=UserWarning)
import sys
from typing import Union, Dict, List, Optional
import numpy as np
import torch
import pandas as pd

from bigdl.nano.common.cpu_schedule import schedule_workers
from bigdl.nano.common.common import _env_variable_is_set, _find_library
from bigdl.chronos.data import TSDataset
from bigdl.chronos.data.repo_dataset import get_public_dataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.chronos.forecaster.lstm_forecaster import LSTMForecaster
from bigdl.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
from bigdl.chronos.forecaster.tcn_forecaster import TCNForecaster
from bigdl.chronos.forecaster.autoformer_forecaster import AutoformerForecaster


def get_bytesize(bytes):
    """
    Scale bytes to its proper format ( B / KB / MB / GB / TB / PB )
    """
    factor = 1024
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if bytes < factor:
            return str(format(bytes,'.2f')) + unit
        bytes /= factor

def _find_path(path_name: str) -> bool:
    """
    Find whether .so files exist under the paths or not. This function will search the path one by one,
    and confirm whether libiomp5.so and libtcmalloc.so exist or not. If .so files can be found, return 
    True. Otherwise, return False.
    :param path_name: These paths to be found.
    :return: True(.so files can be found) or False(not all files can be found)
    """

    path_list = path_name.split(" ")
    libiomp5_flag = 0
    libtcmalloc_flag = 0

    for ipath in path_list:
        if os.path.exists(ipath):
            if ipath.endswith('libiomp5.so'):
                libiomp5_flag = 1
            elif ipath.endswith('libtcmalloc.so'):
                libtcmalloc_flag = 1

    return True if libiomp5_flag and libtcmalloc_flag else False

def get_nano_env_var(use_malloc: str = "tc", use_openmp: bool = True,
                     print_environment: bool = False) -> Dict[str, str]:
    """
    Return proper environment variables for jemalloc and openmp libraries.
    :param use_malloc: Allocator to be chosen, either "je" for jemalloc or "tc" for tcmalloc.
        default as tcmalloc.
    :param use_openmp: If this is set to True, then use intel openmp library. Otherwise disable
        openmp and related environment variables.
    :param print_environment: If this is set to True, print all environment variables after
        setting.
    :return: Dict[str, str], indicates the key-value map of environment variables to be set by
             nano.
    """

    # Get a copy of os environment
    env_copy = os.environ.copy()
    nano_env = {}

    # Find conda directory
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'"
                                            "| sed 's/.*:.//g'",
                                            shell=True).splitlines()[0].decode("utf-8")
    except subprocess.CalledProcessError:
        warnings.warn("Conda is not found on your computer.")

    conda_lib_dir = conda_dir + "/lib" if conda_dir is not None else None
    openmp_lib_dir = _find_library("libiomp5.so", conda_lib_dir)
    jemalloc_lib_dir = _find_library("libjemalloc.so", conda_lib_dir)
    tc_malloc_lib_dir = _find_library("libtcmalloc.so", conda_lib_dir)
    ld_preload_list = []

    # Intel OpenMP library
    if openmp_lib_dir is not None:
        ld_preload_list.append(openmp_lib_dir)

        # Detect number of physical cores
        cpu_procs = schedule_workers(1)
        num_threads = len(cpu_procs[0])

        # Set environment variables
        nano_env["OMP_NUM_THREADS"] = str(num_threads)
        nano_env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        nano_env["KMP_BLOCKTIME"] = "1"
    else:
        warnings.warn("Intel OpenMP library (libiomp5.so) is not found.")

    # jemalloc library
    if jemalloc_lib_dir is not None:
        ld_preload_list.append(jemalloc_lib_dir)

        nano_env["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,"\
                "metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
    else:
        warnings.warn("jemalloc library (libjemalloc.so) is nor found.")

    if tc_malloc_lib_dir is not None:
        ld_preload_list.append(tc_malloc_lib_dir)
    else:
        warnings.warn("tcmalloc library (libtcmalloc.so) is nor found.")

    # Disable openmp or jemalloc according to options
    if not use_openmp:
        nano_env.pop("OMP_NUM_THREADS")
        nano_env.pop("KMP_AFFINITY")
        nano_env.pop("KMP_BLOCKTIME")
        ld_preload_list = [lib for lib in ld_preload_list if "libiomp5.so" not in lib]

    if use_malloc is not "je":
        nano_env.pop("MALLOC_CONF")
        ld_preload_list = [lib for lib in ld_preload_list if "libjemalloc.so" not in lib]

    if use_malloc is not "tc":
        ld_preload_list = [lib for lib in ld_preload_list if "libtcmalloc.so" not in lib]
    
    # Set LD_PRELOAD
    nano_env["LD_PRELOAD"] = " ".join(ld_preload_list)

    # TF support
    nano_env["TF_ENABLE_ONEDNN_OPTS"] = "1"
    nano_env["ENABLE_TF_OPTS"] = "1"
    nano_env["NANO_TF_INTER_OP"] = "1"
    
    if print_environment:
        print(nano_env)

    return nano_env

def CPU_info():
    """
    Capture hardware information, such as CPU model, CPU informations, memory status
    """

    # information about CPU
    socket_num = int(subprocess.getoutput('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
    model_name = subprocess.getoutput('lscpu | grep "Model name"')
    model_name = model_name.partition(":")[2]

    print(">"*20,"Hardware Information",">"*20)
    print('CPU architecture:',platform.processor())
    print('CPU model name:',model_name.lstrip())
    print('Logical Core(s):',psutil.cpu_count())
    print('Physical Core(s):',psutil.cpu_count(logical=False))
    print('Physical Core(s) per socket:',int(psutil.cpu_count(logical=False)/socket_num))
    print('Socket(s):',socket_num)
    print('CPU usage:',str(psutil.cpu_percent()) + '%')
    print('CPU MHz:',format(psutil.cpu_freq().current,'.2f'))
    print('CPU max MHz:',format(psutil.cpu_freq().max,'.2f'))
    print('CPU min MHz:',format(psutil.cpu_freq().min,'.2f'))
    print('Total memory:',get_bytesize(psutil.virtual_memory().total))
    print('Available memory:',get_bytesize(psutil.virtual_memory().available))

    # Support instruction set or not
    disabled_logo = "\033[0;31m\u2718\033[0m"
    abled_logo = "\033[0;32m\u2714\033[0m"

    for flag in ["avx512f", "avx512_bf16", "avx512_vnni"]:
        flag_enabled = int(subprocess.getoutput(f'lscpu | grep -c {flag} '))
        if flag_enabled:
            print("Support", flag, ":", abled_logo)
        else:
            print("Support", flag, ":", disabled_logo)

    print("<"*20, "Hardware Information", "<"*20, "\n")

def check_nano(use_malloc: str = "tc", use_openmp: bool = True) -> None:
    """
    Check whether necessary environment variables are setted properly
    """
    # Get a copy of os environment
    env_copy = os.environ.copy()
    # Get the proper environment
    correct_env = get_nano_env_var()

    # Flags about the environment values are proper or not
    flag = {"LD_PRELOAD" : 1, "tcmalloc" : 1, "Intel OpenMp" : 1, "TF" : 1}

    # Output information
    name = {"LD_PRELOAD" : "", "tcmalloc" : "", "Intel OpenMp" : ": ", "TF" : ": "}
    output_list = []

    # Find conda directory
    conda_dir = None
    try:
        conda_dir = subprocess.check_output("conda info | awk '/active env location/'"
                                            "| sed 's/.*:.//g'",
                                            shell=True).splitlines()[0].decode("utf-8")
        conda_env_name = conda_dir.split("/")[-1]
    except subprocess.CalledProcessError:
        warnings.warn("Conda is not found on your computer.")

    conda_lib_dir = conda_dir + "/lib" if conda_dir is not None else None
    openmp_lib_dir = _find_library("libiomp5.so", conda_lib_dir)
    jemalloc_lib_dir = _find_library("libjemalloc.so", conda_lib_dir)
    tc_malloc_lib_dir = _find_library("libtcmalloc.so", conda_lib_dir)

    # Detect Intel OpenMP library
    if use_openmp:
        if openmp_lib_dir is not None:
            # Check environment variables
            for var in ["OMP_NUM_THREADS", "KMP_AFFINITY", "KMP_BLOCKTIME"]:
                if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
                    flag["Intel OpenMp"] = 0
                    name["Intel OpenMp"] = name["Intel OpenMp"] + var + " "
                    output_list.append("export " + var + "=" + correct_env[var])
        else:
            output_list.append("Intel OpenMP library (libiomp5.so) is not found.")

    # Detect jemalloc library
    if use_malloc is "je":
        if jemalloc_lib_dir is not None:
            if not _env_variable_is_set("MALLOC_CONF", env_copy) or env_copy["MALLOC_CONF"] != correct_env["MALLOC_CONF"]:
                output_list.append("export MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1")
        else:
            output_list.append("jemalloc library (libjemalloc.so) is not found.")

    if use_malloc is "tc":
        if tc_malloc_lib_dir is None:
            flag["tcmalloc"] = 0
            output_list.append("tcmalloc library (libtcmalloc.so) is not found.")

    # Check TF_support
    for var in ["TF_ENABLE_ONEDNN_OPTS", "ENABLE_TF_OPTS", "NANO_TF_INTER_OP"]:
        if not _env_variable_is_set(var, env_copy) or env_copy[var] != correct_env[var]:
            flag["TF"] = 0
            name["TF"] = name["TF"] + var + " "
            output_list.append("export " + var + "=" + correct_env[var])

    # Check LD_PRELOAD
    if not _env_variable_is_set("LD_PRELOAD", env_copy) or not _find_path(env_copy["LD_PRELOAD"]):
        flag["LD_PRELOAD"] = 0
        output_list.append("export LD_PRELOAD=" + correct_env["LD_PRELOAD"])

    # Output overview
    print(">"*20,"Environment Variables",">"*20)
    disabled_logo = "\033[0;31mnot enabled \033[0m" + "\033[0;31m\u2718\033[0m"
    abled_logo = "\033[0;32m enabled \033[0m" + "\033[0;32m\u2714\033[0m"
    for category in ["LD_PRELOAD", "tcmalloc", "Intel OpenMp", "TF"]:
        if flag[category]  == 0:
            print(category, name[category], disabled_logo)
        else:
            print(category, abled_logo)

    # Output suggestions
    if output_list != []:
        print(" ")
        print("+" * 20,"Suggested change: ", "+" * 20)
        for info in output_list:
            print(info)
        print("+" * 60, "\n")
    print("<"*20, "Environment Variables", "<"*20, "\n")

def test_run(model_name, stage, lookback, horizon):
    MODEL_FORECASTER_MAP = {
    "LSTM": LSTMForecaster,
    "Seq2Seq": Seq2SeqForecaster,
    "TCN": TCNForecaster,
    "Autoformer": AutoformerForecaster
    }
    forecaster_func = MODEL_FORECASTER_MAP[model_name]

    # nyc_taxi dataset
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='nyc_taxi', with_split=True,
                                                               val_ratio=0.1, test_ratio=0.1)
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.impute(mode="linear")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))

    if model_name == "Autoformer":
        label_len = (int)(lookback/2)
        train_loader = tsdata_train.to_torch_data_loader(roll=True, lookback=lookback, horizon=horizon,
                                                         time_enc=True, label_len=label_len)
        test_loader = tsdata_test.to_torch_data_loader(roll=True, lookback=lookback, horizon=horizon,
                                                       time_enc=True, label_len=label_len, is_predict=True,
                                                       shuffle=False, batch_size=1)
        forecaster = forecaster_func(past_seq_len = lookback,
                                     future_seq_len = horizon,
                                     input_feature_num = 1,
                                     output_feature_num = 1,
                                     label_len = label_len,
                                     freq = 't')
    else:
        train_loader = tsdata_train.to_torch_data_loader(roll=True, lookback=lookback, horizon=horizon)
        test_loader = tsdata_test.to_torch_data_loader(batch_size=1, roll=True, lookback=lookback, horizon=horizon, shuffle=False)
        if model_name == "LSTM":
            forecaster = forecaster_func.from_tsdataset(tsdata_train)
        else:
            forecaster = forecaster_func.from_tsdataset(tsdata_train, past_seq_len = lookback, future_seq_len = horizon)

    forecaster.num_processes = 1
    forecaster.fit(train_loader, epochs=1, batch_size=32)

    torch.set_num_threads(1)
    if model_name == "Autoformer":
        x = next(iter(test_loader))
        latency1 = Evaluator.get_latency(forecaster.internal.predict_step, *(x, 0), num_running = len(tsdata_test.df))
        return latency1, []
    else:
        x = next(iter(test_loader))[0]
        latency1 = Evaluator.get_latency(forecaster.predict, x.numpy(), num_running = len(tsdata_test.df))
        forecaster.build_onnx(thread_num = 1)
        latency2 = Evaluator.get_latency(forecaster.predict_with_onnx,x.numpy(), num_running = len(tsdata_test.df))
        return latency1, latency2
