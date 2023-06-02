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

import argparse
import os
import subprocess

LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs")


def llama(args):
    if os.name == 'nt':
        exec_path = os.path.join(LIB_DIR, 'main-llama.exe')
    else:
        exec_path = os.path.join(LIB_DIR, 'main-llama')
    command = [exec_path,
               '-t', args.threads,
               '-p', args.prompt,
               '-n', args.n_predict,
               '-m', args.model]
    command = list(map(lambda x: str(x), command))
    subprocess.run(command, check=True)


def bloomz(args):
    if os.name == 'nt':
        exec_path = os.path.join(LIB_DIR, 'main-bloomz.exe')
    else:
        exec_path = os.path.join(LIB_DIR, 'main-bloomz')
    command = [exec_path,
               '-t', args.threads,
               '-p', args.prompt,
               '-n', args.n_predict,
               '-m', args.model]
    command = list(map(lambda x: str(x), command))
    subprocess.run(command, check=True)


def redpajama(args):
    if os.name == 'nt':
        exec_path = os.path.join(LIB_DIR, 'main-gptneox.exe')
    else:
        exec_path = os.path.join(LIB_DIR, 'main-gptneox')
    command = [exec_path,
               '-t', args.threads,
               '-p', args.prompt,
               '-n', args.n_predict,
               '-m', args.model]
    command = list(map(lambda x: str(x), command))
    subprocess.run(command, check=True)


model_exec_map = {
    'llama': llama,
    'bloomz': bloomz,
    'redpajama': redpajama
}


def get_model_family(args):
    # TODO: get model family by model path content
    global model_exec_map
    model_basename = os.path.basename(args.model)
    for model_family in model_exec_map.keys():
        if model_family in model_basename:
            return model_family
    model_dirname = os.path.basename(os.path.dirname(args.model))
    for model_family in model_exec_map.keys():
        if model_family in model_dirname:
            return model_family
    return 'llama'


def main():
    parser = argparse.ArgumentParser(description='Command line wrapper')

    parser.add_argument('-t', '--threads', type=int, default=28,
                        help='number of threads to use during computation (default: 28)')
    parser.add_argument('-p', '--prompt', type=str, default='empty',
                        help='prompt to start generation with (default: empty)')
    parser.add_argument('-n', '--n_predict', type=int, default=-1,
                        help='number of tokens to predict (default: -1, -1 = infinity)')
    parser.add_argument('-m', '--model', type=str,
                        default='llama.cpp/models/lamma-7B/ggml-model.bin',
                        help='model path (default: llama.cpp/models/lamma-7B/ggml-model.bin)')
    parser.add_argument('--model_family', type=str, default='',
                        choices=['llama', 'bloomz', 'redpajama'],
                        help='family name of model')

    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    if args.model_family:
        model_exec_map[args.model_family](args)
    else:
        model_exec_map[get_model_family(args)](args)


if __name__ == '__main__':
    main()
