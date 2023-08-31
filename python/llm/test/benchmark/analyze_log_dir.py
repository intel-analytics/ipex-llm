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
import csv
import os
import re
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OPT generation script', add_help=False)
    parser.add_argument('-m', '--log-dir',  default="./", type=str)
    parser.add_argument('--output-path',
                        default="./model_latency.csv", type=str)
    args = parser.parse_args()
    print(args)
    result_list = []
    for filename in glob.glob(os.path.join(args.log_dir, '*')):
        try:
            basename = os.path.basename(filename)
            name, _ = os.path.splitext(basename)
            model_name, prompt_length, output_length = name.strip().split('-')
            with open(filename, 'r', encoding='utf-8') as f:
                log = f.read()
            first_token_time_list = sorted(map(float,
                                               re.findall(r'First token cost (.*?)s', log)))
            rest_token_time_list = sorted(map(float,
                                              re.findall(r'Rest tokens cost average (.*?)s', log)))
            first_token_latency = sum(first_token_time_list[1:-1]
                                      )/(len(first_token_time_list)-2)
            rest_token_latency = sum(rest_token_time_list[1:-1]
                                     )/(len(rest_token_time_list)-2)
            result_list += {
                'model_name': model_name,
                'prompt_length': int(prompt_length),
                'output_length': int(output_length),
                'first_token_latency': first_token_latency,
                'rest_token_latency': rest_token_latency,
            }
        except Exception as e:
            print(e.args)
            continue

    with open(args.output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result_list[0].keys())
        writer.writeheader()
        writer.writerows(result_list)
    print('Log analyze finished!')
