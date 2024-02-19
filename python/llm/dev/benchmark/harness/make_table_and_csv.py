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
"""
Usage:
   python make_table_results.py <input_dir>
"""

import logging
from pytablewriter import MarkdownTableWriter, LatexTableWriter
import os
import json
import sys
import csv
import datetime
from harness_to_leaderboard import task_to_metric


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Model", "Precision", "Arc", "Hellaswag", "MMLU", "TruthfulQA","Winogrande", "GSM8K"]
    latex_writer.headers = ["Model", "Precision", "Arc", "Hellaswag", "MMLU", "TruthfulQA","Winogrande", "GSM8K"]

    tasks = ["arc", "hellaswag", "mmlu", "truthfulqa", "winogrande", "gsm8k"]
    values = []
    for model, model_results in result_dict.items():
        for precision, prec_results in model_results.items():
            value = [model, precision]
            for task in tasks:

                task_results = prec_results.get(task, None)
                if task_results is None:
                    value.append("")
                else:
                    m = task_to_metric[task]
                    results = task_results["results"]
                    if len(results) > 1:
                        result = results[task]
                    else:
                        result = list(results.values())[0]
                    value.append("%.2f" % (result[m] * 100))
            values.append(value)
            model = ""    
            precision = ""
        
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()

def make_csv(result_dict, output_path=None):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f'results_{current_date}.csv'
    full_path = os.path.join(output_path, file_name) if output_path else file_name
    print('Writing to', full_path)
    file_name = full_path
    headers = ["Index", "Model", "Precision", "Arc", "TruthfulQA", "Winogrande"]
    
    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        index = 0
        for model, model_results in result_dict.items():
            for precision, prec_results in model_results.items():
                row = [index, model, precision]
                for task in headers[3:]:
                    task_results = prec_results.get(task.lower(), None)
                    if task_results is None:
                        row.append("")
                    else:
                        m = task_to_metric[task.lower()]
                        results = task_results["results"]
                        result = list(results.values())[0] if len(results) == 1 else results[task.lower()]
                        row.append("%.2f" % (result[m] * 100))
                writer.writerow(row)
                index += 1


def merge_results(path):
    # loop dirs and subdirs in results dir
    # for each dir, load json files
    print('Read from', path)
    merged_results = dict()
    for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
        # skip dirs without files
        if not filenames:
            continue
        for filename in sorted([f for f in filenames if f.endswith("result.json")]):
            path = os.path.join(dirpath, filename)
            model, device, precision, task = dirpath.split('/')[-4:]
            with open(path, "r") as f:
                result_dict = json.load(f)
            if model not in merged_results:
                merged_results[model] = dict()
            if precision not in merged_results[model]:
                merged_results[model][precision] = dict()
            merged_results[model][precision][task] = result_dict
    return merged_results


def main(*args):
    if len(args) > 1:
        input_path = args[1]
    else:
        raise ValueError("Input path is required")
    
    if len(args) > 2:
        output_path = args[2]  # use the third argument as the output path
    else:
        output_path = "./"  # default to current directory

    merged_results = merge_results(input_path)
    make_csv(merged_results, output_path)
    print(make_table(merged_results))


if __name__ == "__main__":
    # when running from the harness, the first argument is the script name
    # you must name the second argument and the third argument(optional) to be the input_dir and output_dir
    main(*sys.argv)
