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
   python make_csv.py <input_dir> <output_dir>
"""

import logging
from pytablewriter import MarkdownTableWriter, LatexTableWriter
import os
import json
import sys
import csv
import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_csv(result_dict, output_path=None):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f'results_{current_date}.csv'
    full_path = os.path.join(output_path, file_name) if output_path else file_name
    file_name = full_path
    headers = ["Index", "Model", "Precision", "ppl_result"]
    
    with open(file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        index = 0
        for model, model_results in result_dict.items():
            for precision, prec_results in model_results.items():
                row = [index, model, precision]
                for language in ["en","zh"]:
                    task_results = prec_results.get(language.lower(), None)
                    if task_results is None:
                        continue
                    else:
                        result = task_results["results"]
                        row.append("%.4f" % result)
                writer.writerow(row)
                index += 1


def merge_results(path):
    # loop dirs and subdirs in results dir
    # for each dir, load json files
    print('Read from', path)
    merged_results = dict()
    for dirpath, dirnames, filenames in os.walk(path):
        # skip dirs without files
        if not filenames:
            continue
        for filename in sorted([f for f in filenames if f.endswith("result.json")]):
            path = os.path.join(dirpath, filename)
            model, device, precision, language = dirpath.split('/')[-4:]
            with open(path, "r") as f:
                result_dict = json.load(f)
            if model not in merged_results:
                merged_results[model] = dict()
            if precision not in merged_results[model]:
                merged_results[model][precision] = dict()
            merged_results[model][precision][language] = result_dict
    return merged_results


def main(*args):
    assert len(args) > 2, \
    """Usage:
        python make_csv.py <input_dir> <output_dir>
    """

    input_path = args[1]
    output_path = args[2]

    merged_results = merge_results(input_path)

    make_csv(merged_results, output_path)


if __name__ == "__main__":
    # when running from the harness, the first argument is the script name
    # you must name the second argument and the third argument(optional) to be the input_dir and output_dir
    main(*sys.argv)
