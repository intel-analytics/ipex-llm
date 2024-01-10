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
from harness_to_leaderboard import task_to_metric


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Model", "Precision", "Task", "Metric", "Value"]
    latex_writer.headers = ["Model", "Precision", "Task", "Metric", "Value"]

    values = []
    for model, model_results in result_dict.items():
        for precision, prec_results in model_results.items():
            for task, task_results in prec_results.items():

                results = task_results["results"]
                m = task_to_metric[task]
                if len(results) > 1:
                    result = results[task]
                else:
                    result = list(results.values())[0]

                values.append([model, precision, task, m, "%.2f" % (result[m] * 100)])

                model = ""    
                precision = ""
        
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


if __name__ == "__main__":

    # loop dirs and subdirs in results dir
    # for each dir, load json files
    merged_results = dict()
    for dirpath, dirnames, filenames in os.walk(sys.argv[1]):
        # skip dirs without files
        if not filenames:
            continue
        for filename in sorted([f for f in filenames if f.endswith(".json")]):
            path = os.path.join(dirpath, filename)
            model, device, precision, task = dirpath.split('/')[-4:]
            with open(path, "r") as f:
                result_dict = json.load(f)
            if model not in merged_results:
                merged_results[model] = dict()
            if precision not in merged_results[model]:
                merged_results[model][precision] = dict()
            merged_results[model][precision][task] = result_dict
    print(make_table(merged_results))
